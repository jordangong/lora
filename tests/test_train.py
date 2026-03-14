"""Tests for training entrypoints."""

from contextlib import contextmanager

from datasets import Dataset, DatasetDict
from torch import nn

import lora_finetune.train as train_module
from lora_finetune.config import Config, DataConfig, ModelConfig, TrainingConfig


class TestTrainLlm:
    def test_train_llm_uses_trl_native_dataset_for_conversational_data_when_eos_append_disabled(
        self, monkeypatch
    ):
        model = nn.Linear(10, 5)
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "messages": [
                            [
                                {"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "hello"},
                            ]
                        ]
                    }
                )
            }
        )
        config = Config(
            model=ModelConfig(model_type="causal_lm"),
            data=DataConfig(append_eos_token=False, preprocessing_num_workers=1),
            training=TrainingConfig(
                output_dir="./test-output",
                eval_strategy="no",
                load_best_model_at_end=False,
                report_to="none",
                llm_trainer="trl",
            ),
        )

        class MockTokenizer:
            pad_token_id = None

        class FakeTrainer:
            def __init__(self):
                self.train_calls = []

            def train(self, resume_from_checkpoint=None):
                self.train_calls.append(resume_from_checkpoint)

            def save_model(self):
                return None

            def push_to_hub(self):
                return None

        trainer = FakeTrainer()
        prepare_calls = []
        preprocess_calls = []
        create_trainer_calls = []

        def fake_prepare_text_dataset_for_trl(raw_dataset, data_config, shuffle_seed=None):
            prepare_calls.append((raw_dataset, data_config, shuffle_seed))
            return raw_dataset

        def fake_preprocess_text_dataset(raw_dataset, tokenizer, data_config, shuffle_seed=None):
            preprocess_calls.append((raw_dataset, tokenizer, data_config, shuffle_seed))
            return raw_dataset

        def fake_create_trainer(**kwargs):
            create_trainer_calls.append(kwargs)
            return trainer

        monkeypatch.setattr(
            train_module,
            "load_model_and_tokenizer",
            lambda model_config, max_seq_length=None: (model, MockTokenizer()),
        )
        monkeypatch.setattr(train_module, "get_llm_target_modules", lambda model_name: ["q_proj"])
        monkeypatch.setattr(train_module, "get_peft_model_with_lora", lambda *args, **kwargs: model)
        monkeypatch.setattr(
            train_module, "prepare_model_for_training", lambda *args, **kwargs: model
        )
        monkeypatch.setattr(train_module, "print_model_size", lambda *args, **kwargs: None)
        monkeypatch.setattr(train_module, "load_text_dataset", lambda data_config: dataset)
        monkeypatch.setattr(
            train_module,
            "prepare_text_dataset_for_trl",
            fake_prepare_text_dataset_for_trl,
        )
        monkeypatch.setattr(
            train_module,
            "preprocess_text_dataset",
            fake_preprocess_text_dataset,
        )
        monkeypatch.setattr(train_module, "create_trainer", fake_create_trainer)

        train_module.train_llm(config)

        assert len(prepare_calls) == 1
        assert len(preprocess_calls) == 0
        assert len(create_trainer_calls) == 1
        assert create_trainer_calls[0]["train_dataset"].column_names == ["messages"]
        assert create_trainer_calls[0]["data_collator"] is None
        assert trainer.train_calls == [None]

    def test_train_llm_forwards_unsloth_arguments_to_model_loading_and_peft(self, monkeypatch):
        model = nn.Linear(10, 5)
        dataset = DatasetDict({"train": Dataset.from_dict({"text": ["hello"]})})
        config = Config(
            model=ModelConfig(model_type="causal_lm", use_unsloth=True, load_in_4bit=True),
            data=DataConfig(max_seq_length=4096, preprocessing_num_workers=1),
            training=TrainingConfig(
                output_dir="./test-output",
                eval_strategy="no",
                load_best_model_at_end=False,
                report_to="none",
                llm_trainer="transformers",
                gradient_checkpointing=False,
                seed=123,
            ),
        )

        class MockTokenizer:
            pad_token_id = 0

        class FakeTrainer:
            def train(self, resume_from_checkpoint=None):
                return None

            def save_model(self):
                return None

            def push_to_hub(self):
                return None

        load_calls = []
        peft_calls = []

        def fake_load_model_and_tokenizer(model_config, max_seq_length=None):
            load_calls.append((model_config, max_seq_length))
            return model, MockTokenizer()

        def fake_get_peft_model_with_lora(*args, **kwargs):
            peft_calls.append(kwargs)
            return model

        monkeypatch.setattr(train_module, "load_model_and_tokenizer", fake_load_model_and_tokenizer)
        monkeypatch.setattr(train_module, "get_llm_target_modules", lambda model_name: ["q_proj"])
        monkeypatch.setattr(train_module, "get_peft_model_with_lora", fake_get_peft_model_with_lora)
        monkeypatch.setattr(
            train_module, "prepare_model_for_training", lambda *args, **kwargs: model
        )
        monkeypatch.setattr(train_module, "print_model_size", lambda *args, **kwargs: None)
        monkeypatch.setattr(train_module, "load_text_dataset", lambda data_config: dataset)
        monkeypatch.setattr(train_module, "get_text_collator", lambda tokenizer: "collator")
        monkeypatch.setattr(
            train_module,
            "preprocess_text_dataset",
            lambda raw_dataset, tokenizer, data_config, shuffle_seed=None: raw_dataset,
        )
        monkeypatch.setattr(train_module, "create_trainer", lambda **kwargs: FakeTrainer())

        train_module.train_llm(config)

        assert load_calls == [(config.model, 4096)]
        assert len(peft_calls) == 1
        assert peft_calls[0]["use_unsloth"] is True
        assert peft_calls[0]["use_gradient_checkpointing"] is False
        assert peft_calls[0]["random_state"] == 123
        assert peft_calls[0]["max_seq_length"] == 4096

    def test_train_llm_uses_runtime_capture_in_normal_mode(self, monkeypatch):
        model = nn.Linear(10, 5)
        dataset = DatasetDict({"train": Dataset.from_dict({"text": ["hello"]})})
        config = Config(
            model=ModelConfig(model_type="causal_lm"),
            data=DataConfig(max_seq_length=128, preprocessing_num_workers=1),
            training=TrainingConfig(
                output_dir="./test-output",
                eval_strategy="no",
                load_best_model_at_end=False,
                report_to="none",
                llm_trainer="transformers",
            ),
        )

        class MockTokenizer:
            pad_token_id = 0

        class FakeTrainer:
            def train(self, resume_from_checkpoint=None):
                return None

            def save_model(self):
                return None

            def push_to_hub(self):
                return None

        capture_calls = []

        @contextmanager
        def fake_capture_runtime_output(*, enabled=True, rich_console=None):
            capture_calls.append((enabled, rich_console))
            yield

        monkeypatch.setattr(
            train_module,
            "load_model_and_tokenizer",
            lambda model_config, max_seq_length=None: (model, MockTokenizer()),
        )
        monkeypatch.setattr(train_module, "get_llm_target_modules", lambda model_name: ["q_proj"])
        monkeypatch.setattr(train_module, "get_peft_model_with_lora", lambda *args, **kwargs: model)
        monkeypatch.setattr(
            train_module, "prepare_model_for_training", lambda *args, **kwargs: model
        )
        monkeypatch.setattr(train_module, "print_model_size", lambda *args, **kwargs: None)
        monkeypatch.setattr(train_module, "load_text_dataset", lambda data_config: dataset)
        monkeypatch.setattr(train_module, "get_text_collator", lambda tokenizer: "collator")
        monkeypatch.setattr(
            train_module,
            "preprocess_text_dataset",
            lambda raw_dataset, tokenizer, data_config, shuffle_seed=None: raw_dataset,
        )
        monkeypatch.setattr(train_module, "create_trainer", lambda **kwargs: FakeTrainer())
        monkeypatch.setattr(train_module, "capture_runtime_output", fake_capture_runtime_output)
        monkeypatch.setattr(train_module, "verbose_logging_enabled", lambda: False)

        train_module.train_llm(config)

        assert len(capture_calls) == 2
        assert capture_calls[0][0] is True
        assert capture_calls[1][0] is True

    def test_train_llm_disables_runtime_capture_in_verbose_mode(self, monkeypatch):
        model = nn.Linear(10, 5)
        dataset = DatasetDict({"train": Dataset.from_dict({"text": ["hello"]})})
        config = Config(
            model=ModelConfig(model_type="causal_lm"),
            data=DataConfig(max_seq_length=128, preprocessing_num_workers=1),
            training=TrainingConfig(
                output_dir="./test-output",
                eval_strategy="no",
                load_best_model_at_end=False,
                report_to="none",
                llm_trainer="transformers",
            ),
        )

        class MockTokenizer:
            pad_token_id = 0

        class FakeTrainer:
            def train(self, resume_from_checkpoint=None):
                return None

            def save_model(self):
                return None

            def push_to_hub(self):
                return None

        capture_calls = []

        @contextmanager
        def fake_capture_runtime_output(*, enabled=True, rich_console=None):
            capture_calls.append((enabled, rich_console))
            yield

        monkeypatch.setattr(
            train_module,
            "load_model_and_tokenizer",
            lambda model_config, max_seq_length=None: (model, MockTokenizer()),
        )
        monkeypatch.setattr(train_module, "get_llm_target_modules", lambda model_name: ["q_proj"])
        monkeypatch.setattr(train_module, "get_peft_model_with_lora", lambda *args, **kwargs: model)
        monkeypatch.setattr(
            train_module, "prepare_model_for_training", lambda *args, **kwargs: model
        )
        monkeypatch.setattr(train_module, "print_model_size", lambda *args, **kwargs: None)
        monkeypatch.setattr(train_module, "load_text_dataset", lambda data_config: dataset)
        monkeypatch.setattr(train_module, "get_text_collator", lambda tokenizer: "collator")
        monkeypatch.setattr(
            train_module,
            "preprocess_text_dataset",
            lambda raw_dataset, tokenizer, data_config, shuffle_seed=None: raw_dataset,
        )
        monkeypatch.setattr(train_module, "create_trainer", lambda **kwargs: FakeTrainer())
        monkeypatch.setattr(train_module, "capture_runtime_output", fake_capture_runtime_output)
        monkeypatch.setattr(train_module, "verbose_logging_enabled", lambda: True)

        train_module.train_llm(config)

        assert len(capture_calls) == 2
        assert capture_calls[0][0] is False
        assert capture_calls[1][0] is False
