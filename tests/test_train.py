"""Tests for training entrypoints."""

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
            lambda model_config: (model, MockTokenizer()),
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
