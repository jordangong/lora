"""Tests for training entrypoints."""

import argparse
from types import SimpleNamespace

import pytest
import yaml
from datasets import Dataset, DatasetDict
from torch import nn

import lora_finetune.train as train_module
from lora_finetune.config import (
    BenchmarkEvalConfig,
    Config,
    DataConfig,
    DPOConfig,
    GRPOConfig,
    HPOConfig,
    LoraConfig,
    ModelConfig,
    TrainingConfig,
)


class TestTrainLlm:
    def test_save_hpo_best_config_supports_nested_best_run_hyperparameters(
        self, monkeypatch, tmp_path
    ):
        output_dir = tmp_path / "hpo-output"
        sweep_name = "llama3-lora-hpo"
        config = Config(
            training=TrainingConfig(output_dir=str(output_dir), report_to="none"),
            lora=LoraConfig(r=8),
            hpo=HPOConfig(
                enabled=True,
                sweep_name=sweep_name,
                parameters={
                    "learning_rate": {"values": [1e-5, 3e-5]},
                    "lora.r": {"values": [8, 16]},
                },
            ),
        )
        best_run = SimpleNamespace(
            hyperparameters={
                "training": {"learning_rate": 3e-5, "report_to": "wandb"},
                "lora": {"r": 16},
                "metric": "eval/loss",
            }
        )

        monkeypatch.setattr(train_module, "_ensure_runtime_imports", lambda: None)
        monkeypatch.setattr(
            train_module,
            "apply_hpo_parameters_to_config_sections",
            __import__(
                "lora_finetune.trainer", fromlist=["apply_hpo_parameters_to_config_sections"]
            ).apply_hpo_parameters_to_config_sections,
        )

        train_module._save_hpo_best_config(config, best_run)

        saved_config_path = output_dir / sweep_name / "best_hpo_config.yaml"
        assert saved_config_path.exists()
        saved_config = yaml.unsafe_load(saved_config_path.read_text())
        assert saved_config["training"]["output_dir"] == str(output_dir / sweep_name)
        assert saved_config["training"]["learning_rate"] == pytest.approx(3e-5)
        assert saved_config["training"]["report_to"] == "none"
        assert saved_config["lora"]["r"] == 16

    def test_save_hpo_best_config_does_not_double_nest_when_output_dir_already_has_sweep(
        self, monkeypatch, tmp_path
    ):
        sweep_name = "llama3-lora-hpo"
        output_dir = tmp_path / sweep_name
        config = Config(
            training=TrainingConfig(output_dir=str(output_dir), report_to="none"),
            hpo=HPOConfig(
                enabled=True,
                sweep_name=sweep_name,
                parameters={"learning_rate": {"values": [1e-5, 3e-5]}},
            ),
        )
        best_run = SimpleNamespace(hyperparameters={"training": {"learning_rate": 3e-5}})

        monkeypatch.setattr(train_module, "_ensure_runtime_imports", lambda: None)
        monkeypatch.setattr(
            train_module,
            "apply_hpo_parameters_to_config_sections",
            __import__(
                "lora_finetune.trainer", fromlist=["apply_hpo_parameters_to_config_sections"]
            ).apply_hpo_parameters_to_config_sections,
        )

        train_module._save_hpo_best_config(config, best_run)

        saved_config_path = output_dir / "best_hpo_config.yaml"
        assert saved_config_path.exists()
        assert not (output_dir / sweep_name / "best_hpo_config.yaml").exists()
        saved_config = yaml.unsafe_load(saved_config_path.read_text())
        assert saved_config["training"]["output_dir"] == str(output_dir)

    def test_run_trainer_hpo_releases_eager_model_before_search(self, monkeypatch):
        class FakeAccelerator:
            def __init__(self):
                self.free_memory_calls = 0

            def free_memory(self):
                self.free_memory_calls += 1

        trainer = SimpleNamespace(
            model_init=lambda _: nn.Linear(2, 2),
            model="eager-model",
            model_wrapped="wrapped-model",
            optimizer="optimizer",
            lr_scheduler="scheduler",
            accelerator=FakeAccelerator(),
            callback_handler=SimpleNamespace(callbacks=[]),
        )
        config = Config(
            training=TrainingConfig(output_dir="./test-output", report_to="wandb"),
            hpo=HPOConfig(enabled=True, metric_name="eval_loss"),
        )
        released = {}

        def fake_run_hyperparameter_search(current_trainer, training_config, hpo_config):
            released["trainer"] = current_trainer
            return "best-run"

        def fake_release_memory(*objects):
            released["objects"] = objects
            return [None for _ in objects]

        monkeypatch.setattr(
            train_module, "run_hyperparameter_search", fake_run_hyperparameter_search
        )
        monkeypatch.setattr(train_module, "_cleanup_trainer_callbacks", lambda trainer: None)
        monkeypatch.setitem(
            __import__("sys").modules,
            "accelerate.utils.memory",
            SimpleNamespace(release_memory=fake_release_memory),
        )

        result = train_module._run_trainer_hpo(trainer, config)

        assert result == "best-run"
        assert released["objects"] == ("wrapped-model", "eager-model")
        assert trainer.model is None
        assert trainer.model_wrapped is None
        assert trainer.optimizer is None
        assert trainer.lr_scheduler is None
        assert trainer.accelerator.free_memory_calls == 1

    def test_run_trainer_hpo_runs_final_eval_for_each_trial_and_sets_objective(self, monkeypatch):
        class FakeAccelerator:
            def __init__(self):
                self.free_memory_calls = 0

            def free_memory(self):
                self.free_memory_calls += 1

        trainer = SimpleNamespace(
            model_init=lambda _: nn.Linear(2, 2),
            model="eager-model",
            model_wrapped="wrapped-model",
            optimizer="optimizer",
            lr_scheduler="scheduler",
            accelerator=FakeAccelerator(),
            callback_handler=SimpleNamespace(callbacks=[]),
            objective=None,
        )
        final_eval_calls = []

        def add_callback(callback):
            trainer.callback_handler.callbacks.append(callback)

        trainer.add_callback = add_callback

        config = Config(
            training=TrainingConfig(output_dir="./test-output", report_to="wandb"),
            hpo=HPOConfig(enabled=True, metric_name="eval_loss"),
        )

        def fake_run_hyperparameter_search(current_trainer, training_config, hpo_config):
            for callback in list(current_trainer.callback_handler.callbacks):
                on_train_begin = getattr(callback, "on_train_begin", None)
                if callable(on_train_begin):
                    on_train_begin(None, None, "control")
            for callback in list(current_trainer.callback_handler.callbacks):
                on_train_end = getattr(callback, "on_train_end", None)
                if callable(on_train_end):
                    on_train_end(None, None, "control")
            return "best-run"

        monkeypatch.setattr(
            train_module, "run_hyperparameter_search", fake_run_hyperparameter_search
        )
        monkeypatch.setattr(train_module, "_cleanup_trainer_callbacks", lambda trainer: None)
        monkeypatch.setattr(
            train_module,
            "_run_final_trainer_evaluation",
            lambda current_trainer: (
                final_eval_calls.append(current_trainer) or {"final_eval_loss": 0.25}
            ),
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "accelerate.utils.memory",
            SimpleNamespace(release_memory=lambda *objects: [None for _ in objects]),
        )

        result = train_module._run_trainer_hpo(trainer, config)

        assert result == "best-run"
        assert final_eval_calls == [trainer]
        assert trainer.objective == pytest.approx(0.25)

    def test_run_trainer_training_runs_final_eval_before_callback_cleanup(self):
        events = []

        class CleanupCallback:
            def cleanup(self):
                events.append("cleanup")

        class FakeTrainer:
            def __init__(self):
                self.eval_dataset = object()
                self.callback_handler = SimpleNamespace(callbacks=[CleanupCallback()])

            def train(self, resume_from_checkpoint=None):
                events.append(("train", resume_from_checkpoint))

            def evaluate(self, metric_key_prefix="eval"):
                events.append(("evaluate", metric_key_prefix))
                return {"final_eval_loss": 0.1}

        trainer = FakeTrainer()

        train_module._run_trainer_training(trainer, resume_from_checkpoint="checkpoint-7")

        assert events == [
            ("train", "checkpoint-7"),
            ("evaluate", "final_eval"),
            "cleanup",
        ]

    def test_run_trainer_training_skips_final_eval_without_eval_dataset(self):
        events = []

        class CleanupCallback:
            def cleanup(self):
                events.append("cleanup")

        class FakeTrainer:
            def __init__(self):
                self.eval_dataset = None
                self.callback_handler = SimpleNamespace(callbacks=[CleanupCallback()])

            def train(self, resume_from_checkpoint=None):
                events.append(("train", resume_from_checkpoint))

            def evaluate(self, metric_key_prefix="eval"):
                events.append(("evaluate", metric_key_prefix))
                return {"final_eval_loss": 0.1}

        trainer = FakeTrainer()

        train_module._run_trainer_training(trainer)

        assert events == [
            ("train", None),
            "cleanup",
        ]

    def test_run_trainer_training_skips_final_eval_when_disabled(self):
        events = []

        class CleanupCallback:
            def cleanup(self):
                events.append("cleanup")

        class FakeTrainer:
            def __init__(self):
                self.eval_dataset = object()
                self.callback_handler = SimpleNamespace(callbacks=[CleanupCallback()])

            def train(self, resume_from_checkpoint=None):
                events.append(("train", resume_from_checkpoint))

            def evaluate(self, metric_key_prefix="eval"):
                events.append(("evaluate", metric_key_prefix))
                return {"final_eval_loss": 0.1}

        trainer = FakeTrainer()

        train_module._run_trainer_training(trainer, final_evaluation_enabled=False)

        assert events == [
            ("train", None),
            "cleanup",
        ]

    def test_run_trainer_training_runs_final_eval_from_train_end_callback_once(self):
        events = []

        class CleanupCallback:
            def cleanup(self):
                events.append("cleanup")

        class FakeTrainer:
            def __init__(self):
                self.eval_dataset = object()
                self.callback_handler = SimpleNamespace(callbacks=[CleanupCallback()])

            def add_callback(self, callback):
                self.callback_handler.callbacks.append(callback)

            def train(self, resume_from_checkpoint=None):
                events.append(("train", resume_from_checkpoint))
                for callback in list(self.callback_handler.callbacks):
                    on_train_begin = getattr(callback, "on_train_begin", None)
                    if callable(on_train_begin):
                        on_train_begin(None, None, "control")
                for callback in list(self.callback_handler.callbacks):
                    on_train_end = getattr(callback, "on_train_end", None)
                    if callable(on_train_end):
                        on_train_end(None, None, "control")

            def evaluate(self, metric_key_prefix="eval"):
                events.append(("evaluate", metric_key_prefix))
                return {"final_eval_loss": 0.1}

        trainer = FakeTrainer()

        train_module._run_trainer_training(trainer, resume_from_checkpoint="checkpoint-9")

        assert events == [
            ("train", "checkpoint-9"),
            ("evaluate", "final_eval"),
            "cleanup",
        ]

    def test_run_final_trainer_evaluation_logs_dedicated_final_namespace(self, monkeypatch):
        logged = []
        removed_callback_types = []
        restored_callbacks = []
        wandb_callback = object()

        monkeypatch.setitem(
            __import__("sys").modules,
            "wandb",
            SimpleNamespace(
                run=object(),
                log=lambda payload, step=None: logged.append((payload, step)),
            ),
        )

        class FakeTrainer:
            def __init__(self):
                self.eval_dataset = object()
                self.state = SimpleNamespace(global_step=123)

            def pop_callback(self, callback_type):
                removed_callback_types.append(callback_type.__name__)
                return wandb_callback

            def add_callback(self, callback):
                restored_callbacks.append(callback)

            def evaluate(self, metric_key_prefix="eval"):
                assert metric_key_prefix == "final_eval"
                return {
                    "final_eval_loss": 0.1,
                    "final_eval_runtime": 2.5,
                    "train_loss": 9.9,
                }

        trainer = FakeTrainer()

        metrics = train_module._run_final_trainer_evaluation(trainer)

        assert metrics == {
            "final_eval_loss": 0.1,
            "final_eval_runtime": 2.5,
            "train_loss": 9.9,
        }
        assert removed_callback_types == ["WandbCallback"]
        assert restored_callbacks == [wandb_callback]
        assert logged == [
            (
                {
                    "final/eval/loss": 0.1,
                    "final/eval/runtime": 2.5,
                    "train/global_step": 123,
                },
                123,
            )
        ]

    def test_run_benchmark_eval_logs_metrics_to_wandb_with_trainer_step(self, monkeypatch):
        logged = []

        monkeypatch.setattr(train_module, "_ensure_runtime_imports", lambda: None)
        monkeypatch.setattr(
            train_module,
            "run_lighteval",
            lambda **kwargs: {"gsm8k_0|expr_gold_metric": 0.42},
        )
        monkeypatch.setattr(train_module.console, "print", lambda *args, **kwargs: None)
        monkeypatch.setitem(
            __import__("sys").modules,
            "wandb",
            SimpleNamespace(
                run=object(), log=lambda payload, step=None: logged.append((payload, step))
            ),
        )

        trainer = SimpleNamespace(state=SimpleNamespace(global_step=321))

        train_module.run_benchmark_eval(
            model=nn.Linear(2, 2),
            model_name="test-model",
            eval_config=BenchmarkEvalConfig(enabled=True, tasks="gsm8k"),
            trainer=trainer,
        )

        assert logged == [
            (
                {
                    "final/benchmark/gsm8k_0|expr_gold_metric": 0.42,
                    "train/global_step": 321,
                },
                321,
            )
        ]

    def test_train_llm_forwards_trainer_to_post_training_benchmark_eval(self, monkeypatch):
        model = nn.Linear(10, 5)
        dataset = DatasetDict({"train": Dataset.from_dict({"text": ["hello"]})})
        config = Config(
            model=ModelConfig(model_type="causal_lm"),
            data=DataConfig(preprocessing_num_workers=1),
            training=TrainingConfig(
                output_dir="./test-output",
                eval_strategy="no",
                load_best_model_at_end=False,
                report_to="none",
                llm_trainer="transformers",
            ),
            benchmark_eval=BenchmarkEvalConfig(enabled=True, tasks="gsm8k"),
        )

        class MockTokenizer:
            pad_token_id = 0

        trainer = SimpleNamespace(state=SimpleNamespace(global_step=77))
        benchmark_calls = []
        training_calls = []
        final_eval_calls = []

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
        monkeypatch.setattr(train_module, "create_trainer", lambda **kwargs: trainer)
        monkeypatch.setattr(
            train_module,
            "_run_trainer_training",
            lambda current_trainer, resume_from_checkpoint=None, final_evaluation_enabled=True, final_evaluation_fn=None: (
                training_calls.append(
                    {
                        "trainer": current_trainer,
                        "resume_from_checkpoint": resume_from_checkpoint,
                        "final_evaluation_enabled": final_evaluation_enabled,
                        "final_evaluation_fn": final_evaluation_fn,
                    }
                )
            ),
        )
        monkeypatch.setattr(
            train_module, "_save_and_maybe_push_model", lambda current_trainer, config: None
        )
        monkeypatch.setattr(
            train_module,
            "_run_final_trainer_evaluation",
            lambda current_trainer: final_eval_calls.append(current_trainer),
        )
        monkeypatch.setattr(
            train_module,
            "run_benchmark_eval",
            lambda model, model_name, eval_config, trainer=None: benchmark_calls.append(
                {
                    "model": model,
                    "model_name": model_name,
                    "eval_config": eval_config,
                    "trainer": trainer,
                }
            ),
        )

        train_module.train_llm(config)

        assert training_calls == [
            {
                "trainer": trainer,
                "resume_from_checkpoint": None,
                "final_evaluation_enabled": True,
                "final_evaluation_fn": training_calls[0]["final_evaluation_fn"],
            }
        ]
        assert callable(training_calls[0]["final_evaluation_fn"])
        training_calls[0]["final_evaluation_fn"](trainer)
        assert final_eval_calls == []
        assert len(benchmark_calls) == 1
        assert benchmark_calls[0]["model"] is model
        assert benchmark_calls[0]["model_name"] == config.model.model_name_or_path
        assert benchmark_calls[0]["eval_config"] is config.benchmark_eval
        assert benchmark_calls[0]["trainer"] is trainer

    def test_train_llm_skips_duplicate_post_training_benchmark_eval(self, monkeypatch):
        model = nn.Linear(10, 5)
        dataset = DatasetDict({"train": Dataset.from_dict({"text": ["hello"]})})
        config = Config(
            model=ModelConfig(model_type="causal_lm"),
            data=DataConfig(preprocessing_num_workers=1),
            training=TrainingConfig(
                output_dir="./test-output",
                eval_strategy="steps",
                load_best_model_at_end=False,
                report_to="none",
                llm_trainer="transformers",
            ),
            benchmark_eval=BenchmarkEvalConfig(enabled=True, tasks="gsm8k", eval_steps=100),
        )

        class MockTokenizer:
            pad_token_id = 0

        trainer = SimpleNamespace(state=SimpleNamespace(global_step=100))
        benchmark_calls = []

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
        monkeypatch.setattr(train_module, "create_trainer", lambda **kwargs: trainer)
        monkeypatch.setattr(
            train_module,
            "_run_trainer_training",
            lambda current_trainer, resume_from_checkpoint=None, final_evaluation_enabled=True, final_evaluation_fn=None: (
                final_evaluation_fn(current_trainer)
            ),
        )
        monkeypatch.setattr(
            train_module, "_save_and_maybe_push_model", lambda current_trainer, config: None
        )
        monkeypatch.setattr(
            train_module,
            "_run_final_trainer_evaluation",
            lambda current_trainer: None,
        )
        monkeypatch.setattr(
            train_module,
            "run_benchmark_eval",
            lambda model, model_name, eval_config, trainer=None: benchmark_calls.append(
                {
                    "model": model,
                    "model_name": model_name,
                    "eval_config": eval_config,
                    "trainer": trainer,
                }
            ),
        )

        train_module.train_llm(config)

        assert benchmark_calls == []

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

    def test_train_llm_bootstraps_unsloth_before_runtime_imports(self, monkeypatch):
        events = []
        config = Config(
            model=ModelConfig(model_type="causal_lm", use_unsloth=True),
            training=TrainingConfig(output_dir="./test-output", eval_strategy="no"),
        )

        monkeypatch.setattr(
            "lora_finetune._optional_unsloth.ensure_unsloth_imported",
            lambda: events.append("bootstrap"),
        )
        monkeypatch.setattr(
            train_module, "_ensure_runtime_imports", lambda: events.append("runtime")
        )
        monkeypatch.setattr(
            train_module.imported_train_modes,
            "train_llm",
            lambda cfg, deps, logger: events.append("train_llm"),
        )

        train_module.train_llm(config)

        assert events == ["bootstrap", "runtime", "train_llm"]

    def test_train_llm_cleans_up_callbacks_on_keyboard_interrupt(self, monkeypatch):
        model = nn.Linear(10, 5)
        dataset = DatasetDict({"train": Dataset.from_dict({"text": ["hello"]})})
        config = Config(
            model=ModelConfig(model_type="causal_lm"),
            data=DataConfig(preprocessing_num_workers=1),
            training=TrainingConfig(
                output_dir="./test-output",
                eval_strategy="no",
                load_best_model_at_end=False,
                report_to="none",
                llm_trainer="transformers",
            ),
        )

        cleanup_calls = []

        class MockTokenizer:
            pad_token_id = 0

        class CleanupCallback:
            def cleanup(self):
                cleanup_calls.append("cleanup")

        class FakeCallbackHandler:
            def __init__(self):
                self.callbacks = [CleanupCallback()]

        class FakeTrainer:
            def __init__(self):
                self.callback_handler = FakeCallbackHandler()
                self.save_model_calls = 0

            def train(self, resume_from_checkpoint=None):
                raise KeyboardInterrupt

            def save_model(self):
                self.save_model_calls += 1

            def push_to_hub(self):
                return None

        trainer = FakeTrainer()

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
        monkeypatch.setattr(train_module, "create_trainer", lambda **kwargs: trainer)

        with pytest.raises(KeyboardInterrupt):
            train_module.train_llm(config)

        assert cleanup_calls == ["cleanup"]
        assert trainer.save_model_calls == 0

    def test_train_llm_uses_preference_dataset_path_for_dpo(self, monkeypatch):
        model = nn.Linear(10, 5)
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "prompt": ["Question: hi\n\nAnswer: "],
                        "chosen": ["hello"],
                        "rejected": ["goodbye"],
                    }
                )
            }
        )
        config = Config(
            model=ModelConfig(model_type="causal_lm"),
            data=DataConfig(preprocessing_num_workers=1),
            training=TrainingConfig(
                output_dir="./test-output",
                eval_strategy="no",
                load_best_model_at_end=False,
                report_to="none",
                llm_trainer="trl",
                trainer_type="dpo",
            ),
            dpo=DPOConfig(beta=0.2),
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

        preference_calls = []
        preprocess_calls = []
        create_trainer_calls = []

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
            "prepare_preference_dataset_for_trl",
            lambda raw_dataset, data_config, shuffle_seed=None: (
                preference_calls.append((raw_dataset, data_config, shuffle_seed)) or raw_dataset
            ),
        )
        monkeypatch.setattr(
            train_module,
            "preprocess_text_dataset",
            lambda raw_dataset, tokenizer, data_config, shuffle_seed=None: (
                preprocess_calls.append((raw_dataset, tokenizer, data_config, shuffle_seed))
                or raw_dataset
            ),
        )
        monkeypatch.setattr(
            train_module,
            "create_trainer",
            lambda **kwargs: create_trainer_calls.append(kwargs) or FakeTrainer(),
        )

        train_module.train_llm(config)

        assert len(preference_calls) == 1
        assert len(preprocess_calls) == 0
        assert create_trainer_calls[0]["dpo_config"].beta == 0.2
        assert create_trainer_calls[0]["grpo_config"] == config.grpo

    def test_train_llm_uses_grpo_dataset_path_for_grpo(self, monkeypatch):
        model = nn.Linear(10, 5)
        dataset = DatasetDict(
            {"train": Dataset.from_dict({"text": ["Solve 1+1"], "answer": ["2"]})}
        )
        config = Config(
            model=ModelConfig(model_type="causal_lm"),
            data=DataConfig(text_column="text", preprocessing_num_workers=1),
            training=TrainingConfig(
                output_dir="./test-output",
                eval_strategy="no",
                load_best_model_at_end=False,
                report_to="none",
                llm_trainer="trl",
                trainer_type="grpo",
            ),
            grpo=GRPOConfig(reward_funcs=["exact_match"], reward_column="answer"),
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

        grpo_calls = []
        create_trainer_calls = []

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
            "prepare_grpo_dataset_for_trl",
            lambda raw_dataset, data_config, shuffle_seed=None: (
                grpo_calls.append((raw_dataset, data_config, shuffle_seed)) or raw_dataset
            ),
        )
        monkeypatch.setattr(
            train_module,
            "create_trainer",
            lambda **kwargs: create_trainer_calls.append(kwargs) or FakeTrainer(),
        )

        train_module.train_llm(config)

        assert len(grpo_calls) == 1
        assert create_trainer_calls[0]["grpo_config"].reward_funcs == ["exact_match"]


class TestTrainTextClassification:
    def test_train_text_classification_uses_classification_pipeline(self, monkeypatch):
        model = nn.Linear(10, 2)
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict({"sentence": ["good", "bad"], "label": [1, 0]}),
                "validation": Dataset.from_dict({"sentence": ["ok"], "label": [1]}),
            }
        )
        config = Config(
            model=ModelConfig(model_type="text_classification", model_name_or_path="roberta-base"),
            data=DataConfig(
                text_column="sentence",
                label_column="label",
                preprocessing_num_workers=1,
                max_seq_length=128,
            ),
            training=TrainingConfig(
                output_dir="./test-output",
                eval_strategy="steps",
                load_best_model_at_end=False,
                report_to="none",
            ),
        )

        class MockTokenizer:
            pad_token_id = 0

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
        load_calls = []
        preprocess_calls = []
        create_trainer_calls = []

        def fake_load_model_and_tokenizer(
            model_config,
            num_labels=None,
            *,
            max_seq_length=None,
            id2label=None,
            label2id=None,
        ):
            load_calls.append(
                {
                    "model_config": model_config,
                    "num_labels": num_labels,
                    "max_seq_length": max_seq_length,
                    "id2label": id2label,
                    "label2id": label2id,
                }
            )
            return model, MockTokenizer()

        def fake_preprocess_text_classification_dataset(
            raw_dataset, tokenizer, data_config, shuffle_seed=None
        ):
            preprocess_calls.append((raw_dataset, tokenizer, data_config, shuffle_seed))
            return raw_dataset

        def fake_create_trainer(**kwargs):
            create_trainer_calls.append(kwargs)
            return trainer

        monkeypatch.setattr(train_module, "load_text_dataset", lambda data_config: dataset)
        monkeypatch.setattr(
            train_module,
            "load_model_and_tokenizer",
            fake_load_model_and_tokenizer,
        )
        monkeypatch.setattr(
            train_module,
            "get_num_labels_from_dataset",
            lambda split, label_column="label": 2,
        )
        monkeypatch.setattr(
            train_module,
            "get_id2label",
            lambda split, label_column="label": {0: "negative", 1: "positive"},
        )
        monkeypatch.setattr(
            train_module,
            "get_label2id",
            lambda split, label_column="label": {"negative": 0, "positive": 1},
        )
        monkeypatch.setattr(
            train_module,
            "get_text_target_modules",
            lambda model_name: ["query", "value"],
        )
        monkeypatch.setattr(train_module, "get_peft_model_with_lora", lambda *args, **kwargs: model)
        monkeypatch.setattr(
            train_module,
            "prepare_model_for_training",
            lambda *args, **kwargs: model,
        )
        monkeypatch.setattr(train_module, "print_model_size", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            train_module,
            "preprocess_text_classification_dataset",
            fake_preprocess_text_classification_dataset,
        )
        monkeypatch.setattr(
            train_module,
            "get_text_classification_collator",
            lambda tokenizer: "classification-collator",
        )
        monkeypatch.setattr(train_module, "create_trainer", fake_create_trainer)

        train_module.train_text_classification(config)

        assert load_calls == [
            {
                "model_config": config.model,
                "num_labels": 2,
                "max_seq_length": 128,
                "id2label": {0: "negative", 1: "positive"},
                "label2id": {"negative": 0, "positive": 1},
            }
        ]
        assert len(preprocess_calls) == 1
        assert preprocess_calls[0][3] == config.training.data_seed
        assert create_trainer_calls[0]["data_collator"] == "classification-collator"
        assert (
            create_trainer_calls[0]["compute_metrics"]
            is train_module.compute_metrics_for_classification
        )
        assert trainer.train_calls == [None]


class TestMainBootstrap:
    def test_main_bootstraps_unsloth_before_warning_setup(self, monkeypatch):
        events = []
        config = Config(
            model=ModelConfig(model_type="causal_lm", use_unsloth=True),
            training=TrainingConfig(output_dir="./test-output", eval_strategy="no"),
        )

        monkeypatch.setattr(train_module, "parse_args", lambda: argparse.Namespace(verbose=False))
        monkeypatch.setattr(train_module, "build_config", lambda args: config)
        monkeypatch.setattr(
            "lora_finetune._optional_unsloth.ensure_unsloth_imported",
            lambda: events.append("bootstrap"),
        )
        monkeypatch.setattr(train_module, "suppress_warnings", lambda: events.append("suppress"))

        def fake_ensure_runtime_imports():
            events.append("runtime_imports")
            train_module.hf_set_seed = lambda seed: events.append(f"hf_set_seed:{seed}")

        monkeypatch.setattr(train_module, "_ensure_runtime_imports", fake_ensure_runtime_imports)
        monkeypatch.setattr(
            train_module, "setup_logging", lambda level: events.append(f"logging:{level}")
        )
        monkeypatch.setattr(train_module, "set_seed", lambda seed: events.append(f"seed:{seed}"))
        monkeypatch.setattr(train_module, "train_llm", lambda cfg: events.append("train_llm"))
        monkeypatch.setattr(
            train_module.console,
            "print",
            lambda *args, **kwargs: events.append("print_config"),
        )

        train_module.main()

        assert events[:5] == [
            "bootstrap",
            "suppress",
            "runtime_imports",
            "logging:WARNING",
            f"seed:{config.training.seed}",
        ]
        assert f"hf_set_seed:{config.training.seed}" in events
        assert "train_llm" in events

    def test_main_skips_unsloth_bootstrap_when_disabled(self, monkeypatch):
        events = []
        config = Config(
            model=ModelConfig(model_type="causal_lm", use_unsloth=False),
            training=TrainingConfig(output_dir="./test-output", eval_strategy="no"),
        )

        monkeypatch.setattr(train_module, "parse_args", lambda: argparse.Namespace(verbose=True))
        monkeypatch.setattr(train_module, "build_config", lambda args: config)
        monkeypatch.setattr(
            "lora_finetune._optional_unsloth.ensure_unsloth_imported",
            lambda: events.append("bootstrap"),
        )
        monkeypatch.setattr(train_module, "suppress_warnings", lambda: events.append("suppress"))

        def fake_ensure_runtime_imports():
            events.append("runtime_imports")
            train_module.hf_set_seed = lambda seed: events.append(f"hf_set_seed:{seed}")

        monkeypatch.setattr(train_module, "_ensure_runtime_imports", fake_ensure_runtime_imports)
        monkeypatch.setattr(
            train_module, "setup_logging", lambda level: events.append(f"logging:{level}")
        )
        monkeypatch.setattr(train_module, "set_seed", lambda seed: events.append(f"seed:{seed}"))
        monkeypatch.setattr(train_module, "train_llm", lambda cfg: events.append("train_llm"))
        monkeypatch.setattr(
            train_module.console,
            "print",
            lambda *args, **kwargs: events.append("print_config"),
        )

        train_module.main()

        assert "bootstrap" not in events
        assert events[:3] == ["suppress", "runtime_imports", "logging:INFO"]

    def test_main_dispatches_text_classification(self, monkeypatch):
        events = []
        config = Config(
            model=ModelConfig(model_type="text_classification", use_unsloth=False),
            training=TrainingConfig(output_dir="./test-output", eval_strategy="no"),
        )

        monkeypatch.setattr(train_module, "parse_args", lambda: argparse.Namespace(verbose=False))
        monkeypatch.setattr(train_module, "build_config", lambda args: config)
        monkeypatch.setattr(train_module, "suppress_warnings", lambda: events.append("suppress"))

        def fake_ensure_runtime_imports():
            events.append("runtime_imports")
            train_module.hf_set_seed = lambda seed: events.append(f"hf_set_seed:{seed}")

        monkeypatch.setattr(train_module, "_ensure_runtime_imports", fake_ensure_runtime_imports)
        monkeypatch.setattr(
            train_module, "setup_logging", lambda level: events.append(f"logging:{level}")
        )
        monkeypatch.setattr(train_module, "set_seed", lambda seed: events.append(f"seed:{seed}"))
        monkeypatch.setattr(
            train_module,
            "train_text_classification",
            lambda cfg: events.append("train_text_classification"),
        )
        monkeypatch.setattr(train_module.console, "print", lambda *args, **kwargs: None)

        train_module.main()

        assert "train_text_classification" in events
