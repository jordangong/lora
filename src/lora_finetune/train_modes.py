from .config import Config


def train_llm(config: Config, deps, logger) -> None:
    trainer_type = config.training.trainer_type
    if config.model.use_unsloth and trainer_type != "sft":
        raise ValueError("Unsloth is currently only supported with training.trainer_type='sft'")
    if config.benchmark_eval.enabled and trainer_type != "sft":
        raise ValueError(
            "benchmark_eval is currently only supported with training.trainer_type='sft'"
        )

    def _build_llm_model_and_tokenizer():
        model, tokenizer = deps.load_model_and_tokenizer(
            config.model,
            max_seq_length=config.data.max_seq_length,
        )

        deps._resolve_default_target_modules(config, deps.get_llm_target_modules)

        is_quantized = config.model.load_in_4bit or config.model.load_in_8bit
        model = deps.get_peft_model_with_lora(
            model,
            config.lora,
            model_type=config.model.model_type,
            is_quantized=is_quantized,
            use_unsloth=config.model.use_unsloth,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
            random_state=config.training.seed,
            max_seq_length=config.data.max_seq_length,
        )

        model = deps.prepare_model_for_training(model, config.training, tokenizer)
        return model, tokenizer

    model, tokenizer = deps._run_with_status(
        "[bold blue]Loading model...", _build_llm_model_and_tokenizer
    )

    deps.print_model_size(model)

    def _load_and_prepare_llm_dataset():
        dataset = deps.load_text_dataset(config.data)
        use_trl_native_dataset = (
            trainer_type == "sft"
            and config.training.llm_trainer == "trl"
            and (config.data.append_eos_token or deps.requires_trl_native_dataset(dataset))
        )
        if trainer_type == "dpo":
            prepared_dataset = deps.prepare_preference_dataset_for_trl(
                dataset,
                config.data,
                shuffle_seed=config.training.data_seed,
            )
        elif trainer_type == "grpo":
            prepared_dataset = deps.prepare_grpo_dataset_for_trl(
                dataset,
                config.data,
                shuffle_seed=config.training.data_seed,
            )
        elif use_trl_native_dataset:
            prepared_dataset = deps.prepare_text_dataset_for_trl(
                dataset,
                config.data,
                shuffle_seed=config.training.data_seed,
            )
        else:
            prepared_dataset = deps.preprocess_text_dataset(
                dataset,
                tokenizer,
                config.data,
                shuffle_seed=config.training.data_seed,
            )
        return prepared_dataset, use_trl_native_dataset

    prepared_dataset, use_trl_native_dataset = deps._run_with_status(
        "[bold blue]Loading dataset...", _load_and_prepare_llm_dataset
    )

    train_dataset, eval_dataset = deps._get_train_and_eval_datasets(prepared_dataset, config.data)

    if trainer_type == "sft":
        data_collator = None if use_trl_native_dataset else deps.get_text_collator(tokenizer)
    else:
        data_collator = None

    if trainer_type == "sft":
        compute_metrics = deps.compute_metrics_for_lm if eval_dataset is not None else None
    else:
        compute_metrics = None

    # Create benchmark evaluation callback if enabled
    callbacks = []
    if config.benchmark_eval.enabled and trainer_type == "sft":
        eval_callback = deps.LightEvalCallback(
            model_name=config.model.model_name_or_path,
            tasks=config.benchmark_eval.tasks,
            eval_steps=config.benchmark_eval.eval_steps,
            max_samples=config.benchmark_eval.num_samples,
            max_new_tokens=config.benchmark_eval.max_new_tokens,
            batch_size=config.benchmark_eval.batch_size,
        )
        callbacks.append(eval_callback)
        logger.info(
            f"Benchmark eval ({config.benchmark_eval.tasks}) enabled every {config.benchmark_eval.eval_steps} steps"
        )

    hpo_config_sections, model_init = deps._get_hpo_setup(config, _build_llm_model_and_tokenizer)

    trainer = deps.create_trainer(
        model=model,
        training_config=config.training,
        model_config=config.model,
        train_dataset=train_dataset,
        data_config=config.data,
        dpo_config=config.dpo,
        grpo_config=config.grpo,
        benchmark_eval_config=config.benchmark_eval,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        lora_config=config.lora,
        callbacks=callbacks if callbacks else None,
        hpo_config=config.hpo,
        model_init=model_init,
        hpo_config_sections=hpo_config_sections,
    )

    if config.hpo.enabled:
        if deps._run_hpo_if_enabled(trainer, config):
            return

    should_run_final_eval = str(config.training.eval_strategy).lower() != "no"
    should_run_final_benchmark_eval = config.benchmark_eval.enabled and trainer_type == "sft"

    def _run_final_training_actions(current_trainer):
        if should_run_final_eval:
            deps._run_final_trainer_evaluation(current_trainer)

        if should_run_final_benchmark_eval and deps._should_run_final_benchmark_eval(
            current_trainer, config.benchmark_eval
        ):
            deps.run_benchmark_eval(
                model,
                config.model.model_name_or_path,
                config.benchmark_eval,
                trainer=current_trainer,
            )

    deps._run_trainer_training(
        trainer,
        resume_from_checkpoint=config.training.resume_from_checkpoint,
        final_evaluation_enabled=should_run_final_eval or should_run_final_benchmark_eval,
        final_evaluation_fn=_run_final_training_actions,
    )

    deps._save_and_maybe_push_model(trainer, config)


def train_vision(config: Config, deps) -> None:
    # Vision training uses set_transform which needs the original image column
    if config.training.remove_unused_columns:
        config.training.remove_unused_columns = False

    num_labels = None

    def _build_vision_model():
        model, image_processor = deps.load_model_and_tokenizer(config.model, num_labels=num_labels)

        deps._resolve_default_target_modules(config, deps.get_vision_target_modules)

        is_quantized = config.model.load_in_4bit or config.model.load_in_8bit
        model = deps.get_peft_model_with_lora(
            model,
            config.lora,
            model_type="vision",
            is_quantized=is_quantized,
        )

        model = deps.prepare_model_for_training(model, config.training)
        return model, image_processor

    def _load_vision_training_dataset():
        dataset = deps.load_vision_dataset(config.data)
        num_labels = deps.get_num_labels_from_dataset(dataset[config.data.train_split])
        return dataset, num_labels

    dataset, num_labels = deps._run_with_status(
        "[bold blue]Loading dataset...", _load_vision_training_dataset
    )
    model, image_processor = deps._run_with_status(
        "[bold blue]Loading model...", _build_vision_model
    )

    deps.print_model_size(model)

    processed_dataset = deps._run_with_status(
        "[bold blue]Preprocessing dataset...",
        lambda: deps.preprocess_vision_dataset(dataset, image_processor, config.data),
    )

    train_dataset, eval_dataset = deps._get_train_and_eval_datasets(processed_dataset, config.data)

    data_collator = deps.get_vision_collator()

    # Use accuracy metric for vision classification
    compute_metrics = deps.compute_metrics_for_classification if eval_dataset is not None else None

    hpo_config_sections, model_init = deps._get_hpo_setup(config, _build_vision_model)

    trainer = deps.create_trainer(
        model=model,
        training_config=config.training,
        model_config=config.model,
        train_dataset=train_dataset,
        benchmark_eval_config=config.benchmark_eval,
        eval_dataset=eval_dataset,
        processing_class=None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        lora_config=config.lora,
        hpo_config=config.hpo,
        model_init=model_init,
        hpo_config_sections=hpo_config_sections,
    )

    if config.hpo.enabled:
        model = None
        if deps._run_hpo_if_enabled(trainer, config):
            return

    deps._run_trainer_training(
        trainer,
        resume_from_checkpoint=config.training.resume_from_checkpoint,
        final_evaluation_enabled=str(config.training.eval_strategy).lower() != "no",
    )

    deps._save_and_maybe_push_model(trainer, config)


def train_text_classification(config: Config, deps) -> None:
    num_labels = None
    id2label = None
    label2id = None

    def _build_text_classification_model():
        model, tokenizer = deps.load_model_and_tokenizer(
            config.model,
            num_labels=num_labels,
            max_seq_length=config.data.max_seq_length,
            id2label=id2label or None,
            label2id=label2id or None,
        )

        deps._resolve_default_target_modules(config, deps.get_text_target_modules)

        is_quantized = config.model.load_in_4bit or config.model.load_in_8bit
        model = deps.get_peft_model_with_lora(
            model,
            config.lora,
            model_type="text_classification",
            is_quantized=is_quantized,
        )

        model = deps.prepare_model_for_training(model, config.training, tokenizer)
        return model, tokenizer

    def _load_text_classification_dataset():
        dataset = deps.load_text_dataset(config.data)
        train_split = dataset[config.data.train_split]
        num_labels = deps.get_num_labels_from_dataset(
            train_split,
            label_column=config.data.label_column,
        )
        id2label = deps.get_id2label(train_split, label_column=config.data.label_column)
        label2id = deps.get_label2id(train_split, label_column=config.data.label_column)
        return dataset, num_labels, id2label, label2id

    dataset, num_labels, id2label, label2id = deps._run_with_status(
        "[bold blue]Loading dataset...", _load_text_classification_dataset
    )
    model, tokenizer = deps._run_with_status(
        "[bold blue]Loading model...", _build_text_classification_model
    )

    deps.print_model_size(model)

    processed_dataset = deps._run_with_status(
        "[bold blue]Preprocessing dataset...",
        lambda: deps.preprocess_text_classification_dataset(
            dataset,
            tokenizer,
            config.data,
            shuffle_seed=config.training.data_seed,
        ),
    )

    train_dataset, eval_dataset = deps._get_train_and_eval_datasets(processed_dataset, config.data)

    data_collator = deps.get_text_classification_collator(tokenizer)
    compute_metrics = deps.compute_metrics_for_classification if eval_dataset is not None else None

    hpo_config_sections, model_init = deps._get_hpo_setup(config, _build_text_classification_model)

    trainer = deps.create_trainer(
        model=model,
        training_config=config.training,
        model_config=config.model,
        train_dataset=train_dataset,
        data_config=config.data,
        benchmark_eval_config=config.benchmark_eval,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        lora_config=config.lora,
        hpo_config=config.hpo,
        model_init=model_init,
        hpo_config_sections=hpo_config_sections,
    )

    if config.hpo.enabled:
        if deps._run_hpo_if_enabled(trainer, config):
            return

    deps._run_trainer_training(
        trainer,
        resume_from_checkpoint=config.training.resume_from_checkpoint,
        final_evaluation_enabled=str(config.training.eval_strategy).lower() != "no",
    )

    deps._save_and_maybe_push_model(trainer, config)
