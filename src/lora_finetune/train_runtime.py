def ensure_runtime_imports(module) -> None:
    if module.hf_set_seed is None:
        from transformers import set_seed as imported_hf_set_seed

        module.hf_set_seed = imported_hf_set_seed

    if any(
        value is None
        for value in (
            module.get_text_collator,
            module.get_text_classification_collator,
            module.load_text_dataset,
            module.prepare_grpo_dataset_for_trl,
            module.prepare_preference_dataset_for_trl,
            module.prepare_text_dataset_for_trl,
            module.preprocess_text_dataset,
            module.preprocess_text_classification_dataset,
            module.requires_trl_native_dataset,
        )
    ):
        from .data.text_data import (
            get_text_classification_collator as imported_get_text_classification_collator,
        )
        from .data.text_data import (
            get_text_collator as imported_get_text_collator,
        )
        from .data.text_data import (
            load_text_dataset as imported_load_text_dataset,
        )
        from .data.text_data import (
            prepare_grpo_dataset_for_trl as imported_prepare_grpo_dataset_for_trl,
        )
        from .data.text_data import (
            prepare_preference_dataset_for_trl as imported_prepare_preference_dataset_for_trl,
        )
        from .data.text_data import (
            prepare_text_dataset_for_trl as imported_prepare_text_dataset_for_trl,
        )
        from .data.text_data import (
            preprocess_text_classification_dataset as imported_preprocess_text_classification_dataset,
        )
        from .data.text_data import (
            preprocess_text_dataset as imported_preprocess_text_dataset,
        )
        from .data.text_data import (
            requires_trl_native_dataset as imported_requires_trl_native_dataset,
        )

        if module.get_text_collator is None:
            module.get_text_collator = imported_get_text_collator
        if module.get_text_classification_collator is None:
            module.get_text_classification_collator = imported_get_text_classification_collator
        if module.load_text_dataset is None:
            module.load_text_dataset = imported_load_text_dataset
        if module.prepare_grpo_dataset_for_trl is None:
            module.prepare_grpo_dataset_for_trl = imported_prepare_grpo_dataset_for_trl
        if module.prepare_preference_dataset_for_trl is None:
            module.prepare_preference_dataset_for_trl = imported_prepare_preference_dataset_for_trl
        if module.prepare_text_dataset_for_trl is None:
            module.prepare_text_dataset_for_trl = imported_prepare_text_dataset_for_trl
        if module.preprocess_text_dataset is None:
            module.preprocess_text_dataset = imported_preprocess_text_dataset
        if module.preprocess_text_classification_dataset is None:
            module.preprocess_text_classification_dataset = imported_preprocess_text_classification_dataset
        if module.requires_trl_native_dataset is None:
            module.requires_trl_native_dataset = imported_requires_trl_native_dataset

    if any(
        value is None
        for value in (
            module.get_vision_collator,
            module.load_vision_dataset,
            module.preprocess_vision_dataset,
        )
    ):
        from .data.vision_data import (
            get_vision_collator as imported_get_vision_collator,
        )
        from .data.vision_data import (
            load_vision_dataset as imported_load_vision_dataset,
        )
        from .data.vision_data import (
            preprocess_vision_dataset as imported_preprocess_vision_dataset,
        )

        if module.get_vision_collator is None:
            module.get_vision_collator = imported_get_vision_collator
        if module.load_vision_dataset is None:
            module.load_vision_dataset = imported_load_vision_dataset
        if module.preprocess_vision_dataset is None:
            module.preprocess_vision_dataset = imported_preprocess_vision_dataset

    if module.LightEvalCallback is None or module.run_lighteval is None:
        from .evaluators import (
            LightEvalCallback as imported_LightEvalCallback,
        )
        from .evaluators import (
            run_lighteval as imported_run_lighteval,
        )

        if module.LightEvalCallback is None:
            module.LightEvalCallback = imported_LightEvalCallback
        if module.run_lighteval is None:
            module.run_lighteval = imported_run_lighteval

    if module.get_peft_model_with_lora is None or module.load_model_and_tokenizer is None:
        from .models.base import (
            get_peft_model_with_lora as imported_get_peft_model_with_lora,
        )
        from .models.base import (
            load_model_and_tokenizer as imported_load_model_and_tokenizer,
        )

        if module.get_peft_model_with_lora is None:
            module.get_peft_model_with_lora = imported_get_peft_model_with_lora
        if module.load_model_and_tokenizer is None:
            module.load_model_and_tokenizer = imported_load_model_and_tokenizer

    if module.get_llm_target_modules is None:
        from .models.llm import get_llm_target_modules as imported_get_llm_target_modules

        module.get_llm_target_modules = imported_get_llm_target_modules

    if module.get_text_target_modules is None:
        from .models.text import get_text_target_modules as imported_get_text_target_modules

        module.get_text_target_modules = imported_get_text_target_modules

    if any(
        value is None
        for value in (
            module.get_num_labels_from_dataset,
            module.get_id2label,
            module.get_label2id,
            module.get_vision_target_modules,
        )
    ):
        from .models.vision import (
            get_id2label as imported_get_id2label,
        )
        from .models.vision import (
            get_label2id as imported_get_label2id,
        )
        from .models.vision import (
            get_num_labels_from_dataset as imported_get_num_labels_from_dataset,
        )
        from .models.vision import (
            get_vision_target_modules as imported_get_vision_target_modules,
        )

        if module.get_num_labels_from_dataset is None:
            module.get_num_labels_from_dataset = imported_get_num_labels_from_dataset
        if module.get_id2label is None:
            module.get_id2label = imported_get_id2label
        if module.get_label2id is None:
            module.get_label2id = imported_get_label2id
        if module.get_vision_target_modules is None:
            module.get_vision_target_modules = imported_get_vision_target_modules

    if any(
        value is None
        for value in (
            module.compute_metrics_for_classification,
            module.compute_metrics_for_lm,
            module.create_trainer,
            module.prepare_model_for_training,
            module.run_hyperparameter_search,
            module.apply_hpo_parameters_to_config_sections,
        )
    ):
        from .trainer import (
            apply_hpo_parameters_to_config_sections as imported_apply_hpo_parameters_to_config_sections,
        )
        from .trainer import (
            compute_metrics_for_classification as imported_compute_metrics_for_classification,
        )
        from .trainer import (
            compute_metrics_for_lm as imported_compute_metrics_for_lm,
        )
        from .trainer import (
            create_trainer as imported_create_trainer,
        )
        from .trainer import (
            prepare_model_for_training as imported_prepare_model_for_training,
        )
        from .trainer import (
            run_hyperparameter_search as imported_run_hyperparameter_search,
        )

        if module.compute_metrics_for_classification is None:
            module.compute_metrics_for_classification = imported_compute_metrics_for_classification
        if module.compute_metrics_for_lm is None:
            module.compute_metrics_for_lm = imported_compute_metrics_for_lm
        if module.create_trainer is None:
            module.create_trainer = imported_create_trainer
        if module.prepare_model_for_training is None:
            module.prepare_model_for_training = imported_prepare_model_for_training
        if module.run_hyperparameter_search is None:
            module.run_hyperparameter_search = imported_run_hyperparameter_search
        if module.apply_hpo_parameters_to_config_sections is None:
            module.apply_hpo_parameters_to_config_sections = (
                imported_apply_hpo_parameters_to_config_sections
            )
