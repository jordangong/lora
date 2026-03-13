"""Configuration dataclasses for LoRA finetuning."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import yaml

# Supported finetuning methods
FINETUNE_METHODS = Literal[
    "lora", "full", "dora", "adalora", "loraplus", "ia3", "prefix_tuning"
]


@dataclass
class LoraConfig:
    """LoRA and adapter-specific configuration."""

    method: FINETUNE_METHODS = field(
        default="lora",
        metadata={
            "help": "Finetuning method: lora, full, dora, adalora, loraplus, ia3, prefix_tuning"
        },
    )
    r: int = field(
        default=16, metadata={"help": "LoRA rank (dimension of low-rank matrices)"}
    )
    alpha: int = field(default=32, metadata={"help": "LoRA alpha (scaling factor)"})
    dropout: float = field(
        default=0.05, metadata={"help": "Dropout probability for LoRA layers"}
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of module names to apply LoRA to"
        },
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type: 'none', 'all', or 'lora_only'"}
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Task type for PEFT (e.g., CAUSAL_LM, SEQ_2_SEQ_LM)"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules to save (not apply LoRA, but train fully)"},
    )
    # DoRA-specific
    use_dora: bool = field(
        default=False, metadata={"help": "Use weight-decomposed LoRA (DoRA)"}
    )
    # AdaLoRA-specific
    init_r: int = field(default=12, metadata={"help": "Initial rank for AdaLoRA"})
    target_r: int = field(
        default=8, metadata={"help": "Target average rank for AdaLoRA"}
    )
    tinit: int = field(default=0, metadata={"help": "Initial warmup steps for AdaLoRA"})
    tfinal: int = field(
        default=0, metadata={"help": "Final steps for AdaLoRA rank allocation"}
    )
    deltaT: int = field(
        default=1, metadata={"help": "Step interval for AdaLoRA rank update"}
    )
    beta1: float = field(
        default=0.85, metadata={"help": "Beta1 for AdaLoRA importance score EMA"}
    )
    beta2: float = field(
        default=0.85, metadata={"help": "Beta2 for AdaLoRA importance score EMA"}
    )
    orth_reg_weight: float = field(
        default=0.5, metadata={"help": "Orthogonal regularization weight for AdaLoRA"}
    )
    # LoRA+ specific
    loraplus_lr_ratio: float = field(
        default=16.0,
        metadata={
            "help": "Learning rate ratio for B matrix in LoRA+ (lr_B = lr_A * ratio)"
        },
    )
    # IA3-specific
    feedforward_modules: Optional[List[str]] = field(
        default=None, metadata={"help": "List of feedforward module names for IA3"}
    )
    # Prefix tuning specific
    num_virtual_tokens: int = field(
        default=20, metadata={"help": "Number of virtual tokens for prefix tuning"}
    )
    prefix_projection: bool = field(
        default=False, metadata={"help": "Use projection layer for prefix tuning"}
    )
    # RSLoRA
    use_rslora: bool = field(
        default=False, metadata={"help": "Use Rank-Stabilized LoRA scaling"}
    )

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        # Auto-enable DoRA flag when method is dora
        if self.method == "dora":
            self.use_dora = True


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name_or_path: str = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={"help": "Model name or path from HuggingFace Hub or local path"},
    )
    model_type: Literal["causal_lm", "seq2seq", "vision"] = field(
        default="causal_lm",
        metadata={"help": "Model type: causal_lm, seq2seq, or vision"},
    )
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Torch dtype: auto, float16, bfloat16, or float32"},
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code from HuggingFace Hub"}
    )
    use_flash_attention_2: bool = field(
        default=True, metadata={"help": "Use Flash Attention 2 for faster training"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Load model in 4-bit quantization (QLoRA)"}
    )
    load_in_8bit: bool = field(
        default=False, metadata={"help": "Load model in 8-bit quantization"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16", metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4", metadata={"help": "Quantization type: nf4 or fp4"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True, metadata={"help": "Use double quantization for 4-bit"}
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Attention implementation: eager, sdpa, or flash_attention_2"
        },
    )


@dataclass
class AugmentationConfig:
    """Data augmentation configuration for vision models."""

    random_resized_crop: bool = field(
        default=True, metadata={"help": "Enable random resized crop augmentation"}
    )
    random_resized_crop_scale: tuple = field(
        default=(0.08, 1.0),
        metadata={"help": "Scale range for random resized crop (min, max)"},
    )
    random_resized_crop_ratio: tuple = field(
        default=(0.75, 1.333),
        metadata={"help": "Aspect ratio range for random resized crop"},
    )
    random_horizontal_flip: bool = field(
        default=True, metadata={"help": "Enable random horizontal flip augmentation"}
    )
    random_horizontal_flip_p: float = field(
        default=0.5, metadata={"help": "Probability of random horizontal flip"}
    )
    random_vertical_flip: bool = field(
        default=False, metadata={"help": "Enable random vertical flip augmentation"}
    )
    random_vertical_flip_p: float = field(
        default=0.5, metadata={"help": "Probability of random vertical flip"}
    )
    color_jitter: bool = field(
        default=False, metadata={"help": "Enable color jitter augmentation"}
    )
    color_jitter_brightness: float = field(
        default=0.2, metadata={"help": "Brightness factor for color jitter"}
    )
    color_jitter_contrast: float = field(
        default=0.2, metadata={"help": "Contrast factor for color jitter"}
    )
    color_jitter_saturation: float = field(
        default=0.2, metadata={"help": "Saturation factor for color jitter"}
    )
    color_jitter_hue: float = field(
        default=0.1, metadata={"help": "Hue factor for color jitter"}
    )
    random_rotation: bool = field(
        default=False, metadata={"help": "Enable random rotation augmentation"}
    )
    random_rotation_degrees: float = field(
        default=15.0, metadata={"help": "Maximum rotation degrees"}
    )
    random_affine: bool = field(
        default=False, metadata={"help": "Enable random affine transformation"}
    )
    random_affine_degrees: float = field(
        default=0.0, metadata={"help": "Rotation degrees for affine transform"}
    )
    random_affine_translate: Optional[tuple] = field(
        default=None, metadata={"help": "Translation range for affine transform (x, y)"}
    )
    random_affine_scale: Optional[tuple] = field(
        default=None, metadata={"help": "Scale range for affine transform (min, max)"}
    )
    random_affine_shear: Optional[float] = field(
        default=None, metadata={"help": "Shear degrees for affine transform"}
    )
    random_grayscale: bool = field(
        default=False, metadata={"help": "Enable random grayscale augmentation"}
    )
    random_grayscale_p: float = field(
        default=0.1, metadata={"help": "Probability of random grayscale"}
    )
    gaussian_blur: bool = field(
        default=False, metadata={"help": "Enable Gaussian blur augmentation"}
    )
    gaussian_blur_kernel_size: int = field(
        default=23, metadata={"help": "Kernel size for Gaussian blur"}
    )
    gaussian_blur_sigma: tuple = field(
        default=(0.1, 2.0),
        metadata={"help": "Sigma range for Gaussian blur (min, max)"},
    )
    random_erasing: bool = field(
        default=False, metadata={"help": "Enable random erasing augmentation"}
    )
    random_erasing_p: float = field(
        default=0.5, metadata={"help": "Probability of random erasing"}
    )
    random_erasing_scale: tuple = field(
        default=(0.02, 0.33), metadata={"help": "Scale range for random erasing"}
    )
    random_erasing_ratio: tuple = field(
        default=(0.3, 3.3), metadata={"help": "Aspect ratio range for random erasing"}
    )
    auto_augment: Optional[str] = field(
        default=None,
        metadata={"help": "AutoAugment policy: imagenet, cifar10, or svhn"},
    )
    rand_augment: bool = field(default=False, metadata={"help": "Enable RandAugment"})
    rand_augment_num_ops: int = field(
        default=2, metadata={"help": "Number of operations for RandAugment"}
    )
    rand_augment_magnitude: int = field(
        default=9, metadata={"help": "Magnitude for RandAugment (0-30)"}
    )
    trivial_augment: bool = field(
        default=False, metadata={"help": "Enable TrivialAugment"}
    )
    normalize_mean: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Normalization mean values (extracted from processor if None)"
        },
    )
    normalize_std: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Normalization std values (extracted from processor if None)"
        },
    )
    eval_resize_factor: float = field(
        default=1.14, metadata={"help": "Resize factor before center crop during eval"}
    )


@dataclass
class DataConfig:
    """Data configuration."""

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Dataset name from HuggingFace Hub"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "Dataset configuration name"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Path to training data file (JSON/CSV)"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Path to validation data file (JSON/CSV)"}
    )
    text_column: str = field(
        default="text", metadata={"help": "Column name for text data"}
    )
    label_column: str = field(
        default="label", metadata={"help": "Column name for labels"}
    )
    image_column: str = field(
        default="image", metadata={"help": "Column name for image data"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for tokenization"}
    )
    preprocessing_num_workers: int = field(
        default=4, metadata={"help": "Number of workers for data preprocessing"}
    )
    streaming: bool = field(
        default=False, metadata={"help": "Enable streaming mode for large datasets"}
    )
    train_split: str = field(
        default="train", metadata={"help": "Name of the training split"}
    )
    validation_split: str = field(
        default="validation", metadata={"help": "Name of the validation split"}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of training samples"}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of evaluation samples"}
    )
    eval_split_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "Ratio of training data to use for evaluation (e.g., 0.1 for 10%%)"
        },
    )
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Holdout dataset name for zero-shot evaluation (e.g., 'gsm8k', 'hellaswag')"
        },
    )
    eval_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Holdout dataset configuration name"},
    )
    eval_dataset_split: str = field(
        default="test",
        metadata={"help": "Split to use from the holdout eval dataset"},
    )
    image_size: int = field(
        default=224, metadata={"help": "Image size for vision models"}
    )
    prompt_template: Optional[str] = field(
        default=None, metadata={"help": "Prompt template for text formatting"}
    )
    response_only_loss: bool = field(
        default=True,
        metadata={
            "help": "Mask prompt tokens so loss is computed only on responses when supported"
        },
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={
            "help": "Use TRL assistant-only loss masking for conversational datasets when supported"
        },
    )
    append_eos_token: bool = field(
        default=True,
        metadata={
            "help": "Append EOS token to each text example when the tokenizer provides one"
        },
    )
    eos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override EOS token passed through to TRL SFTConfig when supported"
        },
    )
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class BenchmarkEvalConfig:
    """Configuration for benchmark evaluation using lighteval."""

    enabled: bool = field(
        default=False, metadata={"help": "Enable benchmark evaluation during training"}
    )
    tasks: str = field(
        default="gsm8k",
        metadata={"help": "Lighteval task string (e.g. 'gsm8k', 'gsm8k,mmlu')"},
    )
    eval_steps: int = field(
        default=500, metadata={"help": "Run benchmark evaluation every N steps"}
    )
    num_samples: Optional[int] = field(
        default=100, metadata={"help": "Number of samples to evaluate (None for all)"}
    )
    max_new_tokens: int = field(
        default=512, metadata={"help": "Maximum new tokens to generate for evaluation"}
    )
    batch_size: int = field(
        default=1, metadata={"help": "Batch size for benchmark evaluation"}
    )

    def __post_init__(self):
        if self.eval_steps <= 0:
            raise ValueError("benchmark_eval.eval_steps must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("benchmark_eval.batch_size must be greater than 0")
        if self.max_new_tokens <= 0:
            raise ValueError("benchmark_eval.max_new_tokens must be greater than 0")
        if self.num_samples is not None and self.num_samples <= 0:
            raise ValueError(
                "benchmark_eval.num_samples must be greater than 0 when provided"
            )


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str = field(
        default="./outputs", metadata={"help": "Output directory for model and logs"}
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Training batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=4, metadata={"help": "Evaluation batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=4, metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(default=2e-4, metadata={"help": "Learning rate"})
    weight_decay: float = field(
        default=0.01, metadata={"help": "Weight decay for AdamW optimizer"}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Warmup ratio of total training steps"}
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps (overrides warmup_ratio if > 0)"},
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Maximum gradient norm for clipping"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "LR scheduler type: linear, cosine, constant, etc."},
    )
    logging_steps: int = field(default=10, metadata={"help": "Log every N steps"})
    save_steps: int = field(
        default=500, metadata={"help": "Save checkpoint every N steps"}
    )
    save_total_limit: int = field(
        default=3, metadata={"help": "Maximum number of checkpoints to keep"}
    )
    eval_steps: int = field(default=500, metadata={"help": "Evaluate every N steps"})
    eval_strategy: str = field(
        default="steps", metadata={"help": "Evaluation strategy: no, steps, or epoch"}
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={
            "help": "Only compute loss during eval (don't store logits to save memory)"
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "Save strategy: no, steps, or epoch"}
    )
    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "Load best model at end of training"}
    )
    metric_for_best_model: str = field(
        default="eval_loss", metadata={"help": "Metric to use for best model selection"}
    )
    greater_is_better: bool = field(
        default=False, metadata={"help": "Whether higher metric values are better"}
    )
    bf16: bool = field(default=True, metadata={"help": "Use bfloat16 mixed precision"})
    fp16: bool = field(default=False, metadata={"help": "Use float16 mixed precision"})
    tf32: Optional[bool] = field(
        default=None, metadata={"help": "Enable TF32 on Ampere GPUs"}
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Enable gradient checkpointing to save memory"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None, metadata={"help": "Kwargs for gradient checkpointing"}
    )
    optim: str = field(
        default="adamw_torch_fused",
        metadata={"help": "Optimizer: adamw_torch_fused, adamw_torch, sgd, etc."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    data_seed: int = field(
        default=42, metadata={"help": "Random seed for data sampling"}
    )
    dataloader_num_workers: int = field(
        default=4, metadata={"help": "Number of dataloader workers"}
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Pin memory in dataloader"}
    )
    remove_unused_columns: bool = field(
        default=True, metadata={"help": "Remove unused columns from dataset"}
    )
    llm_trainer: Literal["trl", "transformers"] = field(
        default="trl",
        metadata={"help": "Trainer backend for LLM finetuning: trl or transformers"},
    )
    report_to: str = field(
        default="wandb", metadata={"help": "Report to: wandb, tensorboard, or none"}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases project name"}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases run name"}
    )
    wandb_watch: str = field(
        default="false", metadata={"help": "W&B watch mode: gradients, all, or false"}
    )
    wandb_log_model: bool = field(default=False, metadata={"help": "Log model to W&B"})
    run_name: Optional[str] = field(
        default=None, metadata={"help": "Run name for logging"}
    )
    fsdp: Optional[str] = field(
        default=None, metadata={"help": "FSDP configuration string"}
    )
    fsdp_config: Optional[dict] = field(
        default=None, metadata={"help": "FSDP configuration dict"}
    )
    deepspeed: Optional[str] = field(
        default=None, metadata={"help": "Path to DeepSpeed config file"}
    )
    local_rank: int = field(
        default=-1, metadata={"help": "Local rank for distributed training"}
    )
    ddp_find_unused_parameters: bool = field(
        default=False, metadata={"help": "Find unused parameters in DDP"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to checkpoint to resume from"}
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "HuggingFace Hub model ID for pushing"}
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Push model to HuggingFace Hub"}
    )
    hub_token: Optional[str] = field(
        default=None, metadata={"help": "HuggingFace Hub token"}
    )


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    benchmark_eval: BenchmarkEvalConfig = field(default_factory=BenchmarkEvalConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}
        elif not isinstance(config_dict, dict):
            raise ValueError("Config file must contain a YAML mapping at the top level")

        model_config = ModelConfig(**config_dict.get("model", {}))
        lora_config = LoraConfig(**config_dict.get("lora", {}))

        # Handle nested augmentation config
        data_dict = config_dict.get("data", {})
        aug_dict = data_dict.pop("augmentation", {})
        aug_config = (
            AugmentationConfig(**aug_dict) if aug_dict else AugmentationConfig()
        )
        data_config = DataConfig(**data_dict, augmentation=aug_config)

        training_config = TrainingConfig(**config_dict.get("training", {}))
        benchmark_eval_config = BenchmarkEvalConfig(
            **config_dict.get("benchmark_eval", {})
        )

        return cls(
            model=model_config,
            lora=lora_config,
            data=data_config,
            training=training_config,
            benchmark_eval=benchmark_eval_config,
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        data_dict = {k: v for k, v in self.data.__dict__.items() if k != "augmentation"}
        data_dict["augmentation"] = self.data.augmentation.__dict__
        config_dict = {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "data": data_dict,
            "training": self.training.__dict__,
            "benchmark_eval": self.benchmark_eval.__dict__,
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def update_from_args(self, args: dict) -> None:
        """Update configuration from command line arguments."""
        for key, value in args.items():
            if value is None:
                continue
            for config_name in ["model", "lora", "data", "training", "benchmark_eval"]:
                config_obj = getattr(self, config_name)
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    break
