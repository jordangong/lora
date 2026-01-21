"""Configuration dataclasses for LoRA finetuning."""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import yaml


@dataclass
class LoraConfig:
    """LoRA-specific configuration."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Optional[str] = None
    modules_to_save: Optional[List[str]] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B"
    model_type: Literal["causal_lm", "seq2seq", "vision"] = "causal_lm"
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    trust_remote_code: bool = False
    use_flash_attention_2: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    attn_implementation: Optional[str] = None


@dataclass
class AugmentationConfig:
    """Data augmentation configuration for vision models."""

    # RandomResizedCrop
    random_resized_crop: bool = True
    random_resized_crop_scale: tuple = (0.08, 1.0)
    random_resized_crop_ratio: tuple = (0.75, 1.333)

    # RandomHorizontalFlip
    random_horizontal_flip: bool = True
    random_horizontal_flip_p: float = 0.5

    # RandomVerticalFlip
    random_vertical_flip: bool = False
    random_vertical_flip_p: float = 0.5

    # ColorJitter
    color_jitter: bool = False
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1

    # RandomRotation
    random_rotation: bool = False
    random_rotation_degrees: float = 15.0

    # RandomAffine
    random_affine: bool = False
    random_affine_degrees: float = 0.0
    random_affine_translate: Optional[tuple] = None
    random_affine_scale: Optional[tuple] = None
    random_affine_shear: Optional[float] = None

    # RandomGrayscale
    random_grayscale: bool = False
    random_grayscale_p: float = 0.1

    # GaussianBlur
    gaussian_blur: bool = False
    gaussian_blur_kernel_size: int = 23
    gaussian_blur_sigma: tuple = (0.1, 2.0)

    # RandomErasing
    random_erasing: bool = False
    random_erasing_p: float = 0.5
    random_erasing_scale: tuple = (0.02, 0.33)
    random_erasing_ratio: tuple = (0.3, 3.3)

    # AutoAugment / RandAugment / TrivialAugment
    auto_augment: Optional[str] = None  # "imagenet", "cifar10", "svhn"
    rand_augment: bool = False
    rand_augment_num_ops: int = 2
    rand_augment_magnitude: int = 9
    trivial_augment: bool = False

    # Normalization (extracted from image_processor if None)
    normalize_mean: Optional[List[float]] = None
    normalize_std: Optional[List[float]] = None

    # Eval transforms
    eval_resize_factor: float = 1.14  # Resize to image_size * factor before center crop


@dataclass
class DataConfig:
    """Data configuration."""

    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    image_column: str = "image"
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4
    streaming: bool = False
    train_split: str = "train"
    validation_split: str = "validation"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    image_size: int = 224
    prompt_template: Optional[str] = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[dict] = None
    optim: str = "adamw_torch"
    seed: int = 42
    data_seed: int = 42
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = True
    report_to: str = "wandb"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_watch: str = "gradients"
    wandb_log_model: bool = False
    run_name: Optional[str] = None
    fsdp: str = ""
    fsdp_config: Optional[dict] = None
    deepspeed: Optional[str] = None
    local_rank: int = -1
    ddp_find_unused_parameters: bool = False
    resume_from_checkpoint: Optional[str] = None
    hub_model_id: Optional[str] = None
    push_to_hub: bool = False
    hub_token: Optional[str] = None


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        model_config = ModelConfig(**config_dict.get("model", {}))
        lora_config = LoraConfig(**config_dict.get("lora", {}))

        # Handle nested augmentation config
        data_dict = config_dict.get("data", {})
        aug_dict = data_dict.pop("augmentation", {})
        aug_config = AugmentationConfig(**aug_dict) if aug_dict else AugmentationConfig()
        data_config = DataConfig(**data_dict, augmentation=aug_config)

        training_config = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            model=model_config,
            lora=lora_config,
            data=data_config,
            training=training_config,
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
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def update_from_args(self, args: dict) -> None:
        """Update configuration from command line arguments."""
        for key, value in args.items():
            if value is None:
                continue
            for config_name in ["model", "lora", "data", "training"]:
                config_obj = getattr(self, config_name)
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    break
