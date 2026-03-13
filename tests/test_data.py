"""Tests for data utilities."""

import pytest
import torch
from datasets import Dataset, DatasetDict
from PIL import Image
from torchvision import transforms

from lora_finetune.config import AugmentationConfig, DataConfig
from lora_finetune.data.text_data import (
    CHAT_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    format_instruction,
    get_text_collator,
    load_text_dataset,
    prepare_text_dataset_for_trl,
    preprocess_text_dataset,
    requires_trl_native_dataset,
    tokenize_function,
)
from lora_finetune.data.vision_data import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    build_eval_transforms,
    build_train_transforms,
    extract_normalization_from_processor,
    get_eval_transforms,
    get_image_size_from_processor,
    get_train_transforms,
    get_vision_collator,
    make_transform_fn,
    preprocess_vision_example,
)


class TestFormatInstruction:
    """Tests for format_instruction function."""

    def test_format_with_all_fields(self):
        """Test formatting with all fields present."""
        example = {
            "instruction": "Translate to French",
            "input": "Hello world",
            "output": "Bonjour le monde",
        }
        result = format_instruction(example)

        assert "text" in result
        assert "Translate to French" in result["text"]
        assert "Hello world" in result["text"]
        assert "Bonjour le monde" in result["text"]

    def test_format_with_response_field(self):
        """Test formatting with 'response' instead of 'output'."""
        example = {
            "instruction": "Say hello",
            "input": "",
            "response": "Hello!",
        }
        result = format_instruction(example)

        assert "Hello!" in result["text"]

    def test_format_with_custom_template(self):
        """Test formatting with custom template."""
        example = {
            "instruction": "Test instruction",
            "input": "Test input",
            "output": "Test output",
        }
        custom_template = "Q: {instruction}\nA: {output}"
        result = format_instruction(example, template=custom_template)

        assert "Q: Test instruction" in result["text"]
        assert "A: Test output" in result["text"]

    def test_format_with_missing_fields(self):
        """Test formatting with missing optional fields."""
        example = {
            "instruction": "Just an instruction",
        }
        result = format_instruction(example)

        assert "Just an instruction" in result["text"]


class TestTokenizeFunction:
    """Tests for tokenize_function."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""

        class MockTokenizer:
            def __call__(self, texts, truncation, max_length, padding, return_tensors):
                return {
                    "input_ids": [[1, 2, 3] for _ in texts],
                    "attention_mask": [[1, 1, 1] for _ in texts],
                }

        tokenizer = MockTokenizer()
        examples = {"text": ["Hello world", "Test text"]}

        result = tokenize_function(examples, tokenizer, max_length=512, text_column="text")

        assert "input_ids" in result
        assert len(result["input_ids"]) == 2

    def test_tokenize_response_only_loss_masks_prompt_tokens(self):
        class MockTokenizer:
            eos_token = None

            def __call__(self, texts, truncation, max_length, padding, return_tensors):
                return {
                    "input_ids": [[idx + 1 for idx, _ in enumerate(text)] for text in texts],
                    "attention_mask": [[1 for _ in text] for text in texts],
                }

        tokenizer = MockTokenizer()
        examples = {"text": ["promptanswer"], "_source_text": ["prompt"]}

        result = tokenize_function(
            examples,
            tokenizer,
            max_length=512,
            text_column="text",
            source_text_column="_source_text",
            response_only_loss=True,
            append_eos_token=False,
        )

        assert result["labels"][0][:6] == [-100] * 6
        assert result["labels"][0][6:] == result["input_ids"][0][6:]

    def test_tokenize_appends_eos_when_enabled(self):
        class MockTokenizer:
            eos_token = "<eos>"

            def __call__(self, texts, truncation, max_length, padding, return_tensors):
                return {
                    "input_ids": [[idx + 1 for idx, _ in enumerate(text)] for text in texts],
                    "attention_mask": [[1 for _ in text] for text in texts],
                }

        tokenizer = MockTokenizer()

        with_eos = tokenize_function(
            {"text": ["answer"]},
            tokenizer,
            max_length=512,
            append_eos_token=True,
        )
        without_eos = tokenize_function(
            {"text": ["answer"]},
            tokenizer,
            max_length=512,
            append_eos_token=False,
        )

        assert len(with_eos["input_ids"][0]) == len(without_eos["input_ids"][0]) + len(
            tokenizer.eos_token
        )


class TestGetTextCollator:
    """Tests for get_text_collator function."""

    def test_get_collator(self):
        """Test getting text collator."""

        class MockTokenizer:
            pad_token_id = 0
            padding_side = "right"

            def pad(
                self,
                features,
                padding=True,
                max_length=None,
                pad_to_multiple_of=None,
                return_tensors=None,
            ):
                max_len = max(len(feature["input_ids"]) for feature in features)
                if pad_to_multiple_of is not None and max_len % pad_to_multiple_of:
                    max_len = (
                        (max_len + pad_to_multiple_of - 1) // pad_to_multiple_of
                    ) * pad_to_multiple_of
                batch = {
                    "input_ids": [],
                    "attention_mask": [],
                }
                for feature in features:
                    pad_len = max_len - len(feature["input_ids"])
                    batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad_len)
                    batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
                return batch

        tokenizer = MockTokenizer()
        collator = get_text_collator(tokenizer, mlm=False)

        assert collator is not None
        batch = collator(
            [
                {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                {"input_ids": [3], "attention_mask": [1], "labels": [3]},
            ]
        )

        assert batch["labels"].tolist()[0][:2] == [-100, 2]
        assert batch["labels"].tolist()[1][0] == 3

    def test_get_collator_mlm(self):
        """Test getting MLM collator."""

        class MockTokenizer:
            pad_token_id = 0
            mask_token = "<mask>"
            mask_token_id = 4

        tokenizer = MockTokenizer()
        collator = get_text_collator(tokenizer, mlm=True)

        assert collator.mlm is True


class TestExtractNormalizationFromProcessor:
    """Tests for extract_normalization_from_processor function."""

    def test_extract_from_processor(self):
        """Test extracting normalization from image processor."""

        class MockProcessor:
            image_mean = [0.5, 0.5, 0.5]
            image_std = [0.25, 0.25, 0.25]

        processor = MockProcessor()
        mean, std = extract_normalization_from_processor(processor)

        assert mean == [0.5, 0.5, 0.5]
        assert std == [0.25, 0.25, 0.25]

    def test_extract_defaults_when_none(self):
        """Test default values when processor is None."""
        mean, std = extract_normalization_from_processor(None)

        assert mean == DEFAULT_MEAN
        assert std == DEFAULT_STD

    def test_extract_defaults_when_missing_attrs(self):
        """Test default values when processor lacks attributes."""

        class MockProcessor:
            pass

        processor = MockProcessor()
        mean, std = extract_normalization_from_processor(processor)

        assert mean == DEFAULT_MEAN
        assert std == DEFAULT_STD


class TestGetImageSizeFromProcessor:
    """Tests for get_image_size_from_processor function."""

    def test_size_from_dict_height(self):
        """Test extracting size from dict with height key."""

        class MockProcessor:
            size = {"height": 384, "width": 384}

        processor = MockProcessor()
        size = get_image_size_from_processor(processor)
        assert size == 384

    def test_size_from_dict_shortest_edge(self):
        """Test extracting size from dict with shortest_edge key."""

        class MockProcessor:
            size = {"shortest_edge": 256}

        processor = MockProcessor()
        size = get_image_size_from_processor(processor)
        assert size == 256

    def test_size_from_int(self):
        """Test extracting size when it's an int."""

        class MockProcessor:
            size = 224

        processor = MockProcessor()
        size = get_image_size_from_processor(processor)
        assert size == 224

    def test_size_from_crop_size(self):
        """Test extracting size from crop_size."""

        class MockProcessor:
            crop_size = {"height": 448}

        processor = MockProcessor()
        size = get_image_size_from_processor(processor)
        assert size == 448

    def test_default_when_none(self):
        """Test default size when processor is None."""
        size = get_image_size_from_processor(None, default=224)
        assert size == 224


class TestBuildTrainTransforms:
    """Tests for build_train_transforms function."""

    def test_default_transforms(self):
        """Test building default training transforms."""
        aug_config = AugmentationConfig()
        transform = build_train_transforms(224, aug_config, DEFAULT_MEAN, DEFAULT_STD)

        assert isinstance(transform, transforms.Compose)
        assert len(transform.transforms) > 0

    def test_transforms_with_augmentation(self):
        """Test building transforms with various augmentations."""
        aug_config = AugmentationConfig(
            random_resized_crop=True,
            random_horizontal_flip=True,
            color_jitter=True,
            random_rotation=True,
            random_rotation_degrees=15.0,
        )
        transform = build_train_transforms(224, aug_config, DEFAULT_MEAN, DEFAULT_STD)

        assert isinstance(transform, transforms.Compose)

    def test_transforms_without_random_crop(self):
        """Test transforms without random resized crop."""
        aug_config = AugmentationConfig(random_resized_crop=False)
        transform = build_train_transforms(224, aug_config, DEFAULT_MEAN, DEFAULT_STD)

        # Should have Resize and CenterCrop instead
        transform_types = [type(t).__name__ for t in transform.transforms]
        assert "Resize" in transform_types
        assert "CenterCrop" in transform_types

    def test_transforms_with_rand_augment(self):
        """Test transforms with RandAugment."""
        aug_config = AugmentationConfig(rand_augment=True, rand_augment_num_ops=3)
        transform = build_train_transforms(224, aug_config, DEFAULT_MEAN, DEFAULT_STD)

        transform_types = [type(t).__name__ for t in transform.transforms]
        assert "RandAugment" in transform_types

    def test_transforms_with_auto_augment(self):
        """Test transforms with AutoAugment."""
        aug_config = AugmentationConfig(auto_augment="imagenet")
        transform = build_train_transforms(224, aug_config, DEFAULT_MEAN, DEFAULT_STD)

        transform_types = [type(t).__name__ for t in transform.transforms]
        assert "AutoAugment" in transform_types

    def test_transforms_with_random_erasing(self):
        """Test transforms with RandomErasing."""
        aug_config = AugmentationConfig(random_erasing=True)
        transform = build_train_transforms(224, aug_config, DEFAULT_MEAN, DEFAULT_STD)

        transform_types = [type(t).__name__ for t in transform.transforms]
        assert "RandomErasing" in transform_types


class TestBuildEvalTransforms:
    """Tests for build_eval_transforms function."""

    def test_eval_transforms(self):
        """Test building evaluation transforms."""
        aug_config = AugmentationConfig()
        transform = build_eval_transforms(224, aug_config, DEFAULT_MEAN, DEFAULT_STD)

        assert isinstance(transform, transforms.Compose)

        transform_types = [type(t).__name__ for t in transform.transforms]
        assert "Resize" in transform_types
        assert "CenterCrop" in transform_types
        assert "ToTensor" in transform_types
        assert "Normalize" in transform_types

    def test_eval_transforms_resize_factor(self):
        """Test eval transforms with custom resize factor."""
        aug_config = AugmentationConfig(eval_resize_factor=1.2)
        transform = build_eval_transforms(224, aug_config, DEFAULT_MEAN, DEFAULT_STD)

        # First transform should be Resize with size = 224 * 1.2 = 268
        resize_transform = transform.transforms[0]
        assert isinstance(resize_transform, transforms.Resize)


class TestGetTrainTransforms:
    """Tests for get_train_transforms function."""

    def test_with_processor_normalization(self):
        """Test using normalization from processor."""

        class MockProcessor:
            image_mean = [0.5, 0.5, 0.5]
            image_std = [0.25, 0.25, 0.25]

        transform = get_train_transforms(224, image_processor=MockProcessor())
        assert isinstance(transform, transforms.Compose)

    def test_with_config_normalization(self):
        """Test using normalization from config."""
        aug_config = AugmentationConfig(
            normalize_mean=[0.6, 0.6, 0.6],
            normalize_std=[0.3, 0.3, 0.3],
        )
        transform = get_train_transforms(224, aug_config=aug_config)
        assert isinstance(transform, transforms.Compose)


class TestGetEvalTransforms:
    """Tests for get_eval_transforms function."""

    def test_basic_eval_transforms(self):
        """Test basic evaluation transforms."""
        transform = get_eval_transforms(224)
        assert isinstance(transform, transforms.Compose)


class TestPreprocessVisionExample:
    """Tests for preprocess_vision_example function."""

    def test_with_transform(self):
        """Test preprocessing with transform."""
        image = Image.new("RGB", (100, 100), color="red")
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        example = {"image": image, "label": 0}
        result = preprocess_vision_example(
            example, transform=transform, image_column="image", label_column="label"
        )

        assert "pixel_values" in result
        assert "labels" in result
        assert result["labels"] == 0
        assert result["pixel_values"].shape == (3, 224, 224)

    def test_with_pil_image(self):
        """Test preprocessing with PIL image."""
        image = Image.new("RGB", (100, 100), color="blue")
        transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])

        example = {"image": image, "label": 1}
        result = preprocess_vision_example(example, transform=transform)

        assert "pixel_values" in result
        assert isinstance(result["pixel_values"], torch.Tensor)

    def test_raises_without_processor_or_transform(self):
        """Test that error is raised without processor or transform."""
        image = Image.new("RGB", (100, 100))
        example = {"image": image, "label": 0}

        with pytest.raises(ValueError, match="Either image_processor or transform"):
            preprocess_vision_example(example)


class TestMakeTransformFn:
    """Tests for make_transform_fn function."""

    def test_transform_fn_creation(self):
        """Test creating transform function."""
        transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
            ]
        )

        transform_fn = make_transform_fn(transform, "image", "label")
        assert callable(transform_fn)

    def test_transform_fn_application(self):
        """Test applying transform function."""
        transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
            ]
        )

        transform_fn = make_transform_fn(transform, "image", "label")

        images = [Image.new("RGB", (100, 100)) for _ in range(3)]
        examples = {"image": images, "label": [0, 1, 2]}

        result = transform_fn(examples)

        assert "pixel_values" in result
        assert "labels" in result
        assert len(result["pixel_values"]) == 3
        assert result["labels"] == [0, 1, 2]


class TestGetVisionCollator:
    """Tests for get_vision_collator function."""

    def test_collator_creation(self):
        """Test creating vision collator."""
        collator = get_vision_collator()
        assert callable(collator)

    def test_collator_batching(self):
        """Test collator batches correctly."""
        collator = get_vision_collator()

        examples = [
            {"pixel_values": torch.rand(3, 224, 224), "labels": 0},
            {"pixel_values": torch.rand(3, 224, 224), "labels": 1},
            {"pixel_values": torch.rand(3, 224, 224), "labels": 2},
        ]

        batch = collator(examples)

        assert "pixel_values" in batch
        assert "labels" in batch
        assert batch["pixel_values"].shape == (3, 3, 224, 224)
        assert batch["labels"].shape == (3,)
        assert batch["labels"].tolist() == [0, 1, 2]

    def test_collator_without_labels(self):
        """Test collator works without labels."""
        collator = get_vision_collator()

        examples = [
            {"pixel_values": torch.rand(3, 64, 64)},
            {"pixel_values": torch.rand(3, 64, 64)},
        ]

        batch = collator(examples)

        assert "pixel_values" in batch
        assert "labels" not in batch
        assert batch["pixel_values"].shape == (2, 3, 64, 64)


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_default_template_structure(self):
        """Test default prompt template has expected structure."""
        assert "### Instruction:" in DEFAULT_PROMPT_TEMPLATE
        assert "### Input:" in DEFAULT_PROMPT_TEMPLATE
        assert "### Response:" in DEFAULT_PROMPT_TEMPLATE
        assert "{instruction}" in DEFAULT_PROMPT_TEMPLATE
        assert "{input}" in DEFAULT_PROMPT_TEMPLATE
        assert "{output}" in DEFAULT_PROMPT_TEMPLATE

    def test_chat_template_structure(self):
        """Test chat template has expected structure."""
        assert "<|begin_of_text|>" in CHAT_TEMPLATE
        assert "<|start_header_id|>" in CHAT_TEMPLATE
        assert "<|eot_id|>" in CHAT_TEMPLATE
        assert "{instruction}" in CHAT_TEMPLATE
        assert "{output}" in CHAT_TEMPLATE


class TestLoadTextDataset:
    """Tests for load_text_dataset function."""

    def test_load_raises_without_dataset_or_file(self):
        """Test that error is raised without dataset_name or train_file."""
        from lora_finetune.config import DataConfig
        from lora_finetune.data.text_data import load_text_dataset

        config = DataConfig(dataset_name=None, train_file=None)

        with pytest.raises(ValueError, match="Either dataset_name or train_file"):
            load_text_dataset(config)

    def test_load_raises_for_streaming_eval_split_ratio(self):
        """Test eval_split_ratio fails fast for streaming-like datasets."""
        from unittest.mock import patch

        class FakeIterableSplit:
            pass

        fake_dataset = {"train": FakeIterableSplit()}
        config = DataConfig(dataset_name="dummy", streaming=True, eval_split_ratio=0.1)

        with patch("lora_finetune.data.text_data.load_dataset", return_value=fake_dataset):
            with pytest.raises(
                ValueError, match="eval_split_ratio requires a non-streaming dataset"
            ):
                load_text_dataset(config)


class TestPreprocessTextDataset:
    """Tests for preprocess_text_dataset function."""

    def test_instruction_formatting_respects_custom_text_column(self):
        """Test instruction formatting writes to configured text_column."""

        class MockTokenizer:
            def __call__(self, texts, truncation, max_length, padding, return_tensors):
                return {
                    "input_ids": [[1, 2, 3] for _ in texts],
                    "attention_mask": [[1, 1, 1] for _ in texts],
                }

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "instruction": ["Translate"],
                        "input": ["hello"],
                        "output": ["bonjour"],
                    }
                )
            }
        )
        config = DataConfig(text_column="prompt", preprocessing_num_workers=1)

        tokenized = preprocess_text_dataset(dataset, MockTokenizer(), config)

        assert "train" in tokenized
        assert len(tokenized["train"]) == 1
        assert "input_ids" in tokenized["train"].column_names
        assert "labels" in tokenized["train"].column_names

    def test_instruction_examples_mask_prompt_tokens_by_default(self):
        class MockTokenizer:
            eos_token = None

            def __call__(self, texts, truncation, max_length, padding, return_tensors):
                return {
                    "input_ids": [[idx + 1 for idx, _ in enumerate(text)] for text in texts],
                    "attention_mask": [[1 for _ in text] for text in texts],
                }

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "instruction": ["Translate"],
                        "input": ["hello"],
                        "output": ["bonjour"],
                    }
                )
            }
        )
        config = DataConfig(preprocessing_num_workers=1)

        tokenized = preprocess_text_dataset(dataset, MockTokenizer(), config)
        labels = tokenized["train"][0]["labels"]

        assert labels.count(-100) > 0
        assert any(label != -100 for label in labels)

    def test_text_column_examples_use_full_sequence_loss(self):
        class MockTokenizer:
            eos_token = None

            def __call__(self, texts, truncation, max_length, padding, return_tensors):
                return {
                    "input_ids": [[idx + 1 for idx, _ in enumerate(text)] for text in texts],
                    "attention_mask": [[1 for _ in text] for text in texts],
                }

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "text": ["sample"],
                    }
                )
            }
        )
        config = DataConfig(text_column="text", preprocessing_num_workers=1)

        tokenized = preprocess_text_dataset(dataset, MockTokenizer(), config)

        assert tokenized["train"][0]["labels"] == tokenized["train"][0]["input_ids"]

    def test_append_eos_token_can_be_disabled(self):
        class MockTokenizer:
            eos_token = "<eos>"

            def __call__(self, texts, truncation, max_length, padding, return_tensors):
                return {
                    "input_ids": [[idx + 1 for idx, _ in enumerate(text)] for text in texts],
                    "attention_mask": [[1 for _ in text] for text in texts],
                }

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "text": ["sample"],
                    }
                )
            }
        )
        with_eos = preprocess_text_dataset(
            dataset,
            MockTokenizer(),
            DataConfig(text_column="text", preprocessing_num_workers=1, append_eos_token=True),
        )
        without_eos = preprocess_text_dataset(
            dataset,
            MockTokenizer(),
            DataConfig(text_column="text", preprocessing_num_workers=1, append_eos_token=False),
        )

        assert (
            len(with_eos["train"][0]["input_ids"]) == len(without_eos["train"][0]["input_ids"]) + 5
        )

    def test_max_train_samples_uses_shuffled_prefix_when_seeded(self):
        """Test max_train_samples is applied after deterministic shuffling."""

        class MockTokenizer:
            def __call__(self, texts, truncation, max_length, padding, return_tensors):
                ids = [int(text.split("-")[-1]) for text in texts]
                return {
                    "input_ids": [[value] for value in ids],
                    "attention_mask": [[1] for _ in ids],
                }

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "text": [f"sample-{idx}" for idx in range(10)],
                    }
                )
            }
        )
        config = DataConfig(text_column="text", preprocessing_num_workers=1, max_train_samples=4)

        tokenized = preprocess_text_dataset(
            dataset,
            MockTokenizer(),
            config,
            shuffle_seed=123,
        )

        expected_order = dataset["train"].shuffle(seed=123).select(range(4))["text"]
        expected_ids = [int(text.split("-")[-1]) for text in expected_order]
        actual_ids = [row[0] for row in tokenized["train"]["input_ids"]]

        assert actual_ids == expected_ids
        assert actual_ids != [0, 1, 2, 3]

    def test_train_split_not_shuffled_without_max_train_samples(self, monkeypatch):
        """Test train split is left in original order when no truncation is requested."""

        import lora_finetune.data.text_data as text_data

        class MockTokenizer:
            def __call__(self, texts, truncation, max_length, padding, return_tensors):
                ids = [int(text.split("-")[-1]) for text in texts]
                return {
                    "input_ids": [[value] for value in ids],
                    "attention_mask": [[1] for _ in ids],
                }

        def fail_if_called(split_data, seed):
            raise AssertionError("shuffle_dataset_split should not be called")

        monkeypatch.setattr(text_data, "shuffle_dataset_split", fail_if_called)

        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "text": [f"sample-{idx}" for idx in range(5)],
                    }
                )
            }
        )
        config = DataConfig(text_column="text", preprocessing_num_workers=1)

        tokenized = preprocess_text_dataset(
            dataset,
            MockTokenizer(),
            config,
            shuffle_seed=123,
        )

        actual_ids = [row[0] for row in tokenized["train"]["input_ids"]]

        assert actual_ids == [0, 1, 2, 3, 4]


class TestPrepareTextDatasetForTrl:
    """Tests for TRL-native dataset preparation."""

    def test_instruction_examples_become_prompt_completion_pairs(self):
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "instruction": ["Translate"],
                        "input": ["hello"],
                        "output": ["bonjour"],
                    }
                )
            }
        )

        prepared = prepare_text_dataset_for_trl(
            dataset,
            DataConfig(preprocessing_num_workers=1),
        )

        example = prepared["train"][0]
        assert list(prepared["train"].column_names) == ["prompt", "completion"]
        assert "Translate" in example["prompt"]
        assert "hello" in example["prompt"]
        assert example["completion"] == "bonjour"

    def test_conversational_dataset_passes_through_for_trl(self):
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

        prepared = prepare_text_dataset_for_trl(
            dataset,
            DataConfig(preprocessing_num_workers=1),
        )

        assert prepared["train"].column_names == ["messages"]
        assert prepared["train"][0]["messages"][1]["content"] == "hello"

    def test_conversations_dataset_is_normalized_to_messages_for_trl(self):
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "conversations": [
                            [
                                {"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "hello"},
                            ]
                        ],
                        "source": ["chatml"],
                    }
                )
            }
        )

        prepared = prepare_text_dataset_for_trl(
            dataset,
            DataConfig(preprocessing_num_workers=1),
        )

        assert "conversations" not in prepared["train"].column_names
        assert prepared["train"].column_names == ["source", "messages"]
        assert prepared["train"][0]["messages"][0]["role"] == "user"
        assert prepared["train"][0]["messages"][1]["content"] == "hello"

    def test_conversations_from_value_dataset_is_normalized_to_messages_for_trl(self):
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "conversations": [
                            [
                                {"from": "user", "value": "hi"},
                                {"from": "assistant", "value": "hello"},
                            ]
                        ]
                    }
                )
            }
        )

        prepared = prepare_text_dataset_for_trl(
            dataset,
            DataConfig(preprocessing_num_workers=1),
        )

        assert prepared["train"].column_names == ["messages"]
        assert prepared["train"][0]["messages"][0] == {"role": "user", "content": "hi"}
        assert prepared["train"][0]["messages"][1] == {"role": "assistant", "content": "hello"}


class TestRequiresTrlNativeDataset:
    """Tests for TRL-native dataset detection."""

    def test_detects_conversational_dataset(self):
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

        assert requires_trl_native_dataset(dataset) is True

    def test_detects_prompt_completion_dataset(self):
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "prompt": ["Question: hi\n\nAnswer: "],
                        "completion": ["hello"],
                    }
                )
            }
        )

        assert requires_trl_native_dataset(dataset) is True

    def test_ignores_plain_text_dataset(self):
        dataset = DatasetDict({"train": Dataset.from_dict({"text": ["hello"]})})

        assert requires_trl_native_dataset(dataset) is False


class TestPrepareDatasetForCausalLM:
    """Tests for prepare_dataset_for_causal_lm function."""

    def test_prepare_adds_labels(self):
        """Test that labels are added to the result."""
        from lora_finetune.data.text_data import prepare_dataset_for_causal_lm

        class MockTokenizer:
            def __call__(self, texts, truncation, max_length, padding):
                return {
                    "input_ids": [[1, 2, 3, 4] for _ in texts],
                    "attention_mask": [[1, 1, 1, 1] for _ in texts],
                }

        tokenizer = MockTokenizer()
        examples = {"text": ["Hello world", "Test text"]}

        result = prepare_dataset_for_causal_lm(examples, tokenizer, max_length=512)

        assert "input_ids" in result
        assert "labels" in result
        assert result["labels"] == result["input_ids"]
