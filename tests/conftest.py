"""Pytest configuration and fixtures for lora_finetune tests."""

import os
import sys
import tempfile

import pytest
import torch
from PIL import Image

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing."""
    return Image.new("RGB", (224, 224), color="red")


@pytest.fixture
def sample_images():
    """Create multiple sample PIL images for testing."""
    colors = ["red", "green", "blue", "yellow"]
    return [Image.new("RGB", (100, 100), color=c) for c in colors]


@pytest.fixture
def sample_tensor_batch():
    """Create a sample batch of image tensors."""
    return torch.rand(4, 3, 224, 224)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""

    class MockTokenizer:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token = "</s>"
        eos_token_id = 1
        bos_token = "<s>"
        bos_token_id = 2
        unk_token = "<unk>"
        unk_token_id = 3

        def __call__(
            self, texts, truncation=True, max_length=512, padding=False, return_tensors=None
        ):
            if isinstance(texts, str):
                texts = [texts]
            return {
                "input_ids": [[1, 2, 3, 4, 5] for _ in texts],
                "attention_mask": [[1, 1, 1, 1, 1] for _ in texts],
            }

        def __len__(self):
            return 32000

    return MockTokenizer()


@pytest.fixture
def mock_image_processor():
    """Create a mock image processor for testing."""

    class MockImageProcessor:
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        size = {"height": 224, "width": 224}

        def __call__(self, image, return_tensors="pt"):
            return {"pixel_values": torch.rand(1, 3, 224, 224)}

    return MockImageProcessor()


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )


@pytest.fixture
def sample_text_examples():
    """Create sample text examples for testing."""
    return [
        {
            "instruction": "Translate to French",
            "input": "Hello world",
            "output": "Bonjour le monde",
        },
        {
            "instruction": "Summarize",
            "input": "This is a long text that needs summarization.",
            "output": "Short summary.",
        },
    ]


@pytest.fixture
def sample_vision_examples(sample_images):
    """Create sample vision examples for testing."""
    return [{"image": img, "label": i} for i, img in enumerate(sample_images)]


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""

    class MockFeature:
        num_classes = 10
        names = [f"class_{i}" for i in range(10)]

    class MockDataset:
        features = {"label": MockFeature()}

        def __init__(self, data=None):
            self._data = data or []

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

        def select(self, indices):
            return MockDataset([self._data[i] for i in indices])

        def set_transform(self, fn):
            self._transform = fn

    return MockDataset


@pytest.fixture
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
