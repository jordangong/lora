"""Tests for utility functions."""

import io
import logging
import random
import sys
import warnings

import numpy as np
import pytest
import torch

import lora_finetune.utils as utils_module
from lora_finetune.utils import (
    RichWarningHandler,
    _normalize_warning_message,
    capture_runtime_output,
    check_bitsandbytes_available,
    check_flash_attention_available,
    count_parameters,
    ensure_dir,
    find_all_linear_names,
    format_captured_output_message,
    get_device,
    get_gpu_memory_usage,
    get_model_size,
    set_seed,
    setup_logging,
    suppress_warnings,
    verbose_logging_enabled,
)


class TestSetSeed:
    """Tests for set_seed function."""

    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible results."""
        set_seed(42)
        random_val1 = random.random()
        np_val1 = np.random.rand()
        torch_val1 = torch.rand(1).item()

        set_seed(42)
        random_val2 = random.random()
        np_val2 = np.random.rand()
        torch_val2 = torch.rand(1).item()

        assert random_val1 == random_val2
        assert np_val1 == np_val2
        assert torch_val1 == torch_val2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        val1 = torch.rand(1).item()

        set_seed(123)
        val2 = torch.rand(1).item()

        assert val1 != val2


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a logger."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_level(self):
        """Test that setup_logging accepts level parameter."""
        # Just verify the function accepts the level parameter without error
        logger = setup_logging(level="WARNING")
        assert logger is not None

    def test_setup_logging_updates_verbose_state_with_existing_root_handlers(self, monkeypatch):
        """Test verbose state follows the requested level even if logging is already configured."""
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level
        original_verbose = utils_module._VERBOSE_LOGGING_ENABLED

        try:
            logging.basicConfig(level=logging.WARNING)
            setup_logging(level="INFO")

            assert root_logger.getEffectiveLevel() == logging.INFO
            assert verbose_logging_enabled() is True
        finally:
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)
            monkeypatch.setattr(utils_module, "_VERBOSE_LOGGING_ENABLED", original_verbose)


class TestSuppressWarnings:
    """Tests for suppress_warnings function."""

    def test_sets_transformers_logger_handler_and_disables_propagation(self):
        """Ensure transformers warnings don't propagate to root handlers."""
        transformers_logger = logging.getLogger("transformers")
        original_handlers = transformers_logger.handlers[:]
        original_level = transformers_logger.level
        original_propagate = transformers_logger.propagate

        try:
            suppress_warnings()

            assert len(transformers_logger.handlers) == 1
            assert isinstance(transformers_logger.handlers[0], RichWarningHandler)
            assert transformers_logger.propagate is False
        finally:
            transformers_logger.handlers = original_handlers
            transformers_logger.setLevel(original_level)
            transformers_logger.propagate = original_propagate

    def test_showwarning_filters_generation_flags_warning(self, monkeypatch):
        """Ensure non-actionable generation warnings are suppressed."""
        original_showwarning = warnings.showwarning
        captured = []

        try:
            suppress_warnings()
            monkeypatch.setattr(
                "lora_finetune.utils.console.print",
                lambda message: captured.append(message),
            )

            warnings.showwarning(
                "generation flags are not valid for this model",
                UserWarning,
                "test_file.py",
                1,
            )

            assert captured == []
        finally:
            warnings.showwarning = original_showwarning

    def test_normalize_warning_message_summarizes_load_report(self):
        """Ensure multiline load reports are summarized to a compact single line."""
        msg = (
            "\x1b[1mViTForImageClassification LOAD REPORT\x1b[0m from: model\n"
            "Key               | Status   | Info\n"
            "classifier.weight | MISMATCH | Reinit\n"
            "classifier.bias   | MISMATCH | Reinit\n"
        )

        normalized = _normalize_warning_message(msg)

        assert normalized == (
            "Checkpoint shape mismatch detected; reinitialized: classifier.weight, classifier.bias"
        )

    def test_normalize_warning_message_removes_ansi_and_newlines(self):
        """Ensure ANSI artifacts and awkward line breaks are cleaned."""
        msg = "[1mSome warning[0m\n  with extra spacing\n"

        normalized = _normalize_warning_message(msg)

        assert normalized == "Some warning with extra spacing"

    def test_rich_warning_handler_handles_warning_category_args(self):
        """Ensure logging records with warning-category args do not crash formatting."""

        class CaptureConsole:
            def __init__(self):
                self.messages = []

            def print(self, message):
                self.messages.append(message)

        capture_console = CaptureConsole()
        handler = RichWarningHandler(rich_console=capture_console)
        record = logging.LogRecord(
            name="transformers",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="Deprecated attention mask path in use",
            args=(FutureWarning,),
            exc_info=None,
        )

        handler.emit(record)

        assert len(capture_console.messages) == 1
        assert capture_console.messages[0].plain == "  ⚠ Deprecated attention mask path in use"

    def test_format_captured_output_suppresses_unsloth_banner(self):
        """Ensure Unsloth startup banners are hidden in normal mode."""
        assert (
            format_captured_output_message(
                "Unsloth: Will patch your computer to enable 2x faster free finetuning."
            )
            is None
        )

    def test_format_captured_output_compacts_sharded_warning(self):
        """Ensure long sharding warnings are compacted to a single line preview."""
        message = (
            "The following layers were not sharded: model.layers.0.weight, model.layers.1.weight, "
            "lm_head.weight, model.norm.weight, model.embed_tokens.weight"
        )

        assert format_captured_output_message(message) == (
            "Some layers were not sharded: model.layers.0.weight, model.layers.1.weight, "
            "lm_head.weight, model.norm.weight, ..."
        )

    def test_capture_runtime_output_rerenders_and_deduplicates_messages(self):
        """Ensure captured third-party output is normalized and rendered once."""

        class CaptureConsole:
            def __init__(self):
                self.messages = []

            def print(self, message):
                self.messages.append(message)

        capture_console = CaptureConsole()

        with capture_runtime_output(enabled=True, rich_console=capture_console):
            print("Unsloth: Padding-free auto-enabled, enabling faster training.")
            print(
                "Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = 0.05."
            )
            print(
                "Unsloth: Dropout = 0 is supported for fast patching. You are using dropout = 0.05."
            )

        assert [message.plain for message in capture_console.messages] == [
            "  ⚠ Unsloth fast patching is partially disabled because LoRA dropout is non-zero; expect lower performance. Set LoRA dropout to 0 for full fast patching."
        ]

    def test_capture_runtime_output_preserves_unmatched_output(self, capsys):
        """Ensure unknown third-party output still reaches the original stream."""

        class CaptureConsole:
            def __init__(self):
                self.messages = []

            def print(self, message):
                self.messages.append(message)

        capture_console = CaptureConsole()

        with capture_runtime_output(enabled=True, rich_console=capture_console):
            print("Dependency emitted an unmatched diagnostic")

        captured = capsys.readouterr()

        assert "Dependency emitted an unmatched diagnostic" in captured.out
        assert capture_console.messages == []

    def test_capture_runtime_output_unwraps_rich_proxy_for_passthrough(self, monkeypatch):
        """Ensure passthrough writes use the underlying stream instead of Rich's proxy."""

        class CaptureConsole:
            def __init__(self):
                self.messages = []

            def print(self, message):
                self.messages.append(message)

        class FakeRichProxy:
            def __init__(self, target):
                self.rich_proxied_file = target
                self.encoding = "utf-8"

            def write(self, text):
                raise AssertionError("capture should write to rich_proxied_file, not the proxy")

            def flush(self):
                return None

        underlying = io.StringIO()
        proxy = FakeRichProxy(underlying)
        monkeypatch.setattr(sys, "stdout", proxy)
        monkeypatch.setattr(sys, "stderr", proxy)
        capture_console = CaptureConsole()

        with capture_runtime_output(enabled=True, rich_console=capture_console):
            print("Dependency emitted an unmatched diagnostic")

        assert underlying.getvalue() == "Dependency emitted an unmatched diagnostic\n"
        assert capture_console.messages == []

    def test_capture_runtime_output_drops_carriage_return_spinner_redraws(self, monkeypatch):
        """Ensure spinner-style carriage returns do not become repeated output lines."""

        class CaptureConsole:
            def __init__(self):
                self.messages = []

            def print(self, message):
                self.messages.append(message)

        underlying = io.StringIO()
        monkeypatch.setattr(sys, "stdout", underlying)
        monkeypatch.setattr(sys, "stderr", underlying)
        capture_console = CaptureConsole()

        with capture_runtime_output(enabled=True, rich_console=capture_console):
            sys.stdout.write("⠋ Loading model...\r")
            sys.stdout.write("⠙ Loading model...\r")
            print("Dependency emitted an unmatched diagnostic")

        assert underlying.getvalue() == "Dependency emitted an unmatched diagnostic\n"
        assert capture_console.messages == []

    def test_capture_runtime_output_merges_multiline_sharded_warning(self):
        """Ensure split sharded-layer diagnostics render as one compact warning."""

        class CaptureConsole:
            def __init__(self):
                self.messages = []

            def print(self, message):
                self.messages.append(message)

        capture_console = CaptureConsole()

        with capture_runtime_output(enabled=True, rich_console=capture_console):
            print("The following layers were not sharded: model.embed_tokens.weight,")
            print("lm_head.weight,")
            print("model.norm.weight, model.layers.0.input_layernorm.weight")

        assert [message.plain for message in capture_console.messages] == [
            "  ⚠ Some layers were not sharded: model.embed_tokens.weight, lm_head.weight, model.norm.weight, model.layers.0.input_layernorm.weight"
        ]


class TestGetDevice:
    """Tests for get_device function."""

    def test_get_device_returns_device(self):
        """Test that get_device returns a torch device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_valid_type(self):
        """Test that device type is valid."""
        device = get_device()
        assert device.type in ["cuda", "mps", "cpu"]


class TestGetGpuMemoryUsage:
    """Tests for get_gpu_memory_usage function."""

    def test_get_gpu_memory_usage_returns_dict(self):
        """Test that function returns a dictionary."""
        stats = get_gpu_memory_usage()
        assert isinstance(stats, dict)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_gpu_memory_usage_with_cuda(self):
        """Test GPU memory stats when CUDA is available."""
        stats = get_gpu_memory_usage()
        assert len(stats) > 0
        for gpu_name, info in stats.items():
            assert "allocated_gb" in info
            assert "reserved_gb" in info
            assert "total_gb" in info
            assert "free_gb" in info


class TestGetModelSize:
    """Tests for get_model_size function."""

    def test_get_model_size_simple_model(self):
        """Test model size calculation for a simple model."""
        model = torch.nn.Linear(10, 5)
        stats = get_model_size(model)

        assert "total_params" in stats
        assert "trainable_params" in stats
        assert "trainable_percent" in stats
        assert "param_size_mb" in stats

        # Linear(10, 5) has 10*5 + 5 = 55 parameters
        assert stats["total_params"] == 55
        assert stats["trainable_params"] == 55
        assert stats["trainable_percent"] == 100.0

    def test_get_model_size_frozen_params(self):
        """Test model size with frozen parameters."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.Linear(5, 2),
        )
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        stats = get_model_size(model)

        # First layer: 10*5 + 5 = 55, Second layer: 5*2 + 2 = 12
        assert stats["total_params"] == 67
        assert stats["trainable_params"] == 12
        assert stats["trainable_percent"] < 100.0


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        total, trainable = count_parameters(model)

        assert total == 55
        assert trainable == 55

    def test_count_parameters_with_frozen(self):
        """Test parameter counting with frozen layers."""
        model = torch.nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False

        total, trainable = count_parameters(model)

        assert total == 55
        assert trainable == 0


class TestFindAllLinearNames:
    """Tests for find_all_linear_names function."""

    def test_find_linear_names_simple(self):
        """Test finding linear layer names in a simple model."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )
        linear_names = find_all_linear_names(model)

        # Should find the linear layers (named by their index in Sequential)
        assert len(linear_names) > 0

    def test_find_linear_names_nested(self):
        """Test finding linear layer names in a nested model."""

        class NestedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(10, 5),
                    torch.nn.ReLU(),
                )
                self.decoder = torch.nn.Linear(5, 2)

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = NestedModel()
        linear_names = find_all_linear_names(model)

        assert len(linear_names) > 0


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_ensure_dir_creates_directory(self, tmp_path):
        """Test that ensure_dir creates a directory."""
        new_dir = tmp_path / "new_directory"
        result = ensure_dir(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == str(new_dir)

    def test_ensure_dir_existing_directory(self, tmp_path):
        """Test that ensure_dir works with existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensure_dir(str(existing_dir))

        assert existing_dir.exists()
        assert result == str(existing_dir)

    def test_ensure_dir_nested(self, tmp_path):
        """Test that ensure_dir creates nested directories."""
        nested_dir = tmp_path / "a" / "b" / "c"
        result = ensure_dir(str(nested_dir))

        assert nested_dir.exists()
        assert result == str(nested_dir)


class TestCheckAvailability:
    """Tests for availability check functions."""

    def test_check_flash_attention_available(self):
        """Test flash attention availability check."""
        result = check_flash_attention_available()
        assert isinstance(result, bool)

    def test_check_bitsandbytes_available(self):
        """Test bitsandbytes availability check."""
        result = check_bitsandbytes_available()
        assert isinstance(result, bool)
