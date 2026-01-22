"""Tests for benchmark evaluators."""

from unittest.mock import MagicMock, patch

import torch

from lora_finetune.evaluators import GSM8KCallback, GSM8KEvaluator
from lora_finetune.evaluators.gsm8k import GSM8K_PROMPT_TEMPLATE, extract_answer, normalize_answer


class TestExtractAnswer:
    """Tests for extract_answer function."""

    def test_extract_answer_with_hash_format(self):
        """Test extracting answer with #### format."""
        text = "The total is 5 + 3 = 8. #### 8"
        assert extract_answer(text) == "8"

    def test_extract_answer_with_hash_format_spaces(self):
        """Test extracting answer with spaces after ####."""
        text = "So the answer is ####   42"
        assert extract_answer(text) == "42"

    def test_extract_answer_with_commas(self):
        """Test extracting answer with comma-separated number."""
        text = "The result is #### 1,234"
        assert extract_answer(text) == "1234"

    def test_extract_answer_with_decimal(self):
        """Test extracting answer with decimal number."""
        text = "The average is #### 3.14"
        assert extract_answer(text) == "3.14"

    def test_extract_answer_negative_number(self):
        """Test extracting negative answer."""
        text = "The difference is #### -15"
        assert extract_answer(text) == "-15"

    def test_extract_answer_fallback_to_last_number(self):
        """Test fallback to last number when no #### format."""
        text = "First we get 10, then 20, finally the answer is 30"
        assert extract_answer(text) == "30"

    def test_extract_answer_no_number(self):
        """Test returns None when no number found."""
        text = "No numbers here"
        assert extract_answer(text) is None

    def test_extract_answer_empty_string(self):
        """Test empty string returns None."""
        assert extract_answer("") is None

    def test_extract_answer_complex_solution(self):
        """Test extracting from realistic GSM8K-style solution."""
        text = """Let me solve this step by step.
        First, we have 5 apples.
        Then we add 3 more: 5 + 3 = 8
        Finally, we double it: 8 * 2 = 16
        #### 16"""
        assert extract_answer(text) == "16"


class TestNormalizeAnswer:
    """Tests for normalize_answer function."""

    def test_normalize_integer(self):
        """Test normalizing integer string."""
        assert normalize_answer("42") == "42"

    def test_normalize_with_commas(self):
        """Test removing commas from number."""
        assert normalize_answer("1,234,567") == "1234567"

    def test_normalize_with_whitespace(self):
        """Test stripping whitespace."""
        assert normalize_answer("  42  ") == "42"

    def test_normalize_decimal_whole_number(self):
        """Test that 10.0 becomes 10."""
        assert normalize_answer("10.0") == "10"

    def test_normalize_decimal_with_fraction(self):
        """Test preserving decimal fraction."""
        assert normalize_answer("3.14") == "3.14"

    def test_normalize_negative(self):
        """Test normalizing negative number."""
        assert normalize_answer("-5") == "-5"

    def test_normalize_none(self):
        """Test None returns empty string."""
        assert normalize_answer(None) == ""

    def test_normalize_non_numeric(self):
        """Test non-numeric string returns as-is."""
        assert normalize_answer("abc") == "abc"


class TestGSM8KPromptTemplate:
    """Tests for GSM8K prompt template."""

    def test_prompt_template_format(self):
        """Test prompt template has required placeholders."""
        assert "{question}" in GSM8K_PROMPT_TEMPLATE

    def test_prompt_template_contains_instructions(self):
        """Test prompt template contains step-by-step instructions."""
        assert "step by step" in GSM8K_PROMPT_TEMPLATE.lower()
        assert "####" in GSM8K_PROMPT_TEMPLATE


class TestGSM8KEvaluator:
    """Tests for GSM8KEvaluator class."""

    def test_evaluator_initialization(self):
        """Test GSM8KEvaluator initialization."""
        model = MagicMock()
        tokenizer = MagicMock()

        evaluator = GSM8KEvaluator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            batch_size=2,
            num_samples=10,
        )

        assert evaluator.model is model
        assert evaluator.tokenizer is tokenizer
        assert evaluator.max_new_tokens == 256
        assert evaluator.batch_size == 2
        assert evaluator.num_samples == 10

    def test_evaluator_default_device_cuda(self):
        """Test default device is cuda when available."""
        model = MagicMock()
        tokenizer = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer)
            assert evaluator.device == "cuda"

    def test_evaluator_default_device_cpu(self):
        """Test default device is cpu when cuda unavailable."""
        model = MagicMock()
        tokenizer = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer)
            assert evaluator.device == "cpu"

    def test_evaluator_custom_device(self):
        """Test custom device override."""
        model = MagicMock()
        tokenizer = MagicMock()

        evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer, device="mps")
        assert evaluator.device == "mps"

    def test_format_prompt(self):
        """Test prompt formatting."""
        model = MagicMock()
        tokenizer = MagicMock()

        evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer)
        prompt = evaluator.format_prompt("What is 2 + 2?")

        assert "What is 2 + 2?" in prompt
        assert "step by step" in prompt.lower()

    def test_extract_reference_answer(self):
        """Test extracting reference answer from GSM8K format."""
        model = MagicMock()
        tokenizer = MagicMock()

        evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer)
        answer = evaluator.extract_reference_answer(
            "Step 1: 5 + 3 = 8\nStep 2: 8 * 2 = 16\n#### 16"
        )

        assert answer == "16"

    def test_get_max_length_from_config(self):
        """Test getting max length from model config."""
        model = MagicMock()
        model.config.max_position_embeddings = 8192
        tokenizer = MagicMock()

        evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer)
        assert evaluator._get_max_length() == 8192

    def test_get_max_length_fallback(self):
        """Test fallback max length when config has no attribute."""
        model = MagicMock()
        # Remove all possible max length attributes
        model.config = MagicMock(spec=[])
        tokenizer = MagicMock()

        evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer)
        assert evaluator._get_max_length() == 4096  # Fallback default

    @patch("lora_finetune.evaluators.gsm8k.load_dataset")
    def test_load_dataset(self, mock_load_dataset):
        """Test dataset loading."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load_dataset.return_value = mock_dataset

        model = MagicMock()
        tokenizer = MagicMock()
        evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer, num_samples=10)

        evaluator.load_dataset("test")

        mock_load_dataset.assert_called_once_with("openai/gsm8k", "main", split="test")
        mock_dataset.select.assert_called_once()

    @patch("lora_finetune.evaluators.gsm8k.load_dataset")
    def test_load_dataset_no_limit(self, mock_load_dataset):
        """Test dataset loading without sample limit."""
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))
        mock_load_dataset.return_value = mock_dataset

        model = MagicMock()
        tokenizer = MagicMock()
        evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer, num_samples=None)

        evaluator.load_dataset("test")

        mock_load_dataset.assert_called_once()
        mock_dataset.select.assert_not_called()


class TestGSM8KCallback:
    """Tests for GSM8KCallback class."""

    def test_callback_initialization(self):
        """Test GSM8KCallback initialization."""
        tokenizer = MagicMock()

        callback = GSM8KCallback(
            tokenizer=tokenizer,
            eval_steps=100,
            num_samples=50,
            max_new_tokens=256,
            batch_size=4,
        )

        assert callback.tokenizer is tokenizer
        assert callback.eval_steps == 100
        assert callback.num_samples == 50
        assert callback.max_new_tokens == 256
        assert callback.batch_size == 4
        assert callback._dataset is None

    def test_callback_default_values(self):
        """Test GSM8KCallback default values."""
        tokenizer = MagicMock()

        callback = GSM8KCallback(tokenizer=tokenizer)

        assert callback.eval_steps == 500
        assert callback.num_samples == 100
        assert callback.max_new_tokens == 512
        assert callback.batch_size == 1

    def test_callback_has_required_methods(self):
        """Test callback has TrainerCallback methods."""
        tokenizer = MagicMock()
        callback = GSM8KCallback(tokenizer=tokenizer)

        assert hasattr(callback, "on_step_end")
        assert callable(callback.on_step_end)

    def test_on_step_end_skips_non_eval_steps(self):
        """Test that on_step_end skips when not at eval_steps interval."""
        tokenizer = MagicMock()
        callback = GSM8KCallback(tokenizer=tokenizer, eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 50  # Not a multiple of 100
        control = MagicMock()
        model = MagicMock()

        # Should not raise and should return early
        callback.on_step_end(args, state, control, model=model)

    def test_on_step_end_skips_step_zero(self):
        """Test that on_step_end skips at step 0."""
        tokenizer = MagicMock()
        callback = GSM8KCallback(tokenizer=tokenizer, eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 0
        control = MagicMock()
        model = MagicMock()

        # Should not raise and should return early
        callback.on_step_end(args, state, control, model=model)

    def test_on_step_end_skips_when_no_model(self):
        """Test that on_step_end skips when model is None."""
        tokenizer = MagicMock()
        callback = GSM8KCallback(tokenizer=tokenizer, eval_steps=100)

        args = MagicMock()
        state = MagicMock()
        state.global_step = 100
        control = MagicMock()

        # Should not raise and should return early
        callback.on_step_end(args, state, control, model=None)

    @patch("lora_finetune.evaluators.gsm8k.load_dataset")
    def test_get_dataset_caches(self, mock_load_dataset):
        """Test that _get_dataset caches the dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=1000)
        mock_dataset.__iter__ = MagicMock(return_value=iter([{"q": "test"}]))
        mock_load_dataset.return_value = mock_dataset

        tokenizer = MagicMock()
        callback = GSM8KCallback(tokenizer=tokenizer, num_samples=10)

        # Call twice
        callback._get_dataset()
        callback._get_dataset()

        # Should only load once
        assert mock_load_dataset.call_count == 1


class TestGSM8KEvaluatorGenerate:
    """Tests for GSM8KEvaluator.generate method."""

    def test_generate_sets_left_padding(self):
        """Test that generate uses left padding for batch generation."""
        model = MagicMock()
        model.config.max_position_embeddings = 4096
        model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))

        tokenizer = MagicMock()
        tokenizer.padding_side = "right"
        tokenizer.return_tensors = "pt"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        # Mock tokenizer call to return proper tensor structure
        mock_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        tokenizer.return_value = MagicMock(
            to=MagicMock(return_value=mock_inputs),
        )
        tokenizer.return_value.__getitem__ = lambda self, key: mock_inputs[key]
        tokenizer.decode = MagicMock(return_value="#### 42")

        evaluator = GSM8KEvaluator(model=model, tokenizer=tokenizer, device="cpu", batch_size=1)

        evaluator.generate(["Test prompt"])

        # Verify padding side was set to left during generation
        assert tokenizer.padding_side == "right"  # Should be restored


class TestIntegration:
    """Integration tests for evaluator components."""

    def test_answer_extraction_and_normalization_pipeline(self):
        """Test full pipeline of extracting and normalizing answers."""
        # Simulate model output with various formats
        test_cases = [
            ("The answer is #### 42", "42", "42"),
            ("Result: #### 1,000", "1000", "1000"),
            ("#### 3.0", "3.0", "3"),  # 3.0 normalizes to 3
            ("No hash format, just 99", "99", "99"),
        ]

        for text, expected_extracted, expected_normalized in test_cases:
            extracted = extract_answer(text)
            normalized = normalize_answer(extracted)
            assert extracted == expected_extracted, f"Failed extraction for: {text}"
            assert normalized == expected_normalized, f"Failed normalization for: {text}"

    def test_answer_comparison_matching(self):
        """Test that answer comparison works correctly for matching answers."""
        # Pairs that should match after normalization
        matching_pairs = [
            ("42", "42"),
            ("1,000", "1000"),
            ("3.0", "3"),
            (" 100 ", "100"),
            ("-5", "-5"),
        ]

        for ans1, ans2 in matching_pairs:
            norm1 = normalize_answer(ans1)
            norm2 = normalize_answer(ans2)
            assert norm1 == norm2, f"Expected {ans1} and {ans2} to match"

    def test_answer_comparison_non_matching(self):
        """Test that different answers don't match."""
        non_matching_pairs = [
            ("42", "43"),
            ("100", "1000"),
            ("3.14", "3"),
        ]

        for ans1, ans2 in non_matching_pairs:
            norm1 = normalize_answer(ans1)
            norm2 = normalize_answer(ans2)
            assert norm1 != norm2, f"Expected {ans1} and {ans2} to NOT match"
