"""GSM8K benchmark evaluator with generation-based evaluation."""

import logging
import re
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)

GSM8K_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form "#### <answer>" where <answer> is just the final number.

Question: {question}

Answer:"""


def extract_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from model output.

    GSM8K answers are formatted as "#### <number>" at the end.
    """
    # Look for #### followed by a number
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        # Remove commas from numbers like "1,000"
        return match.group(1).replace(",", "")

    # Fallback: try to find the last number in the text
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove commas and whitespace
    answer = answer.replace(",", "").strip()
    # Try to convert to float and back to handle decimal variations
    try:
        num = float(answer)
        # Return integer if it's a whole number
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer


class GSM8KEvaluator:
    """Evaluator for GSM8K math reasoning benchmark."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 512,
        batch_size: int = 1,
        num_samples: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load_dataset(self, split: str = "test") -> List[Dict[str, Any]]:
        """Load GSM8K dataset."""
        dataset = load_dataset("openai/gsm8k", "main", split=split)

        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        return list(dataset)

    def format_prompt(self, question: str) -> str:
        """Format question with prompt template."""
        return GSM8K_PROMPT_TEMPLATE.format(question=question)

    def extract_reference_answer(self, answer_text: str) -> str:
        """Extract reference answer from GSM8K answer field.

        GSM8K answers contain step-by-step solution ending with #### <number>.
        """
        return extract_answer(answer_text)

    def _get_max_length(self) -> int:
        """Get max sequence length from model config."""
        config = self.model.config
        for attr in ["max_position_embeddings", "n_positions", "max_seq_len", "seq_length"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        return 4096  # Fallback default

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for batches of prompts."""
        self.model.eval()

        max_length = self._get_max_length()
        max_input_length = max_length - self.max_new_tokens

        # Ensure left-padding for batched generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        responses = []
        for i in trange(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for deterministic evaluation
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            # Decode only the generated part for each sample
            input_length = inputs["input_ids"].shape[1]
            for output in outputs:
                generated_ids = output[input_length:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                responses.append(response)

        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        return responses

    def evaluate(self, split: str = "test") -> Dict[str, Any]:
        """Run evaluation on GSM8K dataset.

        Returns:
            Dictionary with accuracy, correct count, total count, and detailed results.
        """
        logger.info(f"Loading GSM8K {split} split...")
        dataset = self.load_dataset(split)

        total = len(dataset)
        results = []

        # Prepare all prompts and reference answers
        prompts = []
        reference_answers = []
        questions = []
        for example in dataset:
            questions.append(example["question"])
            prompts.append(self.format_prompt(example["question"]))
            reference_answers.append(self.extract_reference_answer(example["answer"]))

        # Generate all responses in batches
        logger.info(f"Evaluating {total} examples (batch_size={self.batch_size})...")
        responses = self.generate(prompts)

        # Score results
        correct = 0
        for i, response in enumerate(tqdm(responses, desc="Scoring")):
            predicted_answer = extract_answer(response)

            # Normalize and compare
            pred_normalized = normalize_answer(predicted_answer)
            ref_normalized = normalize_answer(reference_answers[i])
            is_correct = pred_normalized == ref_normalized

            if is_correct:
                correct += 1

            results.append(
                {
                    "question": questions[i],
                    "reference_answer": reference_answers[i],
                    "predicted_answer": predicted_answer,
                    "full_response": response,
                    "is_correct": is_correct,
                }
            )

        accuracy = correct / total if total > 0 else 0.0

        logger.info(f"GSM8K Accuracy: {accuracy:.2%} ({correct}/{total})")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }


class GSM8KCallback(TrainerCallback):
    """Callback to run GSM8K evaluation during training."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        eval_steps: int = 500,
        num_samples: Optional[int] = 100,
        max_new_tokens: int = 512,
        batch_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._dataset = None

    def _get_dataset(self) -> List[Dict[str, Any]]:
        """Load and cache the dataset."""
        if self._dataset is None:
            dataset = load_dataset("openai/gsm8k", "main", split="test")
            if self.num_samples is not None:
                dataset = dataset.select(range(min(self.num_samples, len(dataset))))
            self._dataset = list(dataset)
        return self._dataset

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel = None,
        **kwargs,
    ):
        """Run GSM8K evaluation at specified intervals."""
        if state.global_step % self.eval_steps != 0 or state.global_step == 0:
            return

        if model is None:
            return

        logger.info(f"Running GSM8K benchmark evaluation at step {state.global_step}")

        evaluator = GSM8KEvaluator(
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
        )

        results = evaluator.evaluate(split="test")

        # Log to trainer's log history
        metrics = {
            "gsm8k_accuracy": results["accuracy"],
            "gsm8k_correct": results["correct"],
            "gsm8k_total": results["total"],
        }

        # Log metrics (will be picked up by wandb/tensorboard if configured)
        if state.log_history is not None:
            state.log_history.append({"step": state.global_step, **metrics})

        logger.info(
            f"GSM8K @ step {state.global_step}: "
            f"{results['accuracy']:.2%} ({results['correct']}/{results['total']})"
        )
