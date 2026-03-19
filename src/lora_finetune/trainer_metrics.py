import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from transformers import EvalPrediction

from .config import GRPOConfig as ProjectGRPOConfig


def compute_metrics_for_classification(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute accuracy metrics for classification tasks."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def compute_metrics_for_lm(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute perplexity for language modeling tasks."""
    logits, labels = eval_pred

    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    shift_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    shift_labels = shift_labels.reshape(-1)

    mask = shift_labels != -100
    if mask.sum() == 0:
        return {"perplexity": float("inf")}

    max_logits = np.max(shift_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shift_logits - max_logits)
    softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    valid_labels = np.where(mask, shift_labels, 0).astype(np.int64)
    probs = softmax[np.arange(len(valid_labels)), valid_labels]

    log_probs = np.log(probs + 1e-10)
    loss = -np.sum(log_probs * mask) / mask.sum()

    perplexity = np.exp(loss)
    return {"perplexity": float(perplexity)}


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if content is not None:
                    parts.append(str(content))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(completion, dict):
        content = completion.get("content")
        if content is not None:
            return str(content)
    return str(completion)


def build_grpo_reward_functions(
    grpo_config: ProjectGRPOConfig,
) -> List[Callable[..., List[Optional[float]]]]:
    reward_funcs: List[Callable[..., List[Optional[float]]]] = []
    reward_regex = re.compile(grpo_config.reward_regex) if grpo_config.reward_regex else None

    for reward_name in grpo_config.reward_funcs:
        if reward_name == "non_empty":

            def non_empty_reward(completions, **kwargs):
                return [
                    1.0 if _completion_to_text(completion).strip() else 0.0
                    for completion in completions
                ]

            reward_funcs.append(non_empty_reward)
            continue

        if reward_name == "length":

            def length_reward(completions, **kwargs):
                return [
                    float(len(_completion_to_text(completion).strip()))
                    for completion in completions
                ]

            reward_funcs.append(length_reward)
            continue

        if reward_name == "exact_match":

            def exact_match_reward(completions, **kwargs):
                references = kwargs.get(grpo_config.reward_column)
                if references is None:
                    raise ValueError(
                        f"GRPO exact_match reward requires dataset column '{grpo_config.reward_column}'"
                    )
                return [
                    1.0
                    if _completion_to_text(completion).strip() == str(reference).strip()
                    else 0.0
                    for completion, reference in zip(completions, references)
                ]

            reward_funcs.append(exact_match_reward)
            continue

        if reward_name == "regex":
            if reward_regex is None:
                raise ValueError("grpo.reward_regex must be set when using the regex reward")

            def regex_reward(completions, **kwargs):
                return [
                    1.0 if reward_regex.search(_completion_to_text(completion)) else 0.0
                    for completion in completions
                ]

            reward_funcs.append(regex_reward)
            continue

        raise ValueError(
            f"Unsupported GRPO reward function '{reward_name}'. "
            "Supported values: non_empty, length, exact_match, regex"
        )

    return reward_funcs
