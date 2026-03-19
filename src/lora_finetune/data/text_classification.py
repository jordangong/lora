from functools import partial
from typing import Any, Dict, List, Optional

from datasets import DatasetDict
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from ..config import DataConfig
from .text_preprocessing import shuffle_dataset_split


def _get_ordered_classification_labels(
    dataset: DatasetDict,
    label_column: str,
    train_split: str,
) -> List[Any]:
    train_dataset = dataset[train_split]

    labels = None
    if hasattr(train_dataset, "unique"):
        try:
            labels = list(train_dataset.unique(label_column))
        except Exception:
            labels = None

    if labels is None:
        seen_labels = set()
        labels = []
        for example in train_dataset:
            label = example[label_column]
            if label not in seen_labels:
                seen_labels.add(label)
                labels.append(label)

    if not labels:
        raise ValueError(f"Could not infer labels from column '{label_column}'")

    try:
        return sorted(labels)
    except TypeError:
        return labels


def _labels_are_zero_based_contiguous_integers(labels: List[Any]) -> bool:
    if not all(isinstance(label, int) and not isinstance(label, bool) for label in labels):
        return False

    return labels == list(range(len(labels)))


def _build_classification_label_to_id(
    dataset: DatasetDict,
    label_column: str,
    train_split: str,
) -> Optional[Dict[Any, int]]:
    labels = _get_ordered_classification_labels(dataset, label_column, train_split)

    if _labels_are_zero_based_contiguous_integers(labels):
        return None

    return {label: idx for idx, label in enumerate(labels)}


def tokenize_classification_function(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_column: str,
    label_column: str,
    label_to_id: Optional[Dict[Any, int]] = None,
) -> Dict[str, Any]:
    tokenized = tokenizer(
        examples[text_column],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    labels = examples[label_column]
    if label_to_id is not None:
        labels = [label_to_id[label] for label in labels]
    tokenized["label"] = labels
    return tokenized


def preprocess_text_classification_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
    shuffle_seed: Optional[int] = None,
) -> DatasetDict:
    label_to_id = _build_classification_label_to_id(
        dataset,
        config.label_column,
        config.train_split,
    )

    tokenized_splits = {}
    for split_name, split_data in dataset.items():
        columns = split_data.column_names
        if config.text_column not in columns:
            raise ValueError(
                f"Text column '{config.text_column}' not found in split '{split_name}'. "
                f"Available columns: {columns}."
            )
        if config.label_column not in columns:
            raise ValueError(
                f"Label column '{config.label_column}' not found in split '{split_name}'. "
                f"Available columns: {columns}."
            )

        tokenize_fn = partial(
            tokenize_classification_function,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            text_column=config.text_column,
            label_column=config.label_column,
            label_to_id=label_to_id,
        )
        tokenized_splits[split_name] = split_data.map(
            tokenize_fn,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=columns,
            desc=f"Tokenizing classification split ({split_name})",
        )

    tokenized_dataset = DatasetDict(tokenized_splits)

    if config.max_train_samples:
        train_dataset = shuffle_dataset_split(tokenized_dataset[config.train_split], shuffle_seed)
        tokenized_dataset[config.train_split] = train_dataset.select(
            range(min(config.max_train_samples, len(train_dataset)))
        )

    if config.validation_split in tokenized_dataset and config.max_eval_samples:
        eval_dataset = shuffle_dataset_split(
            tokenized_dataset[config.validation_split], shuffle_seed
        )
        tokenized_dataset[config.validation_split] = eval_dataset.select(
            range(min(config.max_eval_samples, len(tokenized_dataset[config.validation_split])))
        )

    return tokenized_dataset


def get_text_classification_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )
