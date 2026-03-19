import logging
from functools import partial
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
)

from ..config import DataConfig
from .text_formatting import (
    DEFAULT_PROMPT_TEMPLATE,
    SOURCE_TEXT_COLUMN,
    format_instruction,
    format_instruction_as_prompt,
    format_instruction_as_prompt_completion,
    format_instruction_with_source,
    format_qa,
    format_qa_as_prompt,
    format_qa_as_prompt_completion,
    format_qa_with_source,
    normalize_conversations_to_messages,
)

logger = logging.getLogger(__name__)


def load_text_dataset(config: DataConfig) -> DatasetDict:
    """Load text dataset from HuggingFace Hub or local files."""
    if config.dataset_name:
        logger.info(f"Loading dataset: {config.dataset_name}")
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            streaming=config.streaming,
        )
    elif config.train_file:
        logger.info(f"Loading dataset from file: {config.train_file}")
        data_files = {"train": config.train_file}
        if config.validation_file:
            data_files["validation"] = config.validation_file

        extension = config.train_file.split(".")[-1]
        if extension == "jsonl":
            extension = "json"

        dataset = load_dataset(extension, data_files=data_files)
    else:
        raise ValueError("Either dataset_name or train_file must be provided")

    if config.eval_dataset_name is not None:
        logger.info(f"Loading holdout eval dataset: {config.eval_dataset_name}")
        eval_dataset = load_dataset(
            config.eval_dataset_name,
            config.eval_dataset_config_name,
            split=config.eval_dataset_split,
        )
        dataset = DatasetDict(
            {
                config.train_split: dataset[config.train_split],
                config.validation_split: eval_dataset,
            }
        )
    elif (
        config.validation_split not in dataset
        and config.eval_split_ratio is not None
        and config.eval_split_ratio > 0
    ):
        if not hasattr(dataset[config.train_split], "train_test_split"):
            raise ValueError(
                "data.eval_split_ratio requires a non-streaming dataset with train_test_split support. "
                "For streaming datasets, provide data.validation_file or data.eval_dataset_name instead."
            )
        logger.info(f"Splitting train data with eval_split_ratio={config.eval_split_ratio}")
        split_dataset = dataset[config.train_split].train_test_split(
            test_size=config.eval_split_ratio,
            seed=42,
        )
        dataset = DatasetDict(
            {
                config.train_split: split_dataset["train"],
                config.validation_split: split_dataset["test"],
            }
        )

    return dataset


def maybe_append_eos(text: str, tokenizer: PreTrainedTokenizer, append_eos_token: bool) -> str:
    eos_token = getattr(tokenizer, "eos_token", None)
    if not append_eos_token or not eos_token or text.endswith(eos_token):
        return text
    return f"{text}{eos_token}"


def tokenize_function(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_column: str = "text",
    source_text_column: Optional[str] = None,
    response_only_loss: bool = False,
    append_eos_token: bool = True,
) -> Dict[str, Any]:
    """Tokenize text examples."""
    texts = [maybe_append_eos(text, tokenizer, append_eos_token) for text in examples[text_column]]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    labels = [input_ids.copy() for input_ids in tokenized["input_ids"]]

    if response_only_loss and source_text_column and source_text_column in examples:
        source_tokenized = tokenizer(
            examples[source_text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        labels = [
            [-100] * min(len(source_ids), len(input_ids))
            + input_ids[min(len(source_ids), len(input_ids)) :].copy()
            for input_ids, source_ids in zip(tokenized["input_ids"], source_tokenized["input_ids"])
        ]

    tokenized["labels"] = labels
    return tokenized


def shuffle_dataset_split(split_data, seed: Optional[int]):
    """Shuffle a dataset split deterministically when supported."""
    if seed is None or not hasattr(split_data, "shuffle"):
        return split_data

    return split_data.shuffle(seed=seed)


def requires_trl_native_dataset(dataset: DatasetDict) -> bool:
    for split_data in dataset.values():
        columns = split_data.column_names
        if "messages" in columns or "conversations" in columns:
            return True
        if "prompt" in columns and "completion" in columns:
            return True
    return False


def _apply_sample_limits(
    dataset: DatasetDict,
    config: DataConfig,
    shuffle_seed: Optional[int] = None,
) -> DatasetDict:
    if config.max_train_samples:
        train_dataset = shuffle_dataset_split(dataset[config.train_split], shuffle_seed)
        dataset[config.train_split] = train_dataset.select(
            range(min(config.max_train_samples, len(train_dataset)))
        )

    if config.validation_split in dataset and config.max_eval_samples:
        eval_dataset = shuffle_dataset_split(dataset[config.validation_split], shuffle_seed)
        dataset[config.validation_split] = eval_dataset.select(
            range(min(config.max_eval_samples, len(dataset[config.validation_split])))
        )

    return dataset


def prepare_text_dataset_for_trl(
    dataset: DatasetDict,
    config: DataConfig,
    shuffle_seed: Optional[int] = None,
) -> DatasetDict:
    template = config.prompt_template or DEFAULT_PROMPT_TEMPLATE
    prepared_splits = {}

    for split_name, split_data in dataset.items():
        columns = split_data.column_names

        if "messages" in columns or ("prompt" in columns and "completion" in columns):
            prepared_splits[split_name] = split_data
            continue

        map_kwargs = {}
        if isinstance(split_data, Dataset):
            map_kwargs["num_proc"] = config.preprocessing_num_workers

        if "conversations" in columns:
            prepared_splits[split_name] = split_data.map(
                normalize_conversations_to_messages,
                remove_columns=["conversations"],
                **map_kwargs,
            )
            continue

        if config.text_column in columns:
            prepared_splits[split_name] = split_data
            continue

        map_kwargs["remove_columns"] = columns

        if "instruction" in columns:
            prepared_splits[split_name] = split_data.map(
                partial(format_instruction_as_prompt_completion, template=template),
                **map_kwargs,
            )
            continue

        if "question" in columns:
            prepared_splits[split_name] = split_data.map(
                format_qa_as_prompt_completion,
                **map_kwargs,
            )
            continue

        raise ValueError(
            f"Could not find text column '{config.text_column}' in split '{split_name}'. "
            "Provide data.text_column or use a dataset with instruction/question fields."
        )

    return _apply_sample_limits(DatasetDict(prepared_splits), config, shuffle_seed)


def prepare_preference_dataset_for_trl(
    dataset: DatasetDict,
    config: DataConfig,
    shuffle_seed: Optional[int] = None,
) -> DatasetDict:
    template = config.prompt_template or DEFAULT_PROMPT_TEMPLATE
    prepared_splits = {}

    for split_name, split_data in dataset.items():
        columns = split_data.column_names
        if {"prompt", "chosen", "rejected"}.issubset(columns):
            prepared_splits[split_name] = split_data
            continue

        map_kwargs = {}
        if isinstance(split_data, Dataset):
            map_kwargs["num_proc"] = config.preprocessing_num_workers

        if {"instruction", "chosen", "rejected"}.issubset(columns):
            prepared_splits[split_name] = split_data.map(
                partial(format_instruction_as_prompt, template=template),
                **map_kwargs,
            )
            continue

        if {"question", "chosen", "rejected"}.issubset(columns):
            prepared_splits[split_name] = split_data.map(
                format_qa_as_prompt,
                **map_kwargs,
            )
            continue

        raise ValueError(
            f"Split '{split_name}' must contain prompt/chosen/rejected columns, or "
            "instruction/question columns together with chosen/rejected."
        )

    return _apply_sample_limits(DatasetDict(prepared_splits), config, shuffle_seed)


def prepare_grpo_dataset_for_trl(
    dataset: DatasetDict,
    config: DataConfig,
    shuffle_seed: Optional[int] = None,
) -> DatasetDict:
    template = config.prompt_template or DEFAULT_PROMPT_TEMPLATE
    prepared_splits = {}

    for split_name, split_data in dataset.items():
        columns = split_data.column_names
        if "prompt" in columns:
            prepared_splits[split_name] = split_data
            continue

        map_kwargs = {}
        if isinstance(split_data, Dataset):
            map_kwargs["num_proc"] = config.preprocessing_num_workers

        if config.text_column in columns:
            prepared_splits[split_name] = split_data.map(
                lambda example, text_column=config.text_column: {"prompt": example[text_column]},
                **map_kwargs,
            )
            continue

        if "instruction" in columns:
            prepared_splits[split_name] = split_data.map(
                partial(format_instruction_as_prompt, template=template),
                **map_kwargs,
            )
            continue

        if "question" in columns:
            prepared_splits[split_name] = split_data.map(
                format_qa_as_prompt,
                **map_kwargs,
            )
            continue

        raise ValueError(
            f"Split '{split_name}' must contain a prompt column, the configured text column "
            f"'{config.text_column}', or instruction/question fields."
        )

    return _apply_sample_limits(DatasetDict(prepared_splits), config, shuffle_seed)


def preprocess_text_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
    shuffle_seed: Optional[int] = None,
) -> DatasetDict:
    """Preprocess text dataset for training."""
    logger.info(f"Preprocessing dataset with max_seq_length={config.max_seq_length}")
    template = config.prompt_template or DEFAULT_PROMPT_TEMPLATE

    tokenized_splits = {}
    for split_name, split_data in dataset.items():
        columns = split_data.column_names
        source_text_column = None

        if config.text_column in columns:
            logger.info(f"Using existing '{config.text_column}' column for {split_name}")
        elif "instruction" in columns:
            logger.info(f"Using instruction template formatting for {split_name}")
            format_fn = partial(
                format_instruction_with_source,
                template=template,
                output_column=config.text_column,
                source_column=SOURCE_TEXT_COLUMN,
            )
            if not config.response_only_loss:
                format_fn = partial(
                    format_instruction,
                    template=template,
                    output_column=config.text_column,
                )
            split_data = split_data.map(
                format_fn,
                remove_columns=[col for col in columns if col not in [config.text_column]],
                num_proc=config.preprocessing_num_workers,
                desc=f"Formatting instructions ({split_name})",
            )
            columns = split_data.column_names
            if config.response_only_loss:
                source_text_column = SOURCE_TEXT_COLUMN
        elif "question" in columns:
            logger.info(f"Using question/answer formatting for {split_name}")
            format_fn = partial(
                format_qa_with_source,
                output_column=config.text_column,
                source_column=SOURCE_TEXT_COLUMN,
            )
            if not config.response_only_loss:
                format_fn = partial(format_qa, output_column=config.text_column)
            split_data = split_data.map(
                format_fn,
                remove_columns=[col for col in columns if col not in [config.text_column]],
                num_proc=config.preprocessing_num_workers,
                desc=f"Formatting Q&A ({split_name})",
            )
            columns = split_data.column_names
            if config.response_only_loss:
                source_text_column = SOURCE_TEXT_COLUMN

        if config.text_column not in columns:
            raise ValueError(
                f"Could not find text column '{config.text_column}' in split '{split_name}'. "
                "Provide data.text_column or use a dataset with instruction/question fields."
            )

        tokenize_fn = partial(
            tokenize_function,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            text_column=config.text_column,
            source_text_column=source_text_column,
            response_only_loss=config.response_only_loss,
            append_eos_token=config.append_eos_token,
        )
        tokenized_splits[split_name] = split_data.map(
            tokenize_fn,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=columns,
            desc=f"Tokenizing ({split_name})",
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


def get_text_collator(
    tokenizer: PreTrainedTokenizer,
    mlm: bool = False,
) -> DataCollatorForLanguageModeling | DataCollatorForSeq2Seq:
    """Get data collator for language modeling."""
    if not mlm:
        return DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
        pad_to_multiple_of=8,
    )


def prepare_dataset_for_causal_lm(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """Prepare dataset for causal language modeling with labels."""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result
