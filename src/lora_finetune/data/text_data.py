"""Text dataset utilities for LLM finetuning."""

import logging
from functools import partial
from typing import Any, Dict

from datasets import DatasetDict, load_dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

from ..config import DataConfig

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}
{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""


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

    # Load holdout eval dataset if specified (takes priority)
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
    # Split train data for evaluation if validation split doesn't exist and eval_split_ratio is set
    elif (
        config.validation_split not in dataset
        and config.eval_split_ratio is not None
        and config.eval_split_ratio > 0
    ):
        logger.info(
            f"Splitting train data with eval_split_ratio={config.eval_split_ratio}"
        )
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


def format_instruction(
    example: Dict[str, Any],
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Dict[str, str]:
    """Format example using instruction template."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", example.get("response", ""))

    text = template.format(
        instruction=instruction,
        input=input_text,
        output=output,
    )
    return {"text": text}


def format_qa(example: Dict[str, Any]) -> Dict[str, str]:
    """Format question/answer style examples (e.g., gsm8k, squad)."""
    question = example.get("question", "")
    answer = example.get("answer", "")
    return {"text": f"Question: {question}\n\nAnswer: {answer}"}


def tokenize_function(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_column: str = "text",
) -> Dict[str, Any]:
    """Tokenize text examples."""
    return tokenizer(
        examples[text_column],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )


def preprocess_text_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
) -> DatasetDict:
    """Preprocess text dataset for training."""
    logger.info(f"Preprocessing dataset with max_seq_length={config.max_seq_length}")
    template = config.prompt_template or DEFAULT_PROMPT_TEMPLATE

    tokenize_fn = partial(
        tokenize_function,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        text_column=config.text_column,
    )

    # Process each split separately (they may have different columns for holdout eval)
    tokenized_splits = {}
    for split_name, split_data in dataset.items():
        columns = split_data.column_names

        # Apply formatting based on column structure (skip if text column exists)
        if config.text_column in columns:
            logger.info(f"Using existing '{config.text_column}' column for {split_name}")
        elif "instruction" in columns:
            logger.info(f"Using instruction template formatting for {split_name}")
            split_data = split_data.map(
                partial(format_instruction, template=template),
                remove_columns=[col for col in columns if col not in ["text"]],
                num_proc=config.preprocessing_num_workers,
                desc=f"Formatting instructions ({split_name})",
            )
            columns = split_data.column_names
        elif "question" in columns:
            logger.info(f"Using question/answer formatting for {split_name}")
            split_data = split_data.map(
                format_qa,
                remove_columns=[col for col in columns if col not in ["text"]],
                num_proc=config.preprocessing_num_workers,
                desc=f"Formatting Q&A ({split_name})",
            )
            columns = split_data.column_names

        # Tokenize
        tokenized_splits[split_name] = split_data.map(
            tokenize_fn,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=columns,
            desc=f"Tokenizing ({split_name})",
        )

    tokenized_dataset = DatasetDict(tokenized_splits)

    if config.max_train_samples:
        tokenized_dataset[config.train_split] = tokenized_dataset[config.train_split].select(
            range(min(config.max_train_samples, len(tokenized_dataset[config.train_split])))
        )

    if config.validation_split in tokenized_dataset and config.max_eval_samples:
        tokenized_dataset[config.validation_split] = tokenized_dataset[
            config.validation_split
        ].select(
            range(min(config.max_eval_samples, len(tokenized_dataset[config.validation_split])))
        )

    return tokenized_dataset


def get_text_collator(
    tokenizer: PreTrainedTokenizer,
    mlm: bool = False,
) -> DataCollatorForLanguageModeling:
    """Get data collator for language modeling."""
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
