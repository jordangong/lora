"""Text dataset utilities for LLM finetuning."""

from functools import partial
from typing import Any, Dict

from datasets import DatasetDict, load_dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

from ..config import DataConfig

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
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            streaming=config.streaming,
        )
    elif config.train_file:
        data_files = {"train": config.train_file}
        if config.validation_file:
            data_files["validation"] = config.validation_file

        extension = config.train_file.split(".")[-1]
        if extension == "jsonl":
            extension = "json"

        dataset = load_dataset(extension, data_files=data_files)
    else:
        raise ValueError("Either dataset_name or train_file must be provided")

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
    template = config.prompt_template or DEFAULT_PROMPT_TEMPLATE

    if "instruction" in dataset[config.train_split].column_names:
        dataset = dataset.map(
            partial(format_instruction, template=template),
            remove_columns=[
                col for col in dataset[config.train_split].column_names if col not in ["text"]
            ],
            num_proc=config.preprocessing_num_workers,
            desc="Formatting instructions",
        )

    tokenize_fn = partial(
        tokenize_function,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        text_column=config.text_column,
    )

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=config.preprocessing_num_workers,
        remove_columns=dataset[config.train_split].column_names,
        desc="Tokenizing",
    )

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
