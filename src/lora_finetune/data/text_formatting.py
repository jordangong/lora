from typing import Any, Dict

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

SOURCE_TEXT_COLUMN = "_source_text"


def format_instruction(
    example: Dict[str, Any],
    template: str = DEFAULT_PROMPT_TEMPLATE,
    output_column: str = "text",
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
    return {output_column: text}


def format_qa(example: Dict[str, Any], output_column: str = "text") -> Dict[str, str]:
    """Format question/answer style examples (e.g., gsm8k, squad)."""
    question = example.get("question", "")
    answer = example.get("answer", "")
    return {output_column: f"Question: {question}\n\nAnswer: {answer}"}


def format_instruction_with_source(
    example: Dict[str, Any],
    template: str = DEFAULT_PROMPT_TEMPLATE,
    output_column: str = "text",
    source_column: str = SOURCE_TEXT_COLUMN,
) -> Dict[str, str]:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", example.get("response", ""))
    if "{output}" in template:
        source_template, template_suffix = template.split("{output}", 1)
        source_text = source_template.format(instruction=instruction, input=input_text)
        text = f"{source_text}{output}{template_suffix}"
    else:
        source_text = template.format(instruction=instruction, input=input_text, output="")
        text = f"{source_text}{output}"
    return {output_column: text, source_column: source_text}


def format_qa_with_source(
    example: Dict[str, Any],
    output_column: str = "text",
    source_column: str = SOURCE_TEXT_COLUMN,
) -> Dict[str, str]:
    question = example.get("question", "")
    answer = example.get("answer", "")
    source_text = f"Question: {question}\n\nAnswer: "
    return {output_column: f"{source_text}{answer}", source_column: source_text}


def format_instruction_as_prompt_completion(
    example: Dict[str, Any],
    template: str = DEFAULT_PROMPT_TEMPLATE,
    prompt_column: str = "prompt",
    completion_column: str = "completion",
) -> Dict[str, str]:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", example.get("response", ""))
    if "{output}" in template:
        prompt_template, completion_suffix = template.split("{output}", 1)
        prompt = prompt_template.format(instruction=instruction, input=input_text)
        completion = f"{output}{completion_suffix}"
    else:
        prompt = template.format(instruction=instruction, input=input_text, output="")
        completion = output
    return {prompt_column: prompt, completion_column: completion}


def format_qa_as_prompt_completion(
    example: Dict[str, Any],
    prompt_column: str = "prompt",
    completion_column: str = "completion",
) -> Dict[str, str]:
    question = example.get("question", "")
    answer = example.get("answer", "")
    return {
        prompt_column: f"Question: {question}\n\nAnswer: ",
        completion_column: answer,
    }


def format_instruction_as_prompt(
    example: Dict[str, Any],
    template: str = DEFAULT_PROMPT_TEMPLATE,
    prompt_column: str = "prompt",
) -> Dict[str, str]:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    if "{output}" in template:
        prompt_template, _ = template.split("{output}", 1)
        prompt = prompt_template.format(instruction=instruction, input=input_text)
    else:
        prompt = template.format(instruction=instruction, input=input_text, output="")
    return {prompt_column: prompt}


def format_qa_as_prompt(
    example: Dict[str, Any],
    prompt_column: str = "prompt",
) -> Dict[str, str]:
    question = example.get("question", "")
    return {
        prompt_column: f"Question: {question}\n\nAnswer: ",
    }


def normalize_conversations_to_messages(
    example: Dict[str, Any],
    source_column: str = "conversations",
    output_column: str = "messages",
) -> Dict[str, Any]:
    conversations = example.get(source_column, [])
    messages = []
    for message in conversations:
        if not isinstance(message, dict):
            raise ValueError(
                f"Expected conversation message to be a dict, got {type(message).__name__}"
            )

        role = message.get("role", message.get("from"))
        content = message.get("content", message.get("value"))
        if role is None or content is None:
            raise ValueError(
                "Conversation messages must include either role/content or from/value fields"
            )

        normalized_message = dict(message)
        normalized_message["role"] = role
        normalized_message["content"] = content
        normalized_message.pop("from", None)
        normalized_message.pop("value", None)
        messages.append(normalized_message)

    return {output_column: messages}
