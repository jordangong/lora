from typing import List

TEXT_TARGET_MODULES = {
    "xlm-roberta": ["query", "key", "value", "dense"],
    "roberta": ["query", "key", "value", "dense"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    "deberta-v3": ["query_proj", "key_proj", "value_proj", "dense"],
    "deberta": ["query_proj", "key_proj", "value_proj", "dense"],
    "distilbert": ["q_lin", "k_lin", "v_lin", "out_lin"],
    "bert": ["query", "key", "value", "dense"],
    "albert": ["query", "key", "value", "dense"],
    "electra": ["query", "key", "value", "dense"],
    "default": ["query", "key", "value", "dense"],
}


def get_text_target_modules(model_name_or_path: str) -> List[str]:
    model_name_lower = model_name_or_path.lower()

    for key in TEXT_TARGET_MODULES:
        if key in model_name_lower:
            return TEXT_TARGET_MODULES[key]

    return TEXT_TARGET_MODULES["default"]
