"""
Code to get get a rough estimate of how many tokens are used to
encode text.

Ex. deberta-v3
445 characters == 138 tokens
"""

import pandas as pd
from transformers import AutoTokenizer
import os
from types import SimpleNamespace

config = SimpleNamespace(
    model="sileod/deberta-v3-base-tasksource-nli",
    cache_dir="../data/HF_CACHE/",
    data_dir="../data/q_data_v14/",
)

def get_token_ids_length(prompt):
    """
    Given a text, return token_ids.
    """
    tokens = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
    return len(tokens["input_ids"][0])

if __name__ == "__main__":
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.cache_dir)

    # Text
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    print(len(text))

    # Get tokens
    tokens = get_token_ids_length(text)
    print(tokens)
