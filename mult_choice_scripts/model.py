from transformers import AutoTokenizer, AutoModelForMultipleChoice

def get_model(config):
    model = AutoModelForMultipleChoice.from_pretrained(
        config.model_name, 
        cache_dir=config.cache_dir,
        ignore_mismatched_sizes=True,
        )
    return model

def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir, use_fast=False)
    return tokenizer