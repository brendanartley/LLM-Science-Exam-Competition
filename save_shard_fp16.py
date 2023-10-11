"""
Template for saving models weights as shards in FP16.

Benefits
- load model onto GPUs when weights do not fit on CPU

ex. CPU RAM = 10GB, GPU RAM = 30GB, Model weights = 15GB
"""


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name_or_path = "google/flan-t5-xxl"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_or_path,
      device_map='auto', 
      cache_dir="../data/HF_CACHE",
      torch_dtype=torch.float16, # HF saves weights as this type automatically
      )

tokenizer.save_pretrained('./flan_t5_xxl_sharded/')
model.save_pretrained('./flan_t5_xxl_sharded/', max_shard_size="3GiB")