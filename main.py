import argparse
import os
import json
from types import SimpleNamespace
import torch
import numpy as np
from transformers import set_seed

# Load environment variables (stored in config.json file)
with open('./config.json') as f:
    data = json.load(f)
DATA_DIR = data["DATA_DIR"]

config = SimpleNamespace(
    project = "LLMSE",
    model_name = "sileod/deberta-v3-base-tasksource-nli",
    run_name = "debertav3large",
    data_dir = DATA_DIR,
    model_save_dir = os.path.join(DATA_DIR, "models/"),
    checkpoint_dir = os.path.join(DATA_DIR, "checkpoints/"),
    cache_dir = os.path.join(DATA_DIR, "HF_CACHE/"),
    q_data_version = "q_data_v14",
    file_name = "e5-base-v2_v14_0_0__train_e5-base-v2_20_v14_a_a.csv", # MUST BE THE all_data path
    text_column = "prompt",
    label_column = "answer",
    prompt_prefix = "",
    device = "cuda" if torch.cuda.is_available() else "cpu",
    warmup_ratio = 0.05,
    lr = 4e-6, # nothing more than 5e-6 with bf16
    epochs = 2, 
    batch_size = 3,
    seed = np.random.randint(0, 100_000),
    grad_acc_steps = 1,
    logging_steps = 250,
    disable_tqdm = False,
    resume = False,
    fast_dev_run = False,
    bf16 = False,
    fp16 = True,
    max_tokens = 512,
    early_stopping_patience = 7,
    kaggle = False,
    no_early_stop = True,
    with_answers = True,
    num_workers = 2,
    train_all = False,
)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fast_dev_run', action='store_true', help='Testing run.')
    parser.add_argument("--seed", type=int, default=config.seed, help="Sets seed for reproducability (not really used).")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Num training iterations.")
    parser.add_argument("--model_name", type=str, default=config.model_name, help="Name of the Huggingface Model to train.")
    parser.add_argument("--run_name", type=str, default=config.run_name, help="Run name.")
    parser.add_argument("--file_name", type=str, default=config.file_name, help="Name of the training data file.")
    parser.add_argument('--bf16', action='store_true', help='Uses BF16 precision for training.')
    parser.add_argument('--fp16', type=bool, help='Uses FP16 precision for training.')
    parser.add_argument('--no_wandb', action='store_true', help='Do not log to W&B.')
    parser.add_argument('--resume', action='store_true', help='Wether to resume from checkpoint if found.')
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size.")
    parser.add_argument("--logging_steps", type=int, default=config.logging_steps, help="Number of steps between logging.")
    parser.add_argument("--lr", type=float, default=config.lr, help="Learning rate.")
    parser.add_argument("--grad_acc_steps", type=int, default=config.grad_acc_steps, help="Gradient accumulation steps.")
    parser.add_argument("--max_tokens", type=int, default=config.max_tokens, help="Max sequence length.")
    parser.add_argument("--early_stopping_patience", type=int, default=config.early_stopping_patience, help="Early stopping patience.")
    parser.add_argument('--no_early_stop', type=bool, default=config.no_early_stop, help='Dont do early stopping.')
    parser.add_argument('--with_answers', type=bool, default=config.with_answers, help='Wether to add other answers into text.')
    parser.add_argument("--num_workers", type=int, default=config.num_workers, help="num workers (for dataloader).")
    parser.add_argument('--train_all', type=bool, help='Train on ALL DATA (no validation set).')
    args = parser.parse_args()
    
    # Update config w/ parameters passed through CLI
    for key, value in vars(args).items():
        setattr(config, key, value)

    return config

def main(config):

    # Variables that need to be set before imports
    # set_seed(config.seed) # UNCOMMENT TO MAKE RUNS REPRODUCABLE
    os.environ['WANDB_PROJECT'] = config.project

    # Note: import here so set_seed() works
    from mult_choice_scripts.run import train
    module = train(config)
    pass

if __name__ == "__main__":
    config = parse_args()
    main(config)