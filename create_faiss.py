import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import argparse
import json
from types import SimpleNamespace
import faiss
from sentence_transformers import SentenceTransformer

import torch
from torch.utils.data import DataLoader
import ctypes
libc = ctypes.CDLL("libc.so.6")

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from create_data import config as cd_config
from create_data import main as cd_main

# Load environment variables (stored in config.json file)
with open('./config.json') as f:
    data = json.load(f)
DATA_DIR = data["DATA_DIR"]

config = SimpleNamespace(
    data_dir = DATA_DIR,
    wiki_data_dir = "../data/wiki_data_clean_v14",
    faiss_index_dir = "../data/faiss_indeces_v14",
    model = "intfloat/e5-base-v2",
    cache_dir = os.path.join(DATA_DIR, "HF_CACHE"),
    csv_file = "train.csv",
    batch_size = 4096,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    add_tags = False,
    add_first_sentence = False,
)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
    
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fast_dev_run', action='store_true', help='Testing run.')
    parser.add_argument("--model", type=str, default=config.model, help="Model to encode wiki data with.")
    parser.add_argument("--add_tags", type=bool, default=config.add_tags, help="Wether to prepend text with tags.")
    parser.add_argument("--add_first_sentence", type=bool, default=config.add_first_sentence, help="Wether to prepend text with first sentence.")
    parser.add_argument("--wiki_data_dir", type=str, default=config.wiki_data_dir, help="Dir for wiki data (infile)")
    parser.add_argument("--faiss_index_dir", type=str, default=config.faiss_index_dir, help="Dir for the faiss index (outfile).")
    args = parser.parse_args()
    
    # Update config w/ parameters passed through CLI
    for key, value in vars(args).items():
        setattr(config, key, value)
    return config

def create_faiss_index(config):

    # Load model
    print(config.model)
    model = SentenceTransformer(config.model, device='cuda', cache_folder=config.cache_dir)
    model.half()

    # Create embedding for every article/text
    # Note: Uses >100GB RAM for 22 million texts, consider writing to disk every N steps
    outputs = []
    for fpath in sorted([x for x in os.listdir(config.wiki_data_dir) if x not in ["metadata.parquet", "wiki_2023_index.parquet"]], key=lambda x: int(x.split(".")[0])):
        if fpath.endswith(".parquet") == False:
            continue

        print("-"*10 + f" Processing: {fpath} " + "-"*10)
        df = pd.read_parquet("{}/{}".format(config.wiki_data_dir, fpath))
        if config.fast_dev_run == True: 
            df = df.head(500)
            outfile = "NAN"
        else:
            outfile = "{}/{}_{}_{}_{}.faiss".format(
                config.faiss_index_dir, 
                config.model.split("/")[1], 
                config.wiki_data_dir.split("_")[-1], 
                int(config.add_tags),
                int(config.add_first_sentence),
                )
            if os.path.exists(outfile):
                assert ValueError("PATH ALREADY EXISTS: {}".format(outfile))

        # Adding prefix (when required)
        if config.model in ["intfloat/e5-small-v2", "intfloat/e5-base-v2", "intfloat/e5-large-v2"]:
            df["text"] = "passage: " + df["text"]

        # Optional: Tags + Text
        if config.add_tags == True:
            df["text"] = df["headers"] + " " + df["text"]

        # Optional: 1st sentence + Text
        if config.add_first_sentence == True:
            df["text"] = df["first_sentence"] + " " + df["text"]

        # Create Embeddings
        df_dataset = SimpleDataset(df["text"].values)
        df_dataloader = DataLoader(df_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=5)
        with torch.no_grad():
            for batch in df_dataloader:
                batch_out = model.encode(
                    batch,
                    device=config.device,
                    show_progress_bar=False, 
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    )
                outputs.append(batch_out.cpu())

        # Testing Run
        if config.fast_dev_run == True and len(outputs) >= 3:
            break

    # Combine Embeddings
    outputs = torch.cat(outputs).cpu().numpy().astype(np.float32)
    print(outputs.shape)

    # Create FAISS index + save
    wiki_index = faiss.index_factory(outputs.shape[1], "Flat")
    wiki_index.add(outputs)
    
    # Write to disk
    if config.fast_dev_run == False:
        faiss.write_index(wiki_index, outfile)

    print("Index len: {}".format(wiki_index.ntotal), outfile)
    return outfile


def main(config):

    # Create Index
    outfile = create_faiss_index(config)

    # # Optional: eval performance of faiss index
    # cd_config.faiss_index_fname = outfile
    # cd_config.wiki_data_dir = "../data/wiki_data_clean_{}".format(config.wiki_data_dir.split("_")[-1])
    # cd_config.out_dir = "../data/q_data_{}".format(config.wiki_data_dir.split("_")[-1])
    # config.max_tokens=512
    # config.with_answers=True
    # cd_main(cd_config)
    return

if __name__ == "__main__":
    config = parse_args()
    main(config)