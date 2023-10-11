import os
import time
import argparse
import gc
from datetime import timedelta
from tqdm.auto import tqdm
import faiss
from sentence_transformers import SentenceTransformer

import torch
import ctypes
libc = ctypes.CDLL("libc.so.6")

import torch
import numpy as np
import pandas as pd
import json
from types import SimpleNamespace

# Load environment variables (stored in config.json file)
with open('./config.json') as f:
    data = json.load(f)
DATA_DIR = data["DATA_DIR"]

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fast_dev_run', action='store_true', default=config.fast_dev_run, help='Testing run.')
    parser.add_argument('--sharded', type=bool, help='If the faiss index is sharded.')
    parser.add_argument('--no_tqdm', type=bool, help='Mute TQDM.')
    parser.add_argument("--model", type=str, default=config.model, help="Model to encode wiki data with.")
    parser.add_argument("--faiss_index_fname", type=str, default=config.faiss_index_fname, help="Faiss index file.")
    parser.add_argument("--train_and_test", type=bool, default=config.train_and_test, help="Wether to create train.csv and all_data.csv.")
    parser.add_argument("--retriever_k", type=int, default=config.retriever_k, help="Number of articles to retrieve.")
    parser.add_argument("--context_top_k", type=int, default=config.context_top_k, help="Number of sections to keep.")
    args = parser.parse_args()
    
    # Update config w/ parameters passed through CLI
    for key, value in vars(args).items():
        setattr(config, key, value)
    return config

config = SimpleNamespace(
    data_dir = DATA_DIR,
    wiki_data_dir = "../data/wiki_data_clean_v14",
    faiss_index_fname = "../data/faiss_indeces_v14/all-MiniLM-L6-v2_v14_0_0.faiss",
    out_dir = "../data/q_data_v14",
    model = "intfloat/e5-base-v2",
    cache_dir = os.path.join(DATA_DIR, "HF_CACHE"),
    batch_size = 16,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    retriever_k = 20,
    context_top_k = 35,
    sharded = False,
    fast_dev_run = False,
    kaggle = False,
    test_run = False,
    infer = True,
    no_tqdm = True,
    train_and_test = False,
)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def process_documents(documents, document_ids):
    sections = []
    doc_ids = []
    for txt, did in zip(documents, document_ids):
        for s in txt.split("\n"):
            sections.append(s)
            doc_ids.append(did)
    df = pd.DataFrame({"document_id": doc_ids, "text": sections})
    return df

def sectionize_documents(documents, document_ids, disable_progress_bar):
    processed_documents = []
    for document_id, document in tqdm(zip(document_ids, documents), total=len(documents), disable=disable_progress_bar):
        row = {}
        text, start, end = (document, 0, len(document))
        row['document_id'] = document_id
        row['text'] = text
        row['offset'] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(['document_id', 'offset']).reset_index(drop=True)
    else:
        return _df

def load_model(config, model_name=config.model):
    """
    Load retrieval/reranker model.

    Note: the model must match the one used to create faiss index.
    """
    print("Model: {}".format(model_name))
    model = SentenceTransformer(model_name, device=config.device, cache_folder=config.cache_dir)
    if config.device == "cuda":
        model.half()
    return model

def load_df(config, csv_fname, kaggle_path="./"):
    """
    Load data as pandas dataframe.
    """
    print("------- {} -------".format(csv_fname))
    if config.kaggle == False: 
        prompt_df = pd.read_csv("{}/q_data/{}".format(config.data_dir, csv_fname))
    else:
        prompt_df = pd.read_csv(kaggle_path)
        prompt_df["answer"] = "A"

    if "id" in prompt_df.columns: 
        prompt_df = prompt_df.drop(columns=["id"])
    if config.fast_dev_run: 
        prompt_df = prompt_df[prompt_df.prompt == "What is the Einstein@Home project?"].head(1)
        prompt_df = prompt_df.head(5)
    return prompt_df

def get_prompt_embeddings(config, model, prompt_df):
    """
    Get the embeddings of each prompt.
    """
    print(len(prompt_df))
    
    # Encode w/ Q - 5 Answers all in one
    prompt_df['prompt_copy'] = prompt_df.apply(lambda x: " ".join([str(x["prompt"]), str(x['C']), str(x['A']), str(x['B']), str(x['D']), str(x['E'])]), axis=1)

    # Specific model prefixes
    if config.model in ["intfloat/e5-small-v2", "intfloat/e5-base-v2", "intfloat/e5-large-v2"]:
        prompt_df["prompt_copy"] = "query: " + prompt_df["prompt_copy"]
    elif config.model in ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"]:
        prompt_df["prompt_copy"] = "Represent this sentence for searching relevant passages: " + prompt_df["prompt_copy"]

    # GPU memory leak with model.encode(), but not a big deal here as array is quite small
    print("encoding prompts.. ({:_})".format(len(prompt_df)))
    embeddings = model.encode(
        prompt_df.prompt_copy.values, 
        batch_size=config.batch_size, 
        device=config.device, 
        show_progress_bar=not config.no_tqdm, 
        convert_to_tensor=True, 
        normalize_embeddings=True,
        )
    embeddings = embeddings.detach().cpu().numpy()

    prompt_df = prompt_df.drop(columns=["prompt_copy"])
    _ = gc.collect()
    return embeddings, prompt_df

def load_and_search_faiss(config, prompt_embeddings):
    """
    Function to load and search FAISS index. Works
    with sharded + unsharded index.
    """
    # Load
    if config.sharded == True or (config.kaggle == True and not config.faiss_index_fname.endswith(".faiss")):
        all_scores, all_indeces = [], []
        index_offset = 0 # add offset to map to non-sharded idx
        all_files = os.listdir(config.faiss_index_fname)
        for i, ifile in enumerate(sorted(all_files, key=lambda x: int(x.split(".")[0].split("_")[-1]))):
            print(f"{i+1}/{len(all_files)}")
            print("loading index..")
            faiss_index = faiss.read_index(os.path.join(config.faiss_index_fname, ifile))
            if config.device == "cuda":
                faiss_index = faiss.index_cpu_to_all_gpus(faiss_index) # send to GPU
            print("loaded index.. FAISS-len: {}".format(faiss_index.ntotal))

            # Searching
            print("searching index..")
            start = time.time()
            search_score, search_index = faiss_index.search(prompt_embeddings, config.retriever_k)
            print("search time: {}".format(str(timedelta(seconds=(time.time() - start))).split(".")[0]))
            all_scores.append(search_score)
            all_indeces.append(search_index + index_offset)
            index_offset += faiss_index.ntotal

            # Clean RAM
            del faiss_index, search_score, search_index
            gc.collect()

        # Grouping and selecting top K
        search_score = np.concatenate(all_scores, axis=1)
        search_index = np.concatenate(all_indeces, axis=1)
        del all_scores, all_indeces
        top_idxs = np.argsort(search_score, axis=1)[:, :config.retriever_k]

        search_score = search_score[np.arange(search_score.shape[0])[:, np.newaxis], top_idxs]
        search_index = search_index[np.arange(search_index.shape[0])[:, np.newaxis], top_idxs]
    else:
        print("1/1")
        print("loading index..")
        faiss_index = faiss.read_index(config.faiss_index_fname)
        if config.device == "cuda":
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index) # send to GPU
        print("loaded index.. FAISS-len: {}".format(faiss_index.ntotal))

        # Searching
        print("searching index..")
        start = time.time()
        search_score, search_index = faiss_index.search(prompt_embeddings, config.retriever_k)
        print("search time: {}".format(str(timedelta(seconds=(time.time() - start))).split(".")[0]))
    return search_score, search_index

def select_wiki_articles(search_score, search_index):
    """
    Selecting the top K wiki articles for each prompt.
    """
    print("selecting wiki article IDs..")
    df = pd.read_parquet("{}/metadata.parquet".format(config.wiki_data_dir), columns=['id', 'file'])
    df["id"] = df["id"].astype(int)

    wiki_article_df = []
    for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score), disable=config.no_tqdm):
        _df = df.iloc[idx, :].copy()
        _df['prompt_id'] = i
        wiki_article_df.append(_df)
    wiki_article_df = pd.concat(wiki_article_df).reset_index(drop=True)
    wiki_article_df = wiki_article_df[['id', 'prompt_id', 'file']].sort_values(['file', 'id']).reset_index(drop=True)

    gc.collect()
    return wiki_article_df

def load_full_articles(wiki_article_df):
    """
    Loading full text data for each selected article.
    """
    print("loading wiki articles..")
    wiki_text_df = []
    for file in tqdm(wiki_article_df.file.unique(), total=len(wiki_article_df.file.unique()), disable=config.no_tqdm):
        _id = wiki_article_df[wiki_article_df['file']==file].id.values
        _df = pd.read_parquet("{}/{}.parquet".format(config.wiki_data_dir, file))
        _df_temp = _df[_df['id'].isin(_id)].copy()
        del _df
        _ = gc.collect()
        wiki_text_df.append(_df_temp)
    wiki_text_df = pd.concat(wiki_text_df).drop_duplicates().reset_index(drop=True)
    wiki_text_df["id"] = wiki_text_df["id"].astype(int)
    return wiki_text_df

def get_section_embeddings(config, wiki_text_data, model):
    """
    Encoding all sentences from relevant articles.
    """
    ## Get embeddings of the split wiki text data
    # NOTE: seems to be a memory leak when using model.encode() on a large array (+2mib/sec)
    # - Source: https://github.com/UKPLab/sentence-transformers/issues/1795
    print("encoding wiki sections..")
    start = time.time()
    from torch.utils.data import DataLoader

    # Specific model prefixes
    if config.model in ["intfloat/e5-small-v2", "intfloat/e5-base-v2", "intfloat/e5-large-v2"]:
        wiki_text_data["text"] = "passage: " + wiki_text_data["text"]

    df_dataset = SimpleDataset(wiki_text_data["text"])    
    df_dataloader = DataLoader(df_dataset, batch_size=config.batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for batch in tqdm(df_dataloader, disable=config.no_tqdm):
            batch_out = model.encode(
                batch,
                batch_size=config.batch_size, 
                device=config.device,
                show_progress_bar=False, # already have progress bar on dataloader 
                convert_to_tensor=True,
                normalize_embeddings=True,
                )
            outputs.append(batch_out.cpu())
    wiki_data_embeddings = torch.cat(outputs).cpu().numpy().astype(np.float32)
    print("splot FAISS-len: {:_}, Encode Time: {}".format(wiki_data_embeddings.shape[0], str(timedelta(seconds=(time.time() - start))).split(".")[0]))
    gc.collect()

    # Remove added prefix
    if config.model in ["intfloat/e5-small-v2", "intfloat/e5-base-v2", "intfloat/e5-large-v2"]:
        wiki_text_data["text"] = wiki_text_data["text"].str.removeprefix("passage: ")

    return wiki_data_embeddings, wiki_text_data

def search_sentences(wiki_article_df, wiki_text_data, wiki_data_embeddings, prompt_df, prompt_embeddings):
    """
    Score sentences from each selected article and
    add context to DF.
    """
    ## Search + add to pandas column
    print("Final search of relevant contexts.")
    contexts = []
    for i in tqdm(range(len(prompt_df)), disable=config.no_tqdm):
        # Get doc_ids for prompt
        prompt_document_ids = wiki_article_df.loc[wiki_article_df['prompt_id']==i, "id"].values
        # Get indices of sentences by doc_ids
        prompt_indices = wiki_text_data[wiki_text_data['document_id'].isin(prompt_document_ids)].index.values
        if prompt_indices.shape[0] > 0:
            prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
            prompt_index.add(wiki_data_embeddings[prompt_indices])

            ## Get the top matches
            context = ""
            s_score, s_index = prompt_index.search(prompt_embeddings, config.context_top_k)
            for ss, si in zip(s_score[i], s_index[i]):
                c = wiki_text_data.loc[prompt_indices]['text'].iloc[si].strip()
                context += c + "<SS>"

        contexts.append(context.strip())
    prompt_df['context'] = contexts

    for col in prompt_df.columns:
        print("{}: {}".format(col, prompt_df.iloc[0][col]))
    return prompt_df

def main(config):

    # Takes ~12-48 hrs to generate all_data.csv
    if config.train_and_test == True:
        create_files = ["train.csv", "all_data.csv"]
    else:
        create_files = ["train.csv"]

    for csv in create_files:
        ## Get output file name
        if config.sharded == True:
            fname = config.faiss_index_fname.split("/")[-1] + "_sharded"
        else:
            fname = config.faiss_index_fname.split("/")[-1].split(".")[0]

        # Formatting output
        csv_fname = "{}/{}__{}_{}_{}_{}_{}_{}.csv".format(
            config.out_dir,
            fname,
            csv.split(".")[0],
            config.model.split("/")[1],
            config.retriever_k,
            config.wiki_data_dir.split("_")[-1],
        )
        print("CSV_NAME: {}".format(csv_fname))
        if os.path.exists(csv_fname) and config.fast_dev_run == False:
            print("PATH ALREADY EXISTS: {}".format(csv_fname))
            continue

        # Load data
        prompt_df = load_df(config, csv_fname=csv)

        # Load model (hack so we use the same model for faiss_index lookup)
        if "all-MiniLM-L6-v2" in config.faiss_index_fname:
            faiss_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        elif "e5-small-v2" in config.faiss_index_fname:
            faiss_model_name = "intfloat/e5-small-v2"
        elif "e5-base-v2" in config.faiss_index_fname:
            faiss_model_name = "intfloat/e5-base-v2"
        elif "e5-large-v2" in config.faiss_index_fname:
            faiss_model_name = "intfloat/e5-large-v2"
        elif "bge-small" in config.faiss_index_fname:
            faiss_model_name = "BAAI/bge-small-en-v1.5"
        elif "bge-base" in config.faiss_index_fname:
            faiss_model_name = "BAAI/bge-base-en-v1.5"
        elif "bge-large" in config.faiss_index_fname:
            faiss_model_name = "BAAI/bge-large-en-v1.5"
        else:
            raise ValueError(f"Nice try. Faiss index model name not recognized: {config.faiss_index_fname}")
        model = load_model(config, model_name=faiss_model_name)

        # Search for related articles
        prompt_embeddings, _ = get_prompt_embeddings(config, model, prompt_df)
        search_score, search_index = load_and_search_faiss(config, prompt_embeddings)

        # Select + Process wikipedia data
        wiki_article_df = select_wiki_articles(search_score, search_index)
        if config.fast_dev_run: print("wiki_article_df", wiki_article_df)
        wiki_text_df = load_full_articles(wiki_article_df)
        if config.fast_dev_run: print("wiki_text_df", wiki_text_df)
        wiki_text_data = process_documents(wiki_text_df.text.values, wiki_text_df.id.values)
        if config.fast_dev_run: print("wiki_text_data", wiki_text_data)

        ## Search + Add context
        model = load_model(config, model_name=config.model)
        wiki_data_embeddings, wiki_text_data = get_section_embeddings(config, wiki_text_data, model)
        prompt_embeddings, _ = get_prompt_embeddings(config, model, prompt_df)
        prompt_df = search_sentences(wiki_article_df, wiki_text_data, wiki_data_embeddings, prompt_df, prompt_embeddings)

        ## Actually Save
        if config.fast_dev_run == False:
            prompt_df[["prompt", "context", "A", "B", "C", "D", "E", "answer"]].to_csv(csv_fname, index=False)
            print("Saved.")

    # # Optional: Run inference w/ pre-trained model
    # if config.infer == True and config.fast_dev_run == False:
    #     from mult_choice_scripts.run import valid
    #     from main import config as cfg

    #     cfg.q_data_version = "q_data_{}".format(config.out_dir.split("_")[-1])
    #     cfg.file_name = csv_fname.split("/")[-1].replace("all_data", "train") # in case we generate both CSVs
    #     # cfg.model_name = "../data/models/test_run_2_26614"
    #     # cfg.max_tokens=512
    #     cfg.model_name = "../data/models/dbv3_26048"
    #     cfg.max_tokens=1024
    #     cfg.with_answers=True
    #     print(cfg.file_name)
    #     valid(cfg)
    print("Done!")
    return

if __name__ == "__main__":
    config = parse_args()
    main(config)