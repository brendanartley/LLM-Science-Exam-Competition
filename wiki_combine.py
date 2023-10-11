import re
import os
import pickle
import pandas as pd
from tqdm import tqdm
from functools import partial
import hashlib
import blingfire as bf
import shutil 

class config:
    wiki_dir = "C:/dev/wiki_data/"
    midfile = "./wiki_data_clean_v10/"
    outfile = "./wiki_data_clean_v14/"

def get_text_and_headers(text, min_filter_len=50):
    """
    Cleans wikipedia text, and extracts tag headers.
    """

    sections = list()
    bl = ["See also", "Notes", "External links", "Overview", "Description", "Career", "Sources"]

    all_pairs = []

    # Get first_sentence
    try:
        _, sentence_offsets = bf.text_to_sentences_and_offsets(t[:500])
        first_sentence = text[0:sentence_offsets[0][1]]
    except:
        first_sentence = text[:100]

    # Split on sections
    for s in text.split("\n"):
        t = s.strip()
        if len(t) == 0:
            continue

        if len(t) == 0:
            continue

        # Stop once we get to references
        if re.search("==\s*[rR]eferences\s*==.*", t):
            # print("BREAKING ON REFERENCES..")
            break

        # Checking if a tag row (== History ==)
        if t[0] == "=" and t[-1] == "=":
            if len(t) >= 7 and len(t) < 100:
                # Splitting by subsection
                if t.count("==") <= 6:
                    all_pairs.append(("\n".join(sections), first_sentence))
                    sections = [t.strip(" =")]
                    # Adding period if not there
                    if len(sections[-1]) > 0 and sections[-1][-1] != ".":
                        sections[-1] += "."
                elif t.count("=") > 6 and t.replace("=", "").replace(" ", "") not in bl:
                    sections.append(t.strip(" ="))
                    if len(sections[-1]) > 0 and sections[-1][-1] != ".":
                        sections[-1] += "."
            else:
                continue
        
        # Check length is greater than 50
        if len(t) < min_filter_len:
            continue
        
        # Remove Latex rows
        if t[0] == "{" and t[-1] == "}":
            continue
        sections.append(t)
    
    # List -> str
    sections = "\n".join(sections)
    all_pairs.append((sections, first_sentence))
    return all_pairs

def process_wiki():
    cur_batch = []
    meta_data = []
    save_c = 0
    row_c = 0
    save_every_n = 250_000

    shutil.rmtree(config.midfile)
    os.mkdir(config.midfile)

    # Every tmp_* file
    for sub_dir in sorted(os.listdir(config.wiki_dir)):
        print("Processing.. ", sub_dir)
        tmp_dir = os.path.join(config.wiki_dir, sub_dir)
        # Every pkl file
        for sub_file in sorted(os.listdir(tmp_dir), key=lambda x: int(x.split("_")[0])):
            tmp_file = os.path.join(tmp_dir, sub_file)
            in_arr = pickle.load(open(tmp_file, 'rb'))
            # Every text
            for raw_text in in_arr:
                if raw_text == "":
                    continue
                    
                # Parse text
                all_pairs = get_text_and_headers(raw_text)

                for text, first_sentence in all_pairs:
                    if len(text) < 50 or "may refer to" in text[:100].lower(): # for final combine
                    # if len(text) < 50: # For getting "missing articles"
                        continue
                    cur_batch.append((row_c, text, first_sentence))
                    meta_data.append((row_c, save_c))
                    row_c += 1

                    # Save in batches/shards
                    if len(cur_batch) == save_every_n:
                        
                        # Saving as parquet file
                        save_file = "{}.parquet".format(save_c)
                        df = pd.DataFrame(cur_batch, columns =['id', 'text', 'first_sentence'])
                        df.to_parquet(os.path.join(config.midfile, save_file), index=False)
                        print("Saving.. ", save_file)

                        # Clear batch
                        cur_batch = []
                        save_c += 1

    # Save remaining batch
    save_file = "{}.parquet".format(save_c)
    df = pd.DataFrame(cur_batch, columns =['id', 'text', 'first_sentence'])
    df.to_parquet(os.path.join(config.midfile, save_file), index=False)
    print("Saving.. ", save_file)

    # Save metadata
    save_file = "metadata.parquet"
    df = pd.DataFrame(meta_data, columns=['id', 'file'])
    df.to_parquet(os.path.join(config.midfile, save_file), index=False)
    print("Len: {:_}, Saving.. {}".format(len(df), save_file))
    return

def clean_wiki_v10():
    s = set()
    df = []
    total_duplicated = 0
    final_len = 0
    orig_len = 0
    meta_data = []

    if os.path.exists(config.outfile) == False:
        os.mkdir(config.outfile)

    text_pkls = sorted([x for x in os.listdir(config.midfile) if x not in ["metadata.parquet", "wiki_2023_index.parquet"]], key=lambda x: int(x.split(".")[0]))
    for i, file in tqdm(enumerate(text_pkls), total=len(text_pkls)):
        if file != "metadata.parquet":
            df = pd.read_parquet(config.midfile + file)
            orig_len += len(df)
            c = 0
            del_arr = [0]*len(df)

            for j, val in enumerate(df.text.values):
                # using hash because texts are quite long..
                text_hash = hashlib.sha256(val.encode()).hexdigest()
                if text_hash in s:
                    c+=1
                    del_arr[j] = 1
                else:
                    s.add(text_hash)
            
            # Drop duplicates
            df["del"] = del_arr
            df = df[df["del"] == 0].drop(columns=["del"])

            # Set ID column
            df["id"] = range(len(df)) 
            df["id"] += final_len

            total_duplicated += c
            final_len += len(df)
            print("File: {}, Duplicates: {}".format(file, c))
            df.to_parquet(config.outfile + file, index=False)
            meta_data.extend([i]*len(df))

    df = pd.DataFrame(meta_data, columns=["file"])
    df["id"] = range(len(df))
    df.to_parquet(config.outfile + "metadata.parquet", index=False)

    print("Duplicated: {:_}, Orig: {:_}, Final: {:_}".format(total_duplicated, orig_len, final_len))
    return

if __name__ == "__main__":
    process_wiki()
    clean_wiki_v10()
    pass