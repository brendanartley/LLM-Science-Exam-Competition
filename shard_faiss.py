import faiss
from tqdm import tqdm
import numpy as np
from types import SimpleNamespace
import os
import shutil

config = SimpleNamespace(
    index_fname = "../data/faiss_indeces_v14/e5-base-v2_v14_0_0.faiss",
    out_dir = "../data/faiss_indeces_v14_sharded/",
    num_shards = 16,
)

index = faiss.read_index(config.index_fname)
num_vectors = index.ntotal  # num vectors
vector_dim = index.d  # vectr dimensions

# Get embeddings
vectors = np.zeros((num_vectors, vector_dim), dtype=np.float32)
for i in tqdm(range(num_vectors)):
    vectors[i] = index.reconstruct(i)

# Shard + save
arrs = np.array_split(vectors, config.num_shards, axis=0)
shard_fname = config.index_fname.split("/")[-1].split(".")[0]
shard_folder = os.path.join(config.out_dir, shard_fname)

if os.path.exists(shard_folder):
    shutil.rmtree(shard_folder)
os.mkdir(shard_folder)

for i in range(len(arrs)):
     # create index
    sharded_index = faiss.index_factory(arrs[i].shape[1], "Flat")
    sharded_index.add(arrs[i])

    # save sharded
    outfname = f"shard_{i}.faiss"
    faiss.write_index(sharded_index, os.path.join(shard_folder, outfname))
    print(sharded_index.ntotal, sharded_index.d, outfname)

    