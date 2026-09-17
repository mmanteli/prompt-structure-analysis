from sentence_transformers import SentenceTransformer
import os
import sys
import json
import numpy as np
import torch
import random
from tqdm import tqdm
from utils.prompts import get_prompts, get_detailed_instruct
from prompt_structure_with_distractors import read_distractors

cos = torch.nn.CosineSimilarity()
# set random behaviour for replication
seed = 42
random.seed(seed)
np.random.seed(seed)

batch_size=8
save_prefix="results"

def report(msg):
    print(msg, flush=True)


def stats(t):
    """Extract summary statistics"""
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy().reshape(-1).tolist()  # apply these if you're handling tensors
    else:
        arr = t
    return {"mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            }

if __name__=="__main__":
    model_name = sys.argv[1]
    model = SentenceTransformer(model_name, trust_remote_code=True)
    split="test"
    template="Instruct-Query"
    data_name = sys.argv[2]
    lang=None
    # if lang is given with column notation
    if ":" in data_name:
        data_name, lang = data_name.split(":")
    queries = []
    distractors = []
    raw_distractors, field = read_distractors(data_name, split_to_select=split, lang=lang)
    # we already know these are in the right order (checked in eval)
    for line in raw_distractors:
        queries.append(line[field])
        distractors.append(line["generated_paraphrase"])
    embeddings_d = model.encode(distractors, normalize_embeddings=True,convert_to_tensor=True,
                                batch_size=batch_size)
    embeddings_q = None   # we will calculate these in a moment! with NO_PROMPT
    results={}
    prompts_to_iterate_over = ["NO_PROMPT", "EMPTY"] + get_prompts(data_name=data_name, lang=lang)
    for prompt_num, p in tqdm(enumerate(prompts_to_iterate_over), total = len(prompts_to_iterate_over)):
        prompts_and_queries = [get_detailed_instruct(p, query) for query in queries]
        print("Sanity:")
        print("-------------------")
        print(prompts_and_queries[0])
        print("-------------------")
        embeddings_pq = model.encode(prompts_and_queries, normalize_embeddings=True, convert_to_tensor=True,
                                    batch_size=batch_size)
        if embeddings_q is None and p == "NO_PROMPT":
            embeddings_q = embeddings_pq   # reduce computation
        # cosine similarity
        sim_q2d = cos(embeddings_d, embeddings_q)
        sim_pq2d = cos(embeddings_d, embeddings_pq)
        sim_q2pq = cos(embeddings_pq, embeddings_q)   # already calculated, but this is a sanity check
        # same for euc
        sim_q2d_euc = torch.linalg.norm(embeddings_d-embeddings_q, dim=1)
        sim_pq2d_euc = torch.linalg.norm(embeddings_d-embeddings_pq, dim=1)
        sim_q2pq_euc = torch.linalg.norm(embeddings_pq-embeddings_q, dim=1)

        results[f"prompt{prompt_num}"] = {
        "prompt_text": p if p != "" else "empty",       # prompt text, with "" redirected to "empty"
        "example_text": prompts_and_queries[0],         # example text as a sanity check
        "sim_q2d": stats(sim_q2d),
        "sim_pq2d" : stats(sim_pq2d),
        "sim_q2pq" : stats(sim_q2pq),
        "sim_q2d_euc": stats(sim_q2d_euc),
        "sim_pq2d_euc" : stats(sim_pq2d_euc),
        "sim_q2pq_euc" : stats(sim_q2pq_euc),
        }
    

    model_safe_name = model_name.replace("/", "__")
    data_safe_name = data_name.replace("/","__")
    if lang is not None:
        data_safe_name += f"_{lang}" # bring this back now
    save_path = f"{save_prefix}/{model_safe_name}/{data_safe_name}/{split}/{template}_template"
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving to {save_path}")
    with open(f'{save_path}/distractor_distances.json', 'w') as f:
        json.dump(results, f, indent=2)

    