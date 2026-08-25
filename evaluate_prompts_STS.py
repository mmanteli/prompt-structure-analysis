from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
import datasets
import jsonargparse
import json
import os
import pickle
import random
import torch
from scipy.stats import spearmanr
from utils.prompts import get_prompts, get_detailed_instruct
from utils.dataset_handling import download_dataset


# set random behaviour for replication
seed = 42
random.seed(seed)
np.random.seed(seed)


cos = torch.nn.CosineSimilarity()

parser = jsonargparse.ArgumentParser(prog="Quickly evaluate a prompt for retrieval/bitext-mining")
#parser.add_argument('--config', action=ActionConfigFile)
parser.add_argument('--model_name', '--model', type=str, default=None, required=True,
                    help="HF-alias or path to downloaded model.")
parser.add_argument('--data_name', '--dataset', type=str, default="mteb/ARCChallenge",
                    help="HF-alias or path to downloaded dataset.")
parser.add_argument('--split', type=str, default=None,
                    help="Which split to select from dataset.")
parser.add_argument('--template', type=str, default="Instruct-Query", choices=["Instruct-Query", "simple"],
                    help="Which prompting template to use")
parser.add_argument('--use_lang_specific_prompts', '--use_language_specific_prompts', action='store_true',
                    help="Use prompts that specifically mention the target language")
parser.add_argument('--k', '--MRR@k', type=int, default=10,
                    help="Which k to use for MRR and Recall")
parser.add_argument('--batch_size', type=int, default=16,
                    help="Batch size for embedding")
parser.add_argument('--num_examples', type=int|bool, default=5000,
                    help="For largest datasets, number of examples to downsample to, set to False for no downsampling")
parser.add_argument('--embedding_prefix', type=str|bool, default=False,
                    help="prefix to save embedings to, works similar to --save_prefix")
parser.add_argument('--save_prefix', type=str, default="results_prompts",
                    help="Saving path; model_name, data_name, prompt_type and k added in script")


def report(msg):
    # for quick flushing
    print(f'{msg}', flush=True)

def append_pkl(data, path):
    with open(path, "ab") as f:
        pickle.dump(data, f)

def yield_from_pkl(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def write_embeddings(file, key, data, embeddings):
    if isinstance(data, datasets.Dataset):
        if embeddings is None:
            d = {data.to_dict()}
        else:
            d = {**data.to_dict(), **{"embeddings":embeddings}}
    else:
        if embeddings is None:
            d = data
        else:
            d = {**data, **{"embeddings":embeddings}}
    append_pkl({"key":key, "data": d}, file)


def apply_template(prompt, query, template="Instruct-Query"):
    return get_detailed_instruct(prompt, query, template=template)



def find_relevant_doc_id(query_id, qrels):
    if isinstance(qrels, dict):
        return ([qrels[query_id]], [1]) if query_id in qrels else ([], [])
    indices_of_query_ids = np.where(np.array(qrels["query_id"]) == query_id)[0]
    associated_corpus_values = np.array(qrels["corpus_id"])[indices_of_query_ids]
    associated_corpus_scores = np.array(qrels["score"])[indices_of_query_ids]
    # sort these to have the best match at the top
    indices_that_sort = np.argsort(associated_corpus_scores)[::-1]
    return (associated_corpus_values[indices_that_sort].tolist(), associated_corpus_scores[indices_that_sort].tolist())

def stats(t):
    """Extract summary statistics"""
    arr = t # .detach().cpu().numpy().reshape(-1)  # apply these if you're handling tensors
    return {"mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            #"full": str(arr)
            }

def calculate_scores(k, scores, corpus_embeddings, query_embeddings):
    """
    Calculate spearman correlation with cosinesimilarities and pair scores.
    ASSUMES corpus and queries to be in order!
    """
    sims = cos(torch.tensor(corpus_embeddings), torch.tensor(query_embeddings)).detach().cpu().numpy()
    score = spearmanr(scores, sims)
    return {f"cosine-spearman": score}


def embed_and_calculate_STS_scores(options, dataset_specific_prompts, corpus, queries, scores):
    """Embed corpus, queries (+prompts) and calculate evaluation scores."""
    # check that we are in the right format
    if isinstance(corpus, datasets.Dataset):
        assert "text" in corpus.keys(), f"Cannot find field 'text' in corpus columns, {corpus.keys()=}"
        corpus = corpus["text"]
    if isinstance(queries, datasets.Dataset):
        assert "text" in corpus.keys(), f"Cannot find field 'text' in corpus columns, {corpus.keys()=}"
        queries = queries["text"]
    assert isinstance(scores, list) or isinstance(scores, np.array)
    
    # download model
    model = SentenceTransformer(options.model_name)
    # embed the corpus==targets/answers --> prompt has no effect on these
    corpus_embeddings = model.encode(corpus, normalize_embeddings=True, batch_size=options.batch_size)
    if options.embeddings:
        os.makedirs(os.path.dirname(options.embeddings), exist_ok=True)
        write_embeddings(file=options.embeddings, key="scores", data={"scores":scores}, embeddings=[None]*len(scores))
        write_embeddings(file=options.embeddings, key="corpus", data={"text":corpus}, embeddings=corpus_embeddings)

    # loop over prompts
    results = {}
    for i,p in enumerate(dataset_specific_prompts):
        # encode template(prompt, query)
        prompted_queries = [apply_template(p, q, template=options.template) for q in queries]
        query_embeddings = model.encode(prompted_queries, normalize_embeddings=True, batch_size=options.batch_size)
        if options.embeddings:
            # construct a new dict here
            write_embeddings(file=options.embeddings, 
                             key=p, 
                             data={"text": prompted_queries}, 
                             embeddings=query_embeddings)
        report(f"----\nNow in prompt {i}, example: \n{prompted_queries[0]}")
        results[f"prompt{i}"] = {**{"prompt_text": p}, **calculate_scores(options.k, scores, corpus_embeddings, query_embeddings)}

    os.makedirs(os.path.dirname(options.save_path), exist_ok=True)
    with open(options.save_path, 'w') as f:
        json.dump(results,f, indent=2)


def read_embeddings_and_calculate_STS_scores(options, dataset_specific_prompts, corpus, queries, scores):
    corpus_embeddings = None
    results = {}
    i = 0 # index for prompts
    for data in yield_from_pkl(options.embeddings):
        # First 2 saved values are qrels and corpus embeddings
        # check that they match, and set corpus_embeddings
        if data["key"] == "scores":
            # qrels is saved in "text_ids"
            assert data["data"]["scores"] == scores, "Mismatch between scores and reading precalculated embeddings"
            continue
        if data["key"] == "corpus":
            assert data["data"]["text"] == corpus, "Mismatch between loaded corpus and precalculated embeddings"
            # set corpus embeddings
            corpus_embeddings = data["data"]["embeddings"]
            continue
        # after reading, we can calculate:
        prompt = data["key"]  # saved here
        report(f"In prompt {prompt}")
        query_embeddings = data["data"]["embeddings"]
        # check that prompt order is the same --> we can save with the same prompt order
        assert get_detailed_instruct(prompt, queries[0], template=options.template) == data["data"]["text"][0], \
            f'{get_detailed_instruct(prompt, queries[0], template=options.template)} != {data["data"]["text"][0]}'
        results[f"prompt{i}"] = {**{"prompt_text": prompt}, **calculate_scores(options.k, scores, corpus_embeddings, query_embeddings)}
        i += 1

    os.makedirs(os.path.dirname(options.save_path), exist_ok=True)
    with open(options.save_path, 'w') as f:
        json.dump(results,f, indent=2)



if __name__=="__main__":
    #sanity_check_sorting()
    options = parser.parse_args()
    lang=None
    # if lang is given with column notation
    if ":" in options.data_name:
        options.data_name, lang = options.data_name.split(":")
    corpus, queries, scores = download_dataset(options.data_name, lang=lang, split_to_select=options.split, downsample=options.num_examples)
    if options.use_lang_specific_prompts:
        prompts = get_prompts(options.data_name, lang=lang)
    else:
        prompts =  get_prompts(options.data_name)
    report("Sanity check: What was downloaded?")
    report(prompts[0])
    report(queries[0])
    report(corpus[0])
    report(scores[0])
    # add a few prompts based on the template (function as baselines)
    if options.template != "simple":
        # These map to NO_PROMPT=vanilla query and EMPTY: misfilled template
        prompts = ["NO_PROMPT", "EMPTY"] + prompts
    else:
        # For the simple template, only vanilla query
        prompts = ["NO_PROMPT"] + prompts

    # create full saving paths
    model_name_ = options.model_name.replace("/","__")
    data_safe_name = options.data_name.replace("/","__")
    if lang is not None:
        data_safe_name += f":{lang}"
    # only set this if the prefix was given
    options.embeddings = "" if not options.embedding_prefix else f'{options.embedding_prefix}/{model_name_}/{data_safe_name}{"_lang_specific" if options.use_lang_specific_prompts else ""}/{options.split}/{options.template}_embeddings.pkl'
    options.save_path =  f'{options.save_prefix}/{model_name_}/{data_safe_name}{"_lang_specific" if options.use_lang_specific_prompts else ""}/{options.split}/{options.template}_template/results@{options.k}.json'
    
    if options.embeddings != "" and os.path.exists(options.embeddings):
        # we have precalculated embeddings
        report(f"Using precalculated embeddings at {options.embeddings} for score calculation")
        read_embeddings_and_calculate_STS_scores(options, prompts, corpus, queries, scores)
    else:
        # calculate everything from scratch
        report("Calculating embeddings and score")
        embed_and_calculate_STS_scores(options, prompts, corpus, queries, scores)
