from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np
import torch
import jsonargparse
import random
import datasets
from utils.dataset_handling import download_dataset
from utils.prompts import get_prompts, get_detailed_instruct
from prompt_structure_with_distractors import read_distractors

cos = torch.nn.CosineSimilarity()
# set random behaviour for replication
seed = 42
random.seed(seed)
np.random.seed(seed)

def report(msg):
    print(msg, flush=True)

parser = jsonargparse.ArgumentParser(prog="Evaluate models with changes in prompting")
#parser.add_argument('--config', action=ActionConfigFile)
parser.add_argument('--model_name', '--model', type=str|list,
                    help="HF-alias or path to downloaded model, can be a list.")
parser.add_argument('--data_name', '--dataset', type=str|list,
                    help="HF-alias or path to downloaded dataset, can be a list.")
parser.add_argument('--split', type=str, default="test",
                    help="Which split to select from dataset.")
parser.add_argument('--template', type=str, default="Instruct-Query", choices=["Instruct-Query", "simple"],
                    help="Which prompting template to use")
parser.add_argument('--use_lang_specific_prompts', action='store_true',
                    help="Use prompts that specifically mention the target language, only for multilingual datasets.")
parser.add_argument('--k', type=int|list, default=1,
                    help="number of closest matches searched")
parser.add_argument('--batch_size', type=int, default=16,
                    help="batch size for embedding")
parser.add_argument('--embedding_prefix', type=str|bool, default=False,
                    help="prefix to save embedings to, works similar to --save_prefix")
parser.add_argument('--save_prefix', type=str, default="results_metrics",
                    help="Saving path; model_name, data_name, prompt_type and k added in script")



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

def find_relevant_doc_id(query_id, qrels):
    if isinstance(qrels, dict):
        return ([qrels[query_id]], [1]) if query_id in qrels else ([], [])
    indices_of_query_ids = np.where(np.array(qrels["query_id"]) == query_id)[0]
    associated_corpus_values = np.array(qrels["corpus_id"])[indices_of_query_ids]
    associated_corpus_scores = np.array(qrels["score"])[indices_of_query_ids]
    # sort these to have the best match at the top
    indices_that_sort = np.argsort(associated_corpus_scores)[::-1]
    return (associated_corpus_values[indices_that_sort].tolist(), associated_corpus_scores[indices_that_sort].tolist())


def calculate_scores_with_qrels_one_k(k, corpus, queries, qrels, corpus_embeddings, query_embeddings, prompt_text=None):
    """
    Calculate NDCG@k and recall@k for given queries and corpus
    corpus=dataset of targets (columns _id and text)
    queries=dataset of queries (columns _id and text)
    qrels=either a dataset of (query_id, corpus_id, score) or binary dict(query_id:corpus_id)
    corpus_embeddings = matrix of corpus embeddings (in the same order as corpus)
    query_embeddings = matrix of query embeddings (again, same order and possibly, a prompt has been added before calculation)
    """
    # calculate similarity matrix
    sims = query_embeddings @ corpus_embeddings.T
    if isinstance(sims, torch.Tensor):
        sims = sims.detach().cpu().numpy()
    # argsort sims to get best matches
    # here the additional ":" is needed together with axis=1
    sims = np.argsort(sims, axis=1)[:, ::-1]
    # initialize data collection
    recall_at_k = []
    ndcg_at_k = []
    for (i, query), sim_line in zip(enumerate(queries), sims):
        # first, find the relevant ids (this works for both types of qrels)
        relevant_ids, associated_scores = find_relevant_doc_id(query["_id"], qrels)
        # from these, calculate the ideal_cumulative_gain (used to normalize discounted cumulative gain)
        ideal_cumulative_gain = np.sum([(2**s-1)/np.log2(rank+1+1.) for rank, s in enumerate(np.sort(associated_scores)[::-1][:k])])   # UP TO k
        # if everything was perfect: highest score at rank1, second at rank2
        # also +1+1 since the rank is zero indexed here -> one +1 to fix rank and other is in the formula
        # Next, find best k matches
        most_similar_docs = sim_line[:k]
        found_ids = [corpus["_id"][j] for j in most_similar_docs]
        # we can already calculate recall results with no rank information
        rec_ = sum(1 for fid in found_ids if fid in relevant_ids) / len(relevant_ids)
        # initialize the rank dependent metrics
        discounted_cumulative_gain = 0
        for rank_, found_id in enumerate(found_ids):
            rank = rank_+ 1 # fix zero indexing
            if found_id in relevant_ids:
                found_score = associated_scores[relevant_ids.index(found_id)]
                discounted_cumulative_gain += (2**found_score-1)/np.log2(rank+1)
        # calculate ndcg@k for this query
        current_ndcg_at_k = 0 if ideal_cumulative_gain == 0 else discounted_cumulative_gain/ideal_cumulative_gain # 0 if nothing was to be discovered
        # collect results
        ndcg_at_k.append(current_ndcg_at_k)
        recall_at_k.append(rec_)

    return {"prompt_text": prompt_text,
            f"recall@{k}": stats(recall_at_k),
            f"ndcg@{k}": stats(ndcg_at_k),
            }

def calculate_scores_with_qrels_multiple_ks(ks, corpus, queries, qrels, corpus_embeddings, query_embeddings, prompt_text=None):
    """
    Calculate NDCG@k and recall@k for given queries and corpus
    corpus=dataset of targets (columns _id and text)
    queries=dataset of queries (columns _id and text)
    qrels=either a dataset of (query_id, corpus_id, score) or binary dict(query_id:corpus_id)
    corpus_embeddings = matrix of corpus embeddings (in the same order as corpus)
    query_embeddings = matrix of query embeddings (again, same order and possibly, a prompt has been added before calculation)
    """
    if isinstance(ks, int):
        ks = [ks]
    max_k = max(ks)

    sims = query_embeddings @ corpus_embeddings.T
    if isinstance(sims, torch.Tensor):
        sims = sims.detach().cpu().numpy()
    sims = np.argsort(sims, axis=1)[:, ::-1][:, :max_k]  # take max k instances -> take k in the loop
    # result collections
    recall_at_k = {k: [] for k in ks}
    ndcg_at_k   = {k: [] for k in ks}

    for (i, query), sim_line in zip(enumerate(queries), sims):
        # for each query, find the most relevant
        relevant_ids, associated_scores = find_relevant_doc_id(query["_id"], qrels)
        relevant_set = set(relevant_ids)  # set for qiucker lookup
        # find up to max_k, then select k in the loop
        found_ids = [corpus["_id"][j] for j in sim_line]
        for k in ks:
            found_ids_k = found_ids[:k]  # select up to k
            found_set = set(found_ids_k)  # set for quicker lookup
            # Ideal cumulative gain: depends on k, so calculated here
            ideal_cumulative_gain = np.sum([
                (2**s - 1) / np.log2(rank + 2.)
                for rank, s in enumerate(np.sort(associated_scores)[::-1][:k])
            ])

            # Recall: our datasets are one query one answer, but if not, this will work then as well
            rec_ = 0.0 if not relevant_set else len(found_set & relevant_set) / len(relevant_set)

            # Discounted cumulative gain
            discounted_cumulative_gain = 0
            for rank_, found_id in enumerate(found_ids_k):
                rank = rank_ + 1
                if found_id in relevant_ids:
                    found_score = associated_scores[relevant_ids.index(found_id)]
                    discounted_cumulative_gain += (2**found_score - 1) / np.log2(rank + 1)
            current_ndcg = 0 if ideal_cumulative_gain == 0 else discounted_cumulative_gain / ideal_cumulative_gain
            # collect results
            recall_at_k[k].append(rec_)
            ndcg_at_k[k].append(current_ndcg)

    results = {"prompt_text": prompt_text}
    for k in ks:
        results[f"recall@{k}"] = stats(recall_at_k[k])
        results[f"ndcg@{k}"]   = stats(ndcg_at_k[k])
    return results

def embed_and_calculate_scores(options, model, dataset_specific_prompts, corpus, queries, qrels):
    """Embed corpus, queries (+prompts) and calculate evaluation scores."""
    # check that we are in the right format
    assert isinstance(queries, datasets.Dataset), f"type(queries) = {type(queries)}, should be datasets.Dataset."
    assert "text" in queries.column_names and "_id" in queries.column_names
    
    # embed the corpus==targets/answers --> prompt has no effect on these
    corpus_embeddings = model.encode(corpus["text"], normalize_embeddings=True, convert_to_tensor=True, 
                                    batch_size=options.batch_size).float()

    # loop over prompts
    results = {}
    for i,p in enumerate(dataset_specific_prompts):
        # encode template(prompt, query)
        prompted_queries = [get_detailed_instruct(p, q, template=options.template) for q in queries[:]["text"]]
        query_embeddings = model.encode(prompted_queries, normalize_embeddings=True, convert_to_tensor=True, 
                                        batch_size=options.batch_size).float()
        report(f"----\nNow in prompt {i}, example: \n{prompted_queries[0]}")
        if isinstance(options.k, int):
            report("RUNNING ON THE OLD VERSION FOR SANITY")
            results[f"prompt{i}"] = {**{"prompt_text": p}, **calculate_scores_with_qrels_one_k(options.k, corpus, queries, qrels, corpus_embeddings, query_embeddings, prompt_text=p)}
        else:
            results[f"prompt{i}"] = {**{"prompt_text": p}, **calculate_scores_with_qrels_multiple_ks(options.k, corpus, queries, qrels, corpus_embeddings, query_embeddings, prompt_text=p)}
    return results



if __name__=="__main__":
    options = parser.parse_args()
    lang=None
    # if lang is given with column notation
    if ":" in options.data_name:
        options.data_name, lang = options.data_name.split(":")

    # download the dataset with data_name
    report("Reading data")
    corpus, queries, qrels = download_dataset(options.data_name, split_to_select=options.split, lang=lang)
    # read distractors:
    report("Reading distractors")
    distractors_raw, field = read_distractors(options.data_name, split_to_select=options.split, lang=lang)
    if distractors_raw:
        # check we have one for each query:
        distractors = {"_id": [], "text":[]}
        
        assert set(queries["_id"])&set(corpus["_id"]) == set(), f'Overlap between query and corpus ids, should not be!!!\n{set(queries["_id"])&set(corpus["_id"])}'
        for i, par in enumerate(distractors_raw):
            q_original, q_par = par[field], par["generated_paraphrase"]
            q, q_id = queries[i]["text"], queries[i]["_id"]
            assert q.rstrip() == q_original.rstrip(), f"Unable to match queries and distractors\n{q} == {q_original}"
            distractors["_id"].append(q_id)
            distractors["text"].append(q_par)

        distractor_corpus = datasets.Dataset.from_dict(distractors)
        report(f"Sanity: {queries[0]=} {distractor_corpus[0]=}")
        # add distractors to corpus2:
        corpus2 = datasets.concatenate_datasets([corpus, distractor_corpus])
    

    report("Downloading prompts")
    # download prompts
    # this returns a list of possible instructions to use on the query side
    if lang is not None:
        prompts = get_prompts(options.data_name, lang=lang)
    else:
        prompts =  get_prompts(options.data_name)

    # add a few prompts based on the template (function as baselines)
    if options.template != "simple":
        # These map to NO_PROMPT=vanilla query and EMPTY: misfilled template
        prompts = ["NO_PROMPT", "EMPTY"] + prompts
    else:
        # For the simple template, only vanilla query
        prompts = ["NO_PROMPT"] + prompts
    
    # pre-make saving locations
    model_safe_name = options.model_name.replace("/", "__")
    data_safe_name = options.data_name.replace("/","__")
    if lang is not None:
        data_safe_name += f"_{lang}" # bring this back now
    specific_prompts = ""
    save_path = f"{options.save_prefix}/{model_safe_name}/{data_safe_name}{specific_prompts if lang is not None else ''}/{options.split}/{options.template}_template/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    safe_k = options.k if isinstance(options.k, int) else "_".join([str(ki) for ki in options.k])

    # download model
    report("Downloading model")
    model = SentenceTransformer(options.model_name, trust_remote_code=True).to('cuda')

    # vanilla eval
    report("Evaluating without distractors")
    results = embed_and_calculate_scores(options, model, prompts, corpus, queries, qrels)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path+f"eval@{safe_k}.json", 'w') as f:
        json.dump(results,f, indent=2)

    if distractors_raw:
        report("Evaluating with distractors")
        results = embed_and_calculate_scores(options, model, prompts, corpus2, queries, qrels)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path+f"eval@{safe_k}_with_distractors.json", 'w') as f:
            json.dump(results,f, indent=2)

