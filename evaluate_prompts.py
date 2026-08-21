from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
import datasets
import jsonargparse
import json
import os
import pickle
import random
from utils.prompts import get_prompts, get_detailed_instruct
from utils.dataset_handling import download_dataset


# set random behaviour for replication
seed = 42
random.seed(seed)
np.random.seed(seed)




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

def sanity_check_sorting():
    report("Sanity checking matrix operations")
    a = normalize(np.random.random((100, 256)))
    qrels = np.random.permutation(list(range(a.shape[0])))
    b = a[qrels,:] + (np.random.random(a.shape)-0.5)*0.01
    b = normalize(b)

    # TEST 1: separate multiplication
    result1 = []
    for bi in b:
        sims = bi@a.T
        result1.append(np.argmax(sims))
    # see if results match relevances
    assert (result1 == qrels).all(), f"qrels = {qrels} != {result1} == result1"

    #print("-----")
    # TEST2: Matrix multiplication + argsort instead of argmax
    result2 = []
    sims = b@a.T
    sort_collect = []
    for line in sims:
        s = np.argsort(line)[::-1]
        result2.append(s[0])
        sort_collect.append(s)
    # see if results match qrels
    assert (result2 == qrels).all(), f"qrels = {qrels} != {result2} == result2"

    #print("----")
    # TEST3: simultaneous sorting
    result3 = []
    sims = b@a.T
    sort = np.argsort(sims, axis=1)[:, ::-1]
    result3.append([line[0] for line in sort])  # best matches only
    assert (result3 == qrels).all(), f"qrels = {qrels} != {result3} == result3"

    # TEST4: if we arrived to the conclusion the same way
    assert np.allclose(sort-sort_collect,0)

    report("All tests passed")


def apply_template(prompt, query, template="Instruct-Query"):
    return get_detailed_instruct(prompt, query, template=template)

def calculate_non_rank_metrics(found_ids, relevant_ids):
    """
    Calculate non-rank related metrics. 
    Both lists assumed to be sorted, found_ids by similarity and relevant_ids by relevance.
    """
    tp = sum(1 for fid in found_ids if fid in relevant_ids)
    # recall is normal
    recall = tp / len(relevant_ids)
    # precision is artificially deflated: if there are 2 relevant docs
    # but we set k=10 for 10 found ids
    # even if we found the two relevant at the top
    # precision will be low
    # hence, also r-precision, which is basically recall again in most cases
    precision = tp / len(found_ids)
    tp_ = sum(1 for fid in found_ids[:len(relevant_ids)] if fid in relevant_ids)
    rprecision = tp_ / len(relevant_ids)
    # for F1, we are interested in the top1 match 
    # this is for bitext mining, a task with one to one binary qrels
    # this is why we're not using this common formula
    #f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    # but this, which is basically just accuracy
    # we will still call it F1 because that what MTEB calls it, even if it reduces to acc
    f1_at_1 = 1 if found_ids[0] == relevant_ids[0] else 0
    return f1_at_1, recall, precision, rprecision

def sanity_check_logic(k=2):
    """Check the logic of metric calculation"""
    # construct examples for the sanity check
    corpus = datasets.Dataset.from_dict({"_id":["c14","c21","c33","c41"], "text":["a","b","c","d"]})
    queries = datasets.Dataset.from_dict({"_id": ["q1", "q2", "q4"], "text":["correct is mostly a, but d is okay", "correct is c", "correct is a+b+c"]})
    qrels = datasets.Dataset.from_dict({"query_id":["q1", "q1", "q2", "q4", "q4", "q4"], "corpus_id":["c14", "c41", "c33", "c14","c21","c33"], "score": [1, 0.9, 1, 1,0.99,0.9]})
    # similarities: normally, this would be emb(queries)@emb(corpus).T
    # Perfect match for q1 (a is highest, d second highes) and second best for q2 (b is highest, followed by c, which is correct)
    # for the last on, the correct ones are correct but the last to are in wrong order
    sims = np.array([[0.8, -0.3, 0.3, 0.6], [-0.2, 0.92, 0.90, 0.1], [0.9, 0.6, 0.7, -0.2]])
    # argsort to get most similar indices
    sims = np.argsort(sims, axis=1)[:, ::-1]
    recall_at_k = []
    mrr_at_k = []
    ndcg_at_k = []
    f1_at_1 = []
    precision_at_k = []
    rprecision_at_k = []
    for (i, query), sim_line in zip(enumerate(queries), sims):
        print("In Q:", query["_id"],", text=", query["text"])
        relevant_ids, associated_scores = find_relevant_doc_id(query["_id"], qrels)
        most_similar_docs = sim_line[:k]
        found_ids = [corpus["_id"][j] for j in most_similar_docs]
        print(f"{relevant_ids=}, {associated_scores=}")
        ideal_cumulative_gain = np.sum([(2**s-1)/np.log2(rank+1+1.) for rank, s in enumerate(np.sort(associated_scores)[::-1][:k])])   # UP TO k
        # if everything was perfect: highest score at rank1, second at rank2
        # also +1+1 since the rank is zero indexed here -> one +1 to fix rank and other is in the formula
        discounted_cumulative_gain = 0
        mrr_ = 0
        # we can already calculate some results with no rank information
        #rec_ = sum(1 for fid in found_ids if fid in relevant_ids) / len(relevant_ids)
        f1_, rec_, prec_, rprec_= calculate_non_rank_metrics(found_ids, relevant_ids)
        for rank_, found_id in enumerate(found_ids):
            rank = rank_+ 1 # fix zero indexing
            print(f"{found_id} found in rank {rank}")
            if found_id in relevant_ids:
                mrr_ = 1/(rank) if mrr_==0 else mrr_  # again, only first match counts, so only set at highest rank
                found_score = associated_scores[relevant_ids.index(found_id)]
                discounted_cumulative_gain += (2**found_score-1)/np.log2(rank+1)
        current_ndcg_at_k = 1 if ideal_cumulative_gain == 0 else discounted_cumulative_gain/ideal_cumulative_gain # 1 for "nothing relevant was to be discovered"
        ndcg_at_k.append(current_ndcg_at_k)
        mrr_at_k.append(mrr_)
        recall_at_k.append(rec_)
        f1_at_1.append(f1_)
        precision_at_k.append(prec_)
        rprecision_at_k.append(rprec_)
    print("Final results")
    print("RECALL", np.mean(recall_at_k))
    print("MRR", np.mean(mrr_at_k))
    print("NDCG", np.mean(ndcg_at_k))
    print("F1", np.mean(f1_at_1))
    print("R-PRECISION", np.mean(rprecision_at_k))

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

def calculate_scores(k, corpus, queries, qrels, corpus_embeddings, query_embeddings):
    """
    Calculate MRR@k and recall@k for given queries and corpus
    corpus=dataset of targets (columns _id and text)
    queries=dataset of queries (columns _id and text)
    qrels=either a dataset of (query_id, corpus_id, score) or binary dict(query_id:corpus_id)
    corpus_embeddings = matrix of corpus embeddings (in the same order as corpus)
    query_embeddings = matrix of query embeddings (again, same order and possibly, a prompt has been added before calculation)
    """
    # calculate similarity matrix
    sims = query_embeddings @ corpus_embeddings.T
    # argsort sims to get best matches
    # here the additional ":" is needed together with axis=1, see sanity_check_sorting()
    sims = np.argsort(sims, axis=1)[:, ::-1]
    # initialize data collection
    recall_at_k = []
    mrr_at_k = []
    ndcg_at_k = []
    f1_at_1 = []  # this is the metric for bitext mining (binary task, hence "at_1")
    precision_at_k = []
    rprecision_at_k = []
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
        
        # we can already calculate some results with no rank information
        #rec_ = sum(1 for fid in found_ids if fid in relevant_ids) / len(relevant_ids)
        f1_, rec_, prec_, rprec_= calculate_non_rank_metrics(found_ids, relevant_ids)
        # initialize the rank dependent metrics
        discounted_cumulative_gain = 0
        mrr_ = 0
        for rank_, found_id in enumerate(found_ids):
            rank = rank_+ 1 # fix zero indexing
            if found_id in relevant_ids:
                mrr_ = 1/(rank) if mrr_==0 else mrr_  # again, only first match counts, so only set at highest rank
                found_score = associated_scores[relevant_ids.index(found_id)]
                discounted_cumulative_gain += (2**found_score-1)/np.log2(rank+1)
        # calculate ndcg@k for this query
        current_ndcg_at_k = 0 if ideal_cumulative_gain == 0 else discounted_cumulative_gain/ideal_cumulative_gain # 0 if nothing was to be discovered
        # collect results
        ndcg_at_k.append(current_ndcg_at_k)
        mrr_at_k.append(mrr_)
        recall_at_k.append(rec_)
        f1_at_1.append(f1_)
        precision_at_k.append(prec_)
        rprecision_at_k.append(rprec_)

    return {f"mrr@{k}": stats(mrr_at_k),
            f"recall@{k}": stats(recall_at_k),
            f"ndcg@{k}": stats(ndcg_at_k),
            "F1": stats(f1_at_1),
            f"precision@{k}": stats(precision_at_k),
            f"rprecision@{k}": stats(rprecision_at_k),
            }


def embed_and_calculate_scores(options, dataset_specific_prompts, corpus, queries, qrels):
    """Embed corpus, queries (+prompts) and calculate evaluation scores."""
    # check that we are in the right format
    assert isinstance(queries, datasets.Dataset), f"type(queries) = {type(queries)}, should be datasets.Dataset."
    assert "text" in queries.column_names and "_id" in queries.column_names
    
    # download model
    model = SentenceTransformer(options.model_name)
    # embed the corpus==targets/answers --> prompt has no effect on these
    corpus_embeddings = model.encode(corpus["text"], normalize_embeddings=True, batch_size=options.batch_size)
    if options.embeddings:
        os.makedirs(os.path.dirname(options.embeddings), exist_ok=True)
        write_embeddings(file=options.embeddings, key="qrels", data=qrels, embeddings=None)
        write_embeddings(file=options.embeddings, key="corpus", data=corpus, embeddings=corpus_embeddings)

    # loop over prompts
    results = {}
    for i,p in enumerate(dataset_specific_prompts):
        # encode template(prompt, query)
        prompted_queries = [apply_template(p, q, template=options.template) for q in queries[:]["text"]]
        query_embeddings = model.encode(prompted_queries, normalize_embeddings=True, batch_size=options.batch_size)
        if options.embeddings:
            # construct a new dict here
            write_embeddings(file=options.embeddings, 
                             key=p, 
                             data={"_id": queries["_id"], "text": prompted_queries}, 
                             embeddings=query_embeddings)
        report(f"----\nNow in prompt {i}, example: \n{prompted_queries[0]}")
        results[f"prompt{i}"] = {**{"prompt_text": p}, **calculate_scores(options.k, corpus, queries, qrels, corpus_embeddings, query_embeddings)}

    os.makedirs(os.path.dirname(options.save_path), exist_ok=True)
    with open(options.save_path, 'w') as f:
        json.dump(results,f, indent=2)


def read_embeddings_and_calculate_scores(options, dataset_specific_prompts, corpus, queries, qrels):
    corpus_embeddings = None
    results = {}
    i = 0 # index for prompts
    for data in yield_from_pkl(options.embeddings):
        # First 2 saved values are qrels and corpus embeddings
        # check that they match, and set corpus_embeddings
        if data["key"] == "qrels":
            # qrels is saved in "text_ids"
            if isinstance(qrels, dict):
                print(data["data"])
                assert data["data"] == qrels, "Mismatch between qrels and precalculated embeddings"
            else:
                assert data["data"] == qrels.to_dict(), "Mismatch between qrels and precalculated embeddings"
            continue
        if data["key"] == "corpus":
            assert data["data"]["_id"] == corpus["_id"], "Mismatch between loaded corpus and precalculated embeddings"
            assert data["data"]["text"] == corpus["text"], "Mismatch between loaded corpus and precalculated embeddings"
            # set corpus embeddings
            corpus_embeddings = data["data"]["embeddings"]
            continue
        # after reading, we can calculate:
        prompt = data["key"]  # saved here
        report(f"In prompt {prompt}")
        query_embeddings = data["data"]["embeddings"]
        # check that prompt order is the same --> we can save with the same prompt order
        assert get_detailed_instruct(prompt, queries[0]["text"], template=options.template) == data["data"]["text"][0], \
            f'{get_detailed_instruct(prompt, queries[0]["text"], template=options.template)} != {data["data"]["text"][0]}'
        results[f"prompt{i}"] = {**{"prompt_text": prompt}, **calculate_scores(options.k, corpus, queries, qrels, corpus_embeddings, query_embeddings)}
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
    corpus, queries, qrels = download_dataset(options.data_name, lang=lang, split_to_select=options.split, downsample=options.num_examples)
    prompts = get_prompts(options.data_name, lang=lang)
    report("Sanity check: What was downloaded?")
    report(prompts[0])
    report(queries[0])
    report(corpus[0])
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
    # only set this if the prefix was given
    options.embeddings = "" if not options.embedding_prefix else f'{options.embedding_prefix}/{model_name_}/{data_safe_name}{"_lang_specific" if options.use_lang_specific_prompts else ""}/{options.split}/{options.template}_embeddings.pkl'
    options.save_path =  f'{options.save_prefix}/{model_name_}/{data_safe_name}{"_lang_specific" if options.use_lang_specific_prompts else ""}/{options.split}/{options.template}_template/results@{options.k}.json'
    
    if options.embeddings != "" and os.path.exists(options.embeddings):
        # we have precalculated embeddings
        report(f"Using precalculated embeddings at {options.embeddings} for score calculation")
        read_embeddings_and_calculate_scores(options, prompts, corpus, queries, qrels)
    else:
        # calculate everything from scratch
        report("Calculating embeddings and score")
        embed_and_calculate_scores(options, prompts, corpus, queries, qrels)

