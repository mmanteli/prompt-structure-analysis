from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
import datasets
import jsonargparse
import json
import os
from prompts import prompts_arcchallenge, prompts_tatoeba, prompts_summeval


parser = jsonargparse.ArgumentParser(prog="Quickly evaluate a prompt for retrieval")
#parser.add_argument('--config', action=ActionConfigFile)
parser.add_argument('--model_name', '--model', type=str, default=None, required=True,
                    help="HF-alias or path to downloaded model.")
parser.add_argument('--data_name', '--dataset', type=str, default="mteb/ARCChallenge",  # for now
                    help="HF-alias or path to downloaded dataset.")
parser.add_argument('--template', type=str, default="Instruct-Query", choices=["Instruct-Query", "simple"],
                    help="Which prompting template to use")
parser.add_argument('--k', '--MRR@k', type=int, default=20,
                    help="Which k to use")
parser.add_argument('--domain', choices = ["A", "QA"], default="A",
                    help='From which set to search the target from, only target side:A, both:QA')
parser.add_argument('--save_prefix', type=str, default="results",
                    help="Saving path, model_name and k added in script")




def sanity_check_arcchallenge(corpus, qrels, queries, split="test"):
    print("---------------SANITY CHECK----------------")
    print("--see that the given example makes sense---")
    print(queries[split][0])
    query_id = queries[split][0]["_id"]
    qrel = qrels.filter(lambda example: example["query-id"]==query_id)
    assert len(qrel) == 1, "More than one relevant document found. Should not be possible for binary relevance."
    corpus_id = qrels[split][0]["corpus-id"]
    target = corpus.filter(lambda example: example["_id"] == corpus_id)
    print(target[split][0])
    print("-------------SANITY CHECK END--------------")


def create_binary_relevance_map_arcchallenge(qrels, split="test"):
    qrels_dict = {}
    for d in qrels[split]:
        q = d["query-id"]
        c = d['corpus-id']
        if d['score'] == 1:
            qrels_dict[q] = c
    return qrels_dict

def download_arcchallenge():
    split_to_select = "test"
    # download all 3 files for retrieval
    corpus = datasets.load_dataset("mteb/ARCChallenge", "corpus")
    qrels = datasets.load_dataset("mteb/ARCChallenge", "qrels")
    queries = datasets.load_dataset("mteb/ARCChallenge", "queries")
    sanity_check_arcchallenge(corpus, qrels, queries)
    return corpus[split_to_select], \
           qrels[split_to_select], \
           queries[split_to_select], \
           create_binary_relevance_map_arcchallenge(qrels, split=split_to_select)

def download_tatoeba(lang):
    split_to_select = "test"
    ds = datasets.load_dataset("mteb/tatoeba-bitext-mining", lang)
    corpus = datasets.Dataset.from_dict({"_id": range(ds[split_to_select]),
                                         "text":ds[split_to_select]["sentence1"]}) # non-english
    queries = datasets.Dataset.from_dict({"_id": range(ds[split_to_select]),
                                         "text":ds[split_to_select]["sentence2"]}) # english
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, None, queries, qrels_map

def download_summeval():
    # this is downloaded locally!
    split_to_select = "test"
    #'text','summary'
    ds = datasets.load_from_disk("/flash/project_462001394/datasets/summeval-2")[split_to_select]
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds)), "text":ds["text"]}) # non-english
    queries = datasets.Dataset.from_dict({"_id": range(len(ds)), "text":ds["summary"]}) # english
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, None, queries, qrels_map


def sanity_check_sorting():
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
    for line in sort:
        result3.append(line[0])
    assert (result3 == qrels).all(), f"qrels = {qrels} != {result3} == result3"

    # TEST4: if we arrived to the conclusion the same way
    assert np.allclose(sort-sort_collect,0)

    print("All tests passed")


def apply_template(prompt, query, template="Instruct-Query"):
    if template=="Instruct-Query":
        return f"""Instruct: {prompt}\nQuery: {query}"""
    if template=="simple":
        return f"""{prompt}. {query}"""
    raise NotImplementedError


def find_relevant_doc_id(query_id, qrels_dict):
    return qrels_dict[query_id] if query_id in qrels_dict else qrels_dict[f"ARC-Challenge-q-{query_id}"]


def calculate_scores(options, model, dataset_specific_prompts, corpus, queries, qrels_dict):
    """Calculate MRR@k for a dataset with different prompts."""
    # embed the corpus==targets/answers --> prompt has no effect on these
    corpus_embeddings = model.encode(corpus["text"], normalize_embeddings=True)
    # see that ids and texts are given correctly
    assert type(queries) is datasets.Dataset()
    assert "text" in queries.colums and "_id" in queries.colums
    # set k to a local variable, as if we introduce queries to search pool, we need to modify it
    k = options.k
    num_examples=len(queries)
    results = {}
    for p in dataset_specific_prompts:
        # initialize
        results[p] = []
        # encode template(prompt, query)
        prompted_queries = [apply_template(p, q, template=options.template) for q in queries[:]["text"]]
        query_embeddings = model.encode(prompted_queries, normalize_embeddings=True)
        # calculate similarity matrix
        if options.domain == "A":  # only consider the corpus==target side
            sims = query_embeddings @ corpus_embeddings.T
        else: # also consider the other queries
            raise NotImplementedError
            # TODO something about qrels???
            #sims = query_embeddings @ torch.cat((corpus_embeddings, query_embeddings)).T
            #k = options.k + 1 # add one since we introduce a perfect match -> the query itself
        # argsort sims to get best matches
        # here the additional ":" is needed together with axis=1, see sanity_check_sorting()
        sims = np.argsort(sims, axis=1)[:, ::-1]
        # then loop over queries and check matches
        for (i, query), sim_line in zip(enumerate(queries), sims):
            # find the id of the correct answer
            relevant_id = find_relevant_doc_id(query["_id"], qrels_dict)  # this id is str
            most_similar_docs = sim_line[:k]  # this is an index
            found_ids = [corpus["_id"] for i in most_similar_docs]   # this is again str
            score = 0
            if relevant_id in found_ids:
                rank_indices = np.where(np.asarray(found_ids) == relevant_id)[0]
                assert len(rank_indices) == 1, f"Duplicate corpus ID in top-k for query {i}"
                # rank is 1-indexed: position 0 → rank 1
                rank = rank_indices[0] + 1
                score = 1.0 / rank
                #score = (k-rank)/k  <- linear decay option
            #results[p].append(relevant_id in found_ids)   # this is recall@k
            results[p].append(score)

    # average over queries
    records = {}
    for p, values in results.items():
        #print(p)
        #print(sum(values)/num_examples)
        #print("")
        records[p] = sum(values)/num_examples

    k_ = k if options.domain=="A" else f"{k-1}_full_domain"
    save_path = f'{options.save_prefix}/{options.model_name.replace("/","__")}/arcchallenge/mrr@{k_}__{options.template}-template.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(records,f)



if __name__=="__main__":
    sanity_check_sorting()
    options = parser.parse_args()
    # download model
    model = SentenceTransformer(options.model_name)
    # dowload dataset and preprocess
    if options.data_name == "mteb/ARCChallenge":
        corpus, qrels, queries, qrels_dict = download_arcchallenge()
        prompts=prompts_arcchallenge
    elif "tatoeba" in options.data_name.lower():
        assert ":" in options.data_name, "Give language to tatoeba separated by column :, e.g. tatoeba:fin-eng"
        tatoeba_, lang = options.data_name.split(":")
        corpus, qrels, queries, qrels_dict = download_tatoeba(lang)
        prompts=prompts_tatoeba
    elif "summeval" in options.data_name.lower():
        corpus, qrels, queries, qrels_dict = download_summeval()
        prompts=prompts_summeval

    # other datasets not implemented yet
    else:
        raise NotImplementedError("Only ARCChallenge+Tatoeba+Summeval implemented")
