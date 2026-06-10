from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
import datasets
import jsonargparse
import json
import os
from utils.prompts import get_prompts_arcchallenge, get_prompts_tatoeba, get_prompts_summeval, get_prompts_webfaq


parser = jsonargparse.ArgumentParser(prog="Quickly evaluate a prompt for retrieval")
#parser.add_argument('--config', action=ActionConfigFile)
parser.add_argument('--model_name', '--model', type=str, default=None, required=True,
                    help="HF-alias or path to downloaded model.")
parser.add_argument('--data_name', '--dataset', type=str, default="mteb/ARCChallenge",
                    help="HF-alias or path to downloaded dataset.")
parser.add_argument('--split', type=str, default=None,
                    help="Which split to select from dataset.")
parser.add_argument('--template', type=str, default="Instruct-Query", choices=["Instruct-Query", "simple"],
                    help="Which prompting template to use")
parser.add_argument('--use_lang_specific_prompts', action='store_true',
                    help="Use prompts that specifically mention the target language")
parser.add_argument('--k', '--MRR@k', type=int, default=20,
                    help="Which k to use")
#parser.add_argument('--domain', choices = ["A", "QA"], default="A",
#                    help='Which set to search the target from, only target side:A, both:QA')  # TODO impelement this
parser.add_argument('--save_prefix', type=str, default="results_prompts",
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

def download_arcchallenge_from_hub():
    split_to_select = "test"   # this is the only choice
    # download all 3 files for retrieval
    corpus = datasets.load_dataset("mteb/ARCChallenge", "corpus")
    qrels = datasets.load_dataset("mteb/ARCChallenge", "qrels")
    queries = datasets.load_dataset("mteb/ARCChallenge", "queries")
    sanity_check_arcchallenge(corpus, qrels, queries)
    return corpus[split_to_select], \
           qrels[split_to_select], \
           queries[split_to_select], \
           create_binary_relevance_map_arcchallenge(qrels, split=split_to_select)

def download_arcchallenge(split_to_select="test"):
    ds = datasets.load_from_disk("/flash/project_462001394/datasets/arcchallenge")
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["document"]})
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["query"]})
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, None, queries, qrels_map

def download_tatoeba_from_hub(lang, split_to_select="test"):
    ds = datasets.load_dataset("mteb/tatoeba-bitext-mining", lang)
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["sentence1"]}) # non-english
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["sentence2"]}) # english
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, None, queries, qrels_map

def download_tatoeba(lang, split_to_select="test"):
    lang_ = "en-" + lang.split("-")[0][:2]
    ds = datasets.load_from_disk(f"/flash/project_462001394/datasets/tatoeba:{lang_}")
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["non_english"]}) # non-english
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["english"]})  # english
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, None, queries, qrels_map

def download_webfaq(lang, split_to_select="test"):
    ds = datasets.load_from_disk(f"/flash/project_462001394/datasets/web-faq-bitext:{lang}")
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["answer2"]}) # 2's contain the target lang
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["question2"]})
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, None, queries, qrels_map

def download_summeval(split_to_select = "test"):
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
    result3.append([line[0] for line in sort])  # best matches only
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
    # check that we are in the right format
    assert isinstance(queries, datasets.Dataset), f"type(queries) = {type(queries)}, should be datasets.Dataset."
    assert "text" in queries.column_names and "_id" in queries.column_names
    print("Sanity check:")
    print(f"Corpus:\n {corpus['text'][0]}")
    print(f"Query:\n {apply_template('<Example prompt: '+dataset_specific_prompts[0]+'>', queries['text'][0], template=options.template)}")
    # embed the corpus==targets/answers --> prompt has no effect on these
    corpus_embeddings = model.encode(corpus["text"], normalize_embeddings=True)

    k = options.k
    #num_examples=len(queries)
    results = {}
    for p in dataset_specific_prompts:
        # initialize
        results[p] = []

        # encode template(prompt, query)
        prompted_queries = [apply_template(p, q, template=options.template) for q in queries[:]["text"]]
        query_embeddings = model.encode(prompted_queries, normalize_embeddings=True)

        # calculate similarity matrix
        sims = query_embeddings @ corpus_embeddings.T
        # argsort sims to get best matches
        # here the additional ":" is needed together with axis=1, see sanity_check_sorting()
        sims = np.argsort(sims, axis=1)[:, ::-1]

        # then loop over queries and check matches
        for (i, query), sim_line in zip(enumerate(queries), sims):
            # find the id of the correct answer
            relevant_id = find_relevant_doc_id(query["_id"], qrels_dict)  # this returns id of the correct answer
            #print("correct answer id:", relevant_id)
            most_similar_docs = sim_line[:k]  # this is an index
            #print("similarity:", sim_line.shape)
            #print("most similar docs:", most_similar_docs)
            found_ids = [corpus["_id"][i] for i in most_similar_docs]   # this is ids found in search
            #print("found ids:", found_ids)
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


    def stats(t):
            """Extract summary statistics"""
            arr = t # .detach().cpu().numpy().reshape(-1)  # this is not a tensor
            return {"mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75)),
                    "full": str(arr)}
    # average over queries
    records = {}
    for p, values in results.items():
        records[p] = stats(values) #sum(values)/num_examples

    model_name_ = options.model_name.replace("/","__")
    save_path = f'{options.save_prefix}/{model_name_}/{options.data_name}{"_lang_specific" if options.use_lang_specific_prompts else ""}/{options.split}/{options.template}_template/mrr@{k}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(records,f, indent=2)



if __name__=="__main__":
    sanity_check_sorting()
    options = parser.parse_args()
    # dowload dataset and preprocess
    if options.data_name == "mteb/ARCChallenge":
        corpus, qrels, queries, qrels_dict = download_arcchallenge_from_hub()
        prompts=get_prompts_arcchallenge()
    elif options.data_name.lower() == "arcchallenge":
        corpus, qrels, queries, qrels_dict = download_arcchallenge(split_to_select=options.split)
        prompts=get_prompts_arcchallenge()
    elif "tatoeba" in options.data_name.lower():
        assert ":" in options.data_name, "Give language to tatoeba separated by column :, e.g. tatoeba:fin-eng"
        tatoeba_, lang = options.data_name.split(":")
        corpus, qrels, queries, qrels_dict = download_tatoeba(lang, split_to_select=options.split)
        prompts=get_prompts_tatoeba(lang) if options.use_lang_specific_prompts else get_prompts_tatoeba()
    elif "summeval" in options.data_name.lower():
        corpus, qrels, queries, qrels_dict = download_summeval(split_to_select=options.split)
        prompts=get_prompts_summeval()
    elif "webfaq" in options.data_name.lower():
        assert ":" in options.data_name, "Give language to webfaq separated by column :, e.g. webfaq:deu"
        webfaq_, lang = options.data_name.split(":")
        corpus, qrels, queries, qrels_dict = download_webfaq(lang, split_to_select=options.split)
        prompts=get_prompts_webfaq(lang) if options.use_lang_specific_prompts else get_prompts_webfaq()
    # other datasets not implemented yet
    else:
        raise NotImplementedError("Only ARCChallenge+Tatoeba+Summeval+WebFAQ implemented")
    #print(corpus)
    #print(qrels_dict)
    #print(queries)
    # download model
    model = SentenceTransformer(options.model_name)
    calculate_scores(options, model, prompts, corpus, queries, qrels_dict)
