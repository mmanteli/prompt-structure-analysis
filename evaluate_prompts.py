from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
import datasets
import jsonargparse
import json
import os
import pickle
import random
from utils.prompts import get_prompts_arcchallenge, get_prompts_tatoeba, get_prompts_summeval, get_prompts_webfaq, get_detailed_instruct
from utils.dataset_handling import download_dataset


# set random behaviour for replication
seed = 42
random.seed(seed)
np.random.seed(seed)




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
parser.add_argument('--use_lang_specific_prompts', '--use_language_specific_prompts', action='store_true',
                    help="Use prompts that specifically mention the target language")
parser.add_argument('--k', '--MRR@k', type=int, default=10,
                    help="Which k to use for MRR and Recall")
parser.add_argument('--batch_size', type=int, default=16,
                    help="Which k to use")
parser.add_argument('--num_examples', type=int|bool, default=5000,
                    help="For largest datasets, number of examples to downsample to, set to False for no downsampling")
parser.add_argument('--embedding_prefix', type=str|bool, default=False,
                    help="prefix to save embedings to, works similar to --save_prefix")
#parser.add_argument('--domain', choices = ["A", "QA"], default="A",
#                    help='Which set to search the target from, only target side:A, both:QA')  # TODO impelement this
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

def write_embeddings(file, key, text_ids, texts, embeddings):
    d = {"key": key, "ids": text_ids, "texts": texts, "embeddings": embeddings}
    append_pkl(d, file)

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


def sanity_check_logic(k=1):
    """Check the logic of recall@k/MRR@k calculation"""
    # construct examples
    corpus = datasets.Dataset.from_dict({"_id":["c1","c2","c3","c4"], "text":["a","b","c","d"]})
    queries = datasets.Dataset.from_dict({"_id": ["q1", "q2"], "text":["correct is a", "correct is c"]})
    qrels_dict = {"q1":"c1", "q2":"c3"}
    # similarities: normally, this would be emb(queries)@emb(corpus).T
    # Perfect match for q1 (a is highest) and second best for q2 (b is highest, followed by c)
    sims = np.array([[0.8, -0.3, 0.3, 0.2], [-0.2, 0.92, 0.9, 0.1]])
    # argsort to get most similar indices
    sims = np.argsort(sims, axis=1)[:, ::-1]
    recalls=[]
    mrrs=[]
    for (i, query), sim_line in zip(enumerate(queries), sims):
        relevant_id = find_relevant_doc_id(query["_id"], qrels_dict)
        most_similar_docs = sim_line[:k]
        found_ids = [corpus["_id"][j] for j in most_similar_docs]
        if relevant_id in found_ids:
            rank_indices = np.where(np.asarray(found_ids) == relevant_id)[0]
            print(rank_indices)
            print("Correct found")
            print("\tQ:", query["text"])
            # problem: this does not produce them "best match first" but in the order of the dataset
            print("\tfound: (best match first)", [corpus[int(j)]["text"] for j in most_similar_docs])
            recalls.append(1.0)
            mrrs.append(1.0/(rank_indices[0]+1))
        else:
            print("Did not find correct")
            print("\tQ:", query["text"])
            print("\tfound: (best match first)", [corpus[int(j)]["text"] for j in most_similar_docs])
            recalls.append(0.0)
            mrrs.append(0.0)
    results = (np.mean(recalls), np.mean(mrrs))
    print(results)

def find_relevant_doc_id(query_id, qrels_dict):
    return qrels_dict[query_id] if query_id in qrels_dict else qrels_dict[f"ARC-Challenge-q-{query_id}"]

def stats(t):
    """Extract summary statistics"""
    arr = t # .detach().cpu().numpy().reshape(-1)  # apply these if you're handling tensors
    return {"mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "full": str(arr)}


def calculate_scores(k, corpus, queries, qrels_dict, corpus_embeddings, query_embeddings):
    """
    Calculate MRR@k and recall@k for given queries and corpus
    corpus=dataset of targets (ids and texts)
    queries=dataset of queries (ids and texts)
    qrels=dict of corpus_ids to query_ids
    corpus_embeddings = matrix of corpus embeddings
    query_embeddings = matrix of query embeddings (and possibly, a prompt has been added before calculation)
    """
    # calculate similarity matrix
    sims = query_embeddings @ corpus_embeddings.T
    # argsort sims to get best matches
    # here the additional ":" is needed together with axis=1, see sanity_check_sorting()
    sims = np.argsort(sims, axis=1)[:, ::-1]

    mrrscores=[]
    recallscores=[]
    # then loop over queries and check matches
    for (i, query), sim_line in zip(enumerate(queries), sims):
        # find the id of the correct answer
        relevant_id = find_relevant_doc_id(query["_id"], qrels_dict)  # this returns id of the correct answer
        #print("correct answer id:", relevant_id)
        most_similar_docs = sim_line[:k]
        #print("similarity:", sim_line.shape)
        #print("most similar docs:", most_similar_docs)
        found_ids = [corpus["_id"][j] for j in most_similar_docs]   # this is ids found in search
        #print("found ids:", found_ids)d
        mrr = 0
        recall = 0
        if relevant_id in found_ids:
            rank_indices = np.where(np.asarray(found_ids) == relevant_id)[0]
            assert len(rank_indices) == 1, f"Duplicate corpus ID in top-k for query {i}, should NOT HAPPEN"
            # rank is 1-indexed: position 0 -> rank 1
            rank = rank_indices[0] + 1
            mrr = 1.0 / rank
            recall = 1  # relevant_id is in found_ids
        mrrscores.append(mrr)
        recallscores.append(recall)

    return {f"mrr@{k}": stats(mrrscores), f"recall@{k}": stats(recallscores)}


def embed_and_calculate_scores(options, dataset_specific_prompts, corpus, queries, qrels_dict):
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
        write_embeddings(file=options.embeddings, key="qrels", text_ids=qrels_dict,texts=[], embeddings=[])
        write_embeddings(file=options.embeddings, key="corpus", text_ids=corpus["_id"],texts=corpus["text"], embeddings=corpus_embeddings)

    # loop over prompts
    results = {}
    for i,p in enumerate(dataset_specific_prompts):
        # encode template(prompt, query)
        prompted_queries = [apply_template(p, q, template=options.template) for q in queries[:]["text"]]
        query_embeddings = model.encode(prompted_queries, normalize_embeddings=True, batch_size=options.batch_size)
        if options.embeddings:
            write_embeddings(file=options.embeddings, key=p, text_ids=queries["_id"], texts=prompted_queries, embeddings=query_embeddings)
        report(f"----\nNow in prompt {i}, example: \n{prompted_queries[0]}")
        results[f"prompt{i}"] = {**{"prompt_text": p}, **calculate_scores(options.k, corpus, queries, qrels_dict, corpus_embeddings, query_embeddings)}

    os.makedirs(os.path.dirname(options.save_path), exist_ok=True)
    with open(options.save_path, 'w') as f:
        json.dump(results,f, indent=2)


def read_embeddings_and_calculate_scores(options, dataset_specific_prompts, corpus, queries, qrels_dict):
    corpus_embeddings = None
    results = {}
    i = 0 # index for prompts
    for data in yield_from_pkl(options.embeddings):
        # First 2 saved values are qrels and corpus embeddings
        # check that they match, and set corpus_embeddings
        if data["key"] == "qrels":
            # qrels is saved in "text_ids"
            assert qrels_dict == data["ids"], "Mismatch between loaded qrels and precalculated embeddings"
            continue
        if data["key"] == "corpus":
            assert corpus["_id"] == data["ids"], "Mismatch between loaded corpus and precalculated embeddings"
            assert corpus["text"] == data["texts"], "Mismatch between loaded corpus and precalculated embeddings"
            # set corpus embeddings
            corpus_embeddings = data["embeddings"]
            continue
        # after reading, we can calculate:
        prompt = data["key"]  # saved here
        report(f"In prompt {prompt}")
        query_embeddings = data["embeddings"]
        # check that prompt order is the same
        assert get_detailed_instruct(prompt, queries[0]["text"], template=options.template) == data["texts"][0], f"{get_detailed_instruct(prompt, queries[0], template=options.template)} == {data['texts'][0]}"
        results[f"prompt{i}"] = {**{"prompt_text": prompt}, **calculate_scores(options.k, corpus, queries, qrels_dict, corpus_embeddings, query_embeddings)}
        i += 1

    os.makedirs(os.path.dirname(options.save_path), exist_ok=True)
    with open(options.save_path, 'w') as f:
        json.dump(results,f, indent=2)



if __name__=="__main__":
    #sanity_check_sorting()
    options = parser.parse_args()
    # dowload dataset and preprocess
    if options.data_name == "mteb/ARCChallenge":
        # directs to HF-hub download
        corpus, qrels, queries, qrels_dict = download_dataset("arcchallenge", "test", None, local=False, num_examples=options.num_examples)
        prompts=get_prompts_arcchallenge()
    elif options.data_name.lower() == "arcchallenge":
        # directs to local download with preprocessed data
        corpus, qrels, queries, qrels_dict = download_dataset("arcchallenge", options.split, None, num_examples=options.num_examples)
        prompts=get_prompts_arcchallenge()
    elif "tatoeba" in options.data_name.lower():
        assert ":" in options.data_name, "Give language to tatoeba separated by column :, e.g. tatoeba:fin-eng"
        tatoeba_, lang = options.data_name.split(":")
        corpus, qrels, queries, qrels_dict = download_dataset("tatoeba", options.split, lang, num_examples=options.num_examples)
        prompts=get_prompts_tatoeba(lang) if options.use_lang_specific_prompts else get_prompts_tatoeba()
    elif "summeval" in options.data_name.lower():
        corpus, qrels, queries, qrels_dict = download_dataset("summeval", options.split, None, num_examples=options.num_examples)
        prompts=get_prompts_summeval()
    elif "webfaq" in options.data_name.lower():
        assert ":" in options.data_name, "Give language to webfaq separated by column :, e.g. webfaq:deu"
        webfaq_, lang = options.data_name.split(":")
        corpus, qrels, queries, qrels_dict =  download_dataset("webfaq", options.split, lang, num_examples=options.num_examples)
        prompts=get_prompts_webfaq(lang) if options.use_lang_specific_prompts else get_prompts_webfaq()
    # other datasets not implemented yet
    else:
        raise NotImplementedError("Only ARCChallenge+Tatoeba+Summeval+WebFAQ implemented")
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
    # only set this if the prefix was given
    options.embeddings = "" if not options.embedding_prefix else f'{options.embedding_prefix}/{model_name_}/{options.data_name}{"_lang_specific" if options.use_lang_specific_prompts else ""}/{options.split}/{options.template}_embeddings.pkl'
    options.save_path =  f'{options.save_prefix}/{model_name_}/{options.data_name}{"_lang_specific" if options.use_lang_specific_prompts else ""}/{options.split}/{options.template}_template/results@{options.k}.json'
    
    if options.embeddings != "" and os.path.exists(options.embeddings):
        # we have precalculated embeddings
        report(f"Using precalculated embeddings at {options.embeddings} for score calculation")
        read_embeddings_and_calculate_scores(options, prompts, corpus, queries, qrels_dict)
    else:
        # calculate everything from scratch
        report("Calculating embeddings and score")
        embed_and_calculate_scores(options, prompts, corpus, queries, qrels_dict)
