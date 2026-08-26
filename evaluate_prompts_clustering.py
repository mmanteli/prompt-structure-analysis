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
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, adjusted_mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

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

def calculate_scores(embeddings, true_labels):
    """
    Calculate V score and AMI for K-means clustering results,
    and Log-regression for classification
    from sklearn.linear_model import LogisticRegression
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0).fit(X, y)
    clf.predict(X[:2, :])
    array([0, 0])
    clf.predict_proba(X[:2, :])
    array([[9.82e-01, 1.82e-02, 1.44e-08],
        [9.72e-01, 2.82e-02, 3.02e-08]])
    clf.score(X, y)
    0.97
    """
    if isinstance(true_labels[0],str):
        # make them numbers
        label2id = {v:k for k, v in enumerate(np.unique(true_labels))}
        true_labels = [label2id[t] for t in true_labels]
    # classification
    clf = LogisticRegression(random_state=seed, max_iter=1000).fit(embeddings, true_labels)
    # accuracy
    acc = clf.score(embeddings, true_labels)
    # F1
    pred_labels_logr = clf.predict(embeddings)
    f1 = f1_score(true_labels, pred_labels_logr, average="micro")


    # clustering
    num_labels = len(np.unique(true_labels))
    kmeans = KMeans(n_clusters=num_labels, random_state=seed)
    pred_labels_clst = kmeans.fit_predict(embeddings)

    return {"V-score": v_measure_score(true_labels, pred_labels_clst),
            "AMI": adjusted_mutual_info_score(true_labels, pred_labels_clst),
            "Accuracy": acc,
            "F1":f1
            }


def embed_and_calculate_cluster_scores(options, dataset_specific_prompts, corpus, labels):
    """Embed corpus (+prompts) and calculate evaluation scores."""
    # check that we are in the right format
    if isinstance(corpus, datasets.Dataset):
        assert "text" in corpus.keys(), f"Cannot find field 'text' in corpus columns, {corpus.keys()=}"
        corpus = corpus["text"]
    
    # download model
    model = SentenceTransformer(options.model_name)
    # Normally, we would encode the "target"/corpus here, but since we will be adding prompt to all
    # text, we can skip this step
    corpus_embeddings = [None]*len(labels) #model.encode(corpus, normalize_embeddings=True, batch_size=options.batch_size)
    if options.embeddings:
        os.makedirs(os.path.dirname(options.embeddings), exist_ok=True)
        write_embeddings(file=options.embeddings, key="labels", data={"label":labels}, embeddings=[None]*len(labels))
        write_embeddings(file=options.embeddings, key="corpus", data={"text": corpus}, embeddings=corpus_embeddings)

    # loop over prompts
    results = {}
    for i,p in enumerate(dataset_specific_prompts):
        # encode template(prompt, query)
        prompted_texts = [apply_template(p, c, template=options.template) for c in corpus]
        prompted_embeddings = model.encode(prompted_texts, normalize_embeddings=True, batch_size=options.batch_size)
        if options.embeddings:
            # construct a new dict here
            write_embeddings(file=options.embeddings, 
                             key=p, 
                             data={"text": prompted_texts}, 
                             embeddings=prompted_embeddings)
        report(f"----\nNow in prompt {i}, example: \n{prompted_texts[0]}")
        results[f"prompt{i}"] = {**{"prompt_text": p}, **calculate_scores(prompted_embeddings, labels)}

    os.makedirs(os.path.dirname(options.save_path), exist_ok=True)
    with open(options.save_path, 'w') as f:
        json.dump(results,f, indent=2)


def read_embeddings_and_calculate_cluster_scores(options, dataset_specific_prompts, corpus, labels):

    if isinstance(corpus, datasets.Dataset):
        assert "text" in corpus.keys(), f"Cannot find field 'text' in corpus columns, {corpus.keys()=}"
        corpus = corpus["text"]
    results = {}
    i = 0 # index for prompts
    for data in yield_from_pkl(options.embeddings):
        # First 2 saved values are qrels and corpus embeddings
        # check that they match, and set corpus_embeddings
        if data["key"] == "labels":
            assert data["data"]["label"] == labels, "Mismatch between labels and reading precalculated embeddings"
            continue
        if data["key"] == "corpus":
            assert data["data"]["text"] == corpus, "Mismatch between loaded corpus and precalculated embeddings"
            # set corpus embeddings
            corpus_embeddings = data["data"]["embeddings"]
            continue
        # after reading, we can calculate:
        prompt = data["key"]  # saved here
        report(f"In prompt {prompt}")
        prompted_embeddings = data["data"]["embeddings"]
        # check that prompt order is the same --> we can save with the same prompt order
        assert get_detailed_instruct(prompt, corpus[0], template=options.template) == data["data"]["text"][0], \
            f'{get_detailed_instruct(prompt, corpus[0], template=options.template) != data["data"]["text"][0]}'
        results[f"prompt{i}"] = {**{"prompt_text": prompt}, **calculate_scores(prompted_embeddings, labels)}
        i += 1

    os.makedirs(os.path.dirname(options.save_path), exist_ok=True)
    report(f"Saving to {options.save_path}")
    with open(options.save_path, 'w') as f:
        json.dump(results,f, indent=2)



if __name__=="__main__":
    #sanity_check_sorting()
    options = parser.parse_args()
    lang=None
    # if lang is given with column notation
    if ":" in options.data_name:
        options.data_name, lang = options.data_name.split(":")
    corpus, _, labels = download_dataset(options.data_name, lang=lang, split_to_select=options.split, downsample=options.num_examples)
    if options.use_lang_specific_prompts:
        prompts = get_prompts(options.data_name, lang=lang)
    else:
        prompts =  get_prompts(options.data_name)
    report("Sanity check: What was downloaded?")
    report(prompts[0])
    report(corpus[0])
    report(labels[0])
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
        read_embeddings_and_calculate_cluster_scores(options, prompts, corpus, labels)
    else:
        # calculate everything from scratch
        report("Calculating embeddings and score")
        embed_and_calculate_cluster_scores(options, prompts, corpus, labels)
