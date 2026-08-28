from sentence_transformers import SentenceTransformer
import datasets
import os
import json
import numpy as np
import torch
import jsonargparse
import sys
import pickle
import random
from evaluate_prompts import find_relevant_doc_id, write_embeddings
from scipy.spatial.distance import pdist, squareform
from utils.dataset_handling import download_dataset
from utils.prompts import get_prompts, get_detailed_instruct
from prompting_metrics import pairwise_distances, knn_overlap_score, stats
# this contains simply lists and dictionaries that help select the correct prompts

cos = torch.nn.CosineSimilarity()
# set random behaviour for replication
seed = 42
random.seed(seed)
np.random.seed(seed)

parser = jsonargparse.ArgumentParser(prog="Measure structural changes in prompting")
#parser.add_argument('--config', action=ActionConfigFile)
parser.add_argument('--model_name', '--model', type=str|list,
                    help="HF-alias or path to downloaded model, can be a list.")
parser.add_argument('--data_name', '--dataset', type=str|list,
                    help="HF-alias or path to downloaded dataset, can be a list.")
parser.add_argument('--split', type=str, default="fit",
                    help="Which split to select from dataset.")
parser.add_argument('--template', type=str, default="Instruct-Query", choices=["Instruct-Query", "simple"],
                    help="Which prompting template to use")
parser.add_argument('--use_lang_specific_prompts', action='store_true',
                    help="Use prompts that specifically mention the target language, only for multilingual datasets.")
parser.add_argument('--k', type=int, default=10,
                    help="which neighborhood size to use (for knn_ret and false_negs)")
parser.add_argument('--batch_size', type=int, default=16,
                    help="batch size for embedding")
parser.add_argument('--num_examples', type=int|bool, default=5000,
                    help="For largest datasets, number of examples to downsample to, set to False for no downsampling")
parser.add_argument('--embedding_prefix', type=str|bool, default=False,
                    help="prefix to save embedings to, works similar to --save_prefix")
parser.add_argument('--save_prefix', type=str, default="results_metrics",
                    help="Saving path; model_name, data_name, prompt_type and k added in script")


def report(msg):
    # for quick flushing
    print(f'{msg}', flush=True)

def yield_from_pkl(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
# EXTREMELY INEFFICIENT; ONLY USE THIS IN TIME OF EXTREME NEED
def read_from_pickle(filename, key_to_find):
    for line_ in yield_from_pkl(filename):
        if line_["key"] == key_to_find:
            return line_
    raise Exception(f"Could not locate the precalculated stuff with given key {key_to_find}")
             
    

def create_STS_correspondence(corpus, queries, qrels, minmax=None):
    """
    For each query, find the best match in the corpus.
    Return them in order, and additionally, return the 'unused'/'filler' corpus texts
    minmax=(low, high) for range of scores, otherwise calculated from data
    """
    # check if we already have the structure we need, if so, only normalize the scores==qrels
    # the download_dataset() can return these as already prepocessed
    if isinstance(qrels, list) and isinstance(corpus, list) and isinstance(queries, list) and len(corpus) == len(queries) == len(qrels):
        if minmax is None:
            min_score, max_score = np.min(qrels), np.max(qrels)
        else:
            min_score, max_score = minmax
        # normalize scores
        scores = np.array([2*(s-min_score)/(max_score-min_score)-1 for s in qrels])
        return corpus, queries, scores
    # otherwise, construct them
    # here using question/target simply not to cofuse with the corpus and query variable names
    questions = []
    targets = []
    scores = []
    found_ids = set()
    # loop over queries
    for line in queries:
        q_id, q_text = line["_id"], line["text"]
        #print(f"Now in {q_id=}, {q_text=}")
        # find the relevant answers based on the query id
        # this funtion returns the best match at the top of the list
        relevant_ids, assoc_scores = find_relevant_doc_id(q_id, qrels)
        #print(f"{relevant_ids=}, {assoc_scores=}")
        most_relevant_id = relevant_ids[0]
        found = [l for l in corpus if l["_id"] == most_relevant_id]
        assert len(found) == 1, f"Duplicate ids in corpus, {most_relevant_id=} resulted in {found=}"
        c_id, c_text = found[0]["_id"], found[0]["text"]
        questions.append(q_text)
        targets.append(c_text)
        scores.append(assoc_scores[0])
        found_ids.add(most_relevant_id)  # here we could also choose all relevant ids
    unmatched_targets = corpus.filter(lambda example: example["_id"] not in found_ids)
    # create a mask from the scores
    # if the result is relevant, we multiply all scores with 1
    # if non-relevant, we multiply with -1 
    # i.e. minmax normalize scores to -1 to 1
    if minmax is None:
        min_score, max_score = np.min(scores), np.max(scores)
    else:
        min_score, max_score = minmax
    scores = np.array([2*(s-min_score)/(max_score-min_score)-1 for s in scores])
    # NOTE: this could also be a -1/1 mask based on the mean score..?
    assert len(unmatched_targets["text"]) == 0, "We found unmatched targets for STS/Pair classification, should not be possible"
    return  targets, questions, scores


def calculate_metrics(model_name, queries, answers, scores, prompts, template, k=10, batch_size=8, embeddings=None):
    """
    model_name: path or huggingface alias
    queries: datasets-object with columns "_id" and "text"
    anwers: datasets-object with columns "_id" and "text"
        NOTE: queries and corpus need to have 1 to 1 correspondence, i.e. no qrels here
    prompts: prompts to iterate over
    k = number of neighbors considered. for kNN, it is on the query and target side (1 to 1)
        while in the negative metrics, it is on the target side
    """

    # load the model or precalculated results
    calculate_embeddings = True
    if embeddings is not None and os.path.exists(embeddings):
        calculate_embeddings = False # we already have results
    # load the model or read precalculated
    if calculate_embeddings:  
        model = SentenceTransformer(model_name, trust_remote_code=True)
        print("Model loaded.")
        # embed the "ground truth values": regular queries and targets
        embeddings_q = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
        embeddings_a = model.encode(answers, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
        if embeddings:  # now we can still save the embeddings if we want to
            report(f"Writing embeddings to {embeddings}")
            write_embeddings(file=embeddings, key="corpus", data={"text":answers}, embeddings=embeddings_a)
            write_embeddings(file=embeddings, key="queries", data={"text":queries}, embeddings=embeddings_q)
    else:  # we read embeddings, not calculate them
        print("Reading precalculated embeddings")
        prec = read_from_pickle(embeddings, "corpus")
        assert prec["data"]["text"][0] == answers[0], f'{prec["data"]["text"][0]}!={answers[0]}'# here sanity check
        embeddings_a = prec["data"]["embeddings"]
        prec = read_from_pickle(embeddings, "queries")
        embeddings_q = prec["data"]["embeddings"]


    # calculate everything we can calculate without using the prompt
    # chord vector from query to answer
    #delta_a = embeddings_a - embeddings_q   # not needed here: we add prompt to answer side as well

    # baseline: how similar are q and a without any prompt?
    sim_qa = cos(embeddings_q, embeddings_a)

    # negatives for metrics 6 & 7: here we can add the wrong answers (if they exist)
    N_pool = embeddings_a.shape[0]
    N_q = embeddings_q.shape[0]
    k_hard = min(k, N_pool - 1)
    # add the filler
    sim_q_all = torch.mm(embeddings_q, embeddings_a.T)
    # For each query, find indices of k nearest *wrong* answers
    # we can still just mask the "diagonal" (i)
    # but then just search the larger area in torch.topk
    hard_neg_indices = []
    for i in range(N_q):
        sims_i = sim_q_all[i].clone()
        # mask out the correct pair
        sims_i[i] = -float('inf')
        hard_neg_indices.append(torch.topk(sims_i, k_hard).indices)
    
    # calculate metrics per prompt
    results = {}
    for prompt_num, p in enumerate(prompts):
        # apply template and embed the prompt+query
        prompts_and_queries = [get_detailed_instruct(p, q, template=template) for q in queries]
        prompts_and_answers = [get_detailed_instruct(p, a, template=template) for a in answers]
        # sanity check printout: see that template is filled correctly
        print(f"Example of what is embedded:\n----\n{prompts_and_queries[0]}\n----\n")  
        if calculate_embeddings:
            embeddings_pq = model.encode(prompts_and_queries, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
            embeddings_pa = model.encode(prompts_and_answers, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
            if embeddings:
                write_embeddings(file=embeddings, key=p, data={"text": prompts_and_queries}, embeddings=embeddings_pq)
                write_embeddings(file=embeddings, key="answers:"+p, data={"text": prompts_and_answers}, embeddings=embeddings_pa)
        else:  # just read
            prec = read_from_pickle(embeddings, p)
            assert prec["data"]["text"] == prompts_and_queries, "mismatch between precalculated and dataset"
            embeddings_pq = prec["data"]["embeddings"]
            prec = read_from_pickle(embeddings, "answers:"+p)
            embeddings_pa = prec["data"]["embeddings"]

        # Chord vector from query to prompted query
        delta_pq = embeddings_pq - embeddings_q 
        delta_pa = embeddings_pa - embeddings_a

        # ---- Metric 1: Angulation toward answer ----
        # cosine similarity between the two chord vectors
        # "Does the prompt move the query in the same direction as the answer?"
        chord_sim = cos(delta_pa, delta_pq) 

        # ---- Metric 2: Direct similarity improvement ----
        # "Does adding the prompt make pq closer to pa than q was to a?"
        sim_pqa = cos(embeddings_pq, embeddings_pa)
        sim_improvement = sim_pqa - sim_qa 

        # ---- Metric 3: Direct distance ----
        # "How far did the prompt move the query?"
        displacement = torch.linalg.norm(delta_pq, dim=1)

        # ---- Metric 4: Parallel vs orthogonal decomposition ----
        # Project delta_pq onto the direction of delta_pa
        # out of the total movement caused by the prompt,
        # how much is *toward the answer* vs *sideways*?
        delta_pa_norm = delta_pa / (torch.linalg.norm(delta_pa, dim=1, keepdim=True) + 1e-10)
        # 1e-10 here to avoid NaN --> has not other effect since norm is 0 <==> tensor is identically 0
        parallel_magnitude = torch.sum(delta_pq * delta_pa_norm, dim=1)  #signed scalar projection
        orthogonal_magnitude = torch.sqrt(
            torch.clamp(torch.sum(delta_pq ** 2, dim=1) - parallel_magnitude ** 2, min=0.0),
        )  # follows from pythagorean theorem

        # Ratio: what fraction of the movement is toward the answer?
        parallel_fraction = parallel_magnitude / (displacement + 1e-10)

        # sanity check: parallel_fraction and chord_sim should equal (simply from pythagorean theorem)
        #assert torch.allclose(parallel_fraction, chord_sim, atol=1e-4), \
        #    f"Parallel fraction and chord sim do not match: {parallel_fraction} != {chord_sim}"
        # assert removed since result analysis will handle it

        # ---- Metric 5: knn retention ----
        # "Does adding prompt make the query side structure resemble the answer side"
        # not directly applicable for STS: select only the positive pairs
        positive_mask = scores > 0
        knn_retention = knn_overlap_score(embeddings_pq[positive_mask].detach().cpu(), 
                                          embeddings_pa[positive_mask].detach().cpu(), 
                                          k=min(k, positive_mask.sum() - 1)
                                          )


        # Here we refer to the past closest false neighbors
        # hence the indices we calculated before, but we use the new embeddings

        # ---- Metric 6: displacement vs. wrong answers ----
        # "Does adding prompt take you away from incorrect answers?"
        # For the k-hardest wrong answers (most similar to the unprompted query),
        # measure how much their similarity changes after prompting.
        # Negative = prompt moved query away from hard negatives (desirable).
        # so, add multiplication by -1 
        # now positive = desirable
        sim_pq_all = torch.mm(embeddings_pq, embeddings_pa.T)
        hard_neg_sim_change = -1*torch.tensor([
            (sim_pq_all[i, hard_neg_indices[i]]
            - sim_q_all[i, hard_neg_indices[i]]).mean().item()
            for i in range(N_q)
        ]) 


        # ---- Metric 7: angle between delta_pq and delta_(closest k wrong targets) ----
        # For each query's k-nearest wrong answers, compute the chord vector
        # from the query to that wrong answer, then measure cosine with delta_pq.
        # Positive = prompt pushes toward hard negatives (undesirable).
        # Negative = prompt pushes away from hard negatives (desirable).
        # so again, multiply by -1
        all_embeddings = embeddings_pa
        hard_neg_angulation = -1 * torch.tensor([
            cos(
                delta_pq[i].unsqueeze(0).expand(k_hard, -1),
                all_embeddings[hard_neg_indices[i]] - embeddings_q[i]
            ).mean().item()
            for i in range(N_q)
        ])



        # finally, the difference based on retrieval: weight the results with score
        # e.g. we saw that similarity decreased with semantically non-related
        # then that is good! so we multiply by a negative number
        # works similartly for the angles (bad angle that should be bad == good angle)
        # with some, this does not work
        # for example displacement is always >0
        # same for knn and the neg metrics which are not really applicable here
        # hence we took the measures that were described in the comments
        results[f"prompt{prompt_num}"] = {
            "prompt_text": p if p != "" else "empty",           # prompt text, with "" redirected to "empty"
            "example_text": prompts_and_queries[0],             # example text as a sanity check
            "chord_similarity":     stats(np.array(chord_sim.cpu())*scores),           # angulation (same as par. fraction)
            "sim_q_a":              stats(np.array(sim_qa.cpu())*scores),              # baseline similarity
            "sim_pq_a":             stats(np.array(sim_pqa.cpu())*scores),             # prompted similarity
            "sim_improvement":      stats(np.array(sim_improvement.cpu())*scores),     # delta
            "displacement":         stats(np.array(displacement.cpu())),                # how far prompt moved q
            "parallel_magnitude":   stats(np.array(parallel_magnitude.cpu())*scores),  # movement toward answer (signed)
            "orthogonal_magnitude": stats(np.array(orthogonal_magnitude.cpu())),        # movement sideways
            "parallel_fraction":    stats(np.array(parallel_fraction.cpu())*scores),   # fraction toward answer
            "knn_retention":        stats(np.array(knn_retention)),                      # how much structure we gain
            "hard_neg_sim_change":  stats(np.array(hard_neg_sim_change.cpu())),     # sim change to hard negatives
            "hard_neg_angulation":  stats(np.array(hard_neg_angulation.cpu())),     # angle toward hard negatives
        }

    return results

    


if __name__=="__main__":
    report("This script is specifically for STS: scores are multiplied with 1/-1 based on STS score")
    options = parser.parse_args()
    # dowload dataset and preprocess
    lang = None # initialize
    # if lang is given with column notation
    if ":" in options.data_name:
        options.data_name, lang = options.data_name.split(":")
    # download the dataset with data_name
    # this returns queries and corpus, both datasets.Dataset, and 
    # qrels which may be a datasets.Dataset or a dict{query_id:corpus_is}
    # and specifically for STS datasets that sometimes do not have natural (corpus,query,qrel) structure in hfhub
    # they can also be lists in some cases
    # qrels in the case of STS datasets is the scores (e.g. 5 for semantically similar, 1 for not at all)
    corpus, queries, qrels = download_dataset(options.data_name, lang=lang, split_to_select=options.split)
    # download prompts
    # this returns a list of possible instructions to use on the query side
    if options.use_lang_specific_prompts:
        assert lang is not None, "Give language for language specific prompts"
        prompts = get_prompts(options.data_name, lang=lang)
    else:
        prompts =  get_prompts(options.data_name)
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

    # preprocess the data
    # create 1-to-1 correspondence (STS datasets already have this)
    # and most importantly: normalize scores to -1 to 1
    targets, questions, scores = create_STS_correspondence(corpus, queries, qrels)
    # filler targets is in almost ALL cases None here
    print(f"Sanity check:  These should have score {scores[0]}:\n{questions[0]=}\n{targets[0]}")

    # save the results
    model_safe_name = options.model_name.replace("/", "__")
    data_safe_name = options.data_name.replace("/","__")
    if lang is not None:
        data_safe_name += f":{lang}"
    specific_prompts = "_lang_specific" if options.use_lang_specific_prompts else ""
    save_path = f"{options.save_prefix}/{model_safe_name}/{data_safe_name}{specific_prompts if lang is not None else ''}/{options.split}/{options.template}_template"
    embeddings_path = None
    if options.embedding_prefix:
        embeddings_path = f"{options.embedding_prefix}/{model_safe_name}/{data_safe_name}{specific_prompts if lang is not None else ''}/{options.split}/{options.template}_embeddings.pkl"
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    # calculate results
    results = calculate_metrics(options.model_name, questions, targets, scores, prompts, options.template, k = options.k, batch_size=options.batch_size, embeddings=embeddings_path)
    
    os.makedirs(save_path, exist_ok=True)
    report(f"Saving to {save_path}")
    with open(f'{save_path}/prompt_geometry.json', 'w') as f:
        json.dump(results, f, indent=2)