from sentence_transformers import SentenceTransformer
import datasets
import os
import json
import numpy as np
import torch
import jsonargparse
import sys
import random
from evaluate_prompts import find_relevant_doc_id
from scipy.spatial.distance import pdist, squareform
from utils.dataset_handling import download_dataset
from utils.prompts import get_prompts, get_detailed_instruct
from prompting_metrics import pairwise_distances, knn_overlap_score
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





def create_one_to_one_correspondence(corpus, queries, qrels):
    """
    For each query, find the best match in the corpus.
    Return them in order, and additionally, return the 'unused'/'filler' corpus texts
    """
    # if the dataset is already sorted in this way
    if len(corpus) == len(queries) and qrels == {k:k for k in range(len(queries))}:
        return corpus["text"], queries["text"], None
    questions = []
    targets = []
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
        found_ids.add(most_relevant_id)  # here we could also choose all relevant ids
    unmatched_targets = corpus.filter(lambda example: example["_id"] not in found_ids)
    return  targets, questions, unmatched_targets["text"]




def calculate_metrics_for_clustering(model_name, corpus, labels, prompts, template, k=10, batch_size=8):
    """
    model_name: path or huggingface alias
    corpus: texts to be embedded (with prompts)
    labels: cluster ids for each text
        NOTE: queries and corpus need to have 1 to 1 correspondence, i.e. no qrels here
    prompts: prompts to iterate over
    wrong_answers: "leftovers" from corpus, texts that do not correpond to a query. These
        will be added in the "negative" metrics
    k = number of neighbors considered. for kNN, it is on the query and target side (1 to 1)
        while in the negative metrics, it is on the target side
    """
    # load the model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print("Model loaded.")
    # embed the corpus (here it functions as the vanilla query)
    embeddings_q = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
    
    
    
    # calculate metrics per prompt
    results = {}
    for i, p in enumerate(prompts):
        report("OK SANITY TIME")
        report(f"{embeddings_q.shape=}")
        report(f"{len(labels)=}")
        # apply template and embed the prompt+query
        prompts_and_corpus = [get_detailed_instruct(p, c, template=template) for c in corpus]
        # sanity check printout: see that template is filled correctly
        print(f"Example of what is embedded:\n----\n{prompts_and_corpus[0]}\n----\n")  
        embeddings_pq = model.encode(prompts_and_corpus, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
        # now, we construct the anchor point, which functions as the answer now
        cluster_centers = {}
        for l in np.unique(labels):
            associated_indices = np.where(np.array(labels) == l)[0]
            #print(f"{len(associated_indices)=}")
            embeddings_in_this_cluster = [e for e in embeddings_pq[associated_indices]]
            #print(f"{len(embeddings_in_this_cluster)=}, {embeddings_in_this_cluster[0].shape=}")
            #print(f"Stacked {torch.stack(embeddings_in_this_cluster).shape}")
            cluster_centers[l] = torch.mean(torch.stack(embeddings_in_this_cluster), axis=0)
            #print(f"Shape of cluster centers = {cluster_centers[l].shape}")
        embeddings_a = torch.stack([cluster_centers[l] for l in labels])
        # we will need this later: ids that contain the cluster centers
        # these could also be the last, random, as long as they map each to a different one
        # and with correct label <= this is because the function is index based
        example_cluster_locations = {l:np.where(np.array(labels)==l)[0][0] for l in labels }

        # Now, everything works as before, except the hard negs, 
        # which should be the closest false cluster centers

        # calculate everything we can calculate without using the prompt
        # chord vector from query to answer
        delta_a = embeddings_a - embeddings_q  # (N, D)

        # baseline: how similar are q and a without any prompt?
        sim_qa = cos(embeddings_q, embeddings_a)  # (N,)

        # negatives for metrics 6 & 7: other cluster centers
        N = embeddings_q.shape[0]
        k_hard = min(k, N - 1)
        hard_neg_indices = []
        for i in range(N):
            # for each query, get it's embedding and label
            current_cluster = labels[i]
            current_embedding = embeddings_q[i]
            # find the clusters that the current is NOT in
            other_cluster_embeddings = torch.stack([cluster_centers[l] for l in example_cluster_locations.keys() if l!=current_cluster], axis=0)
            # and their labels
            associated_cluster_labels = np.array([l for l in example_cluster_locations.keys() if l!=current_cluster])
            # find the closest incorrect clusters (and their labels)
            cluster_sims = current_embedding@other_cluster_embeddings.T
            cluster_ids = torch.argsort(cluster_sims, descending=True)[:k_hard]
            asoc_labels = associated_cluster_labels[np.array(cluster_ids.cpu())]
            # then just use the dictionary of locations where this clusters embedding is located
            # does not matter which, because the comparison is simply to the embeddings
            ids_to_mark_as_negatives = [example_cluster_locations[l] for l in asoc_labels]
            hard_neg_indices.append(ids_to_mark_as_negatives)
        
        # Chord vector from query to prompted query ( to compare with delta_a)
        delta_pq = embeddings_pq - embeddings_q  # (N, D)

        # ---- Metric 1: Angulation toward answer ----
        # cosine similarity between the two chord vectors
        # "Does the prompt move the query in the same direction as the answer?"
        chord_sim = cos(delta_a, delta_pq)  # (N,)

        # ---- Metric 2: Direct similarity improvement ----
        # "Does adding the prompt make pq closer to a than q was?"
        sim_pqa = cos(embeddings_pq, embeddings_a)  # (N,)
        sim_improvement = sim_pqa - sim_qa           # (N,)

        # ---- Metric 3: Direct distance ----
        # "How far did the prompt move the query?"
        displacement = torch.norm(delta_pq, dim=1)  # (N,)

        # ---- Metric 4: Parallel vs orthogonal decomposition ----
        # Project delta_pq onto the direction of delta_a
        # out of the total movement caused by the prompt,
        # how much is *toward the answer* vs *sideways*?
        delta_a_norm = delta_a / (torch.norm(delta_a, dim=1, keepdim=True) + 1e-10)
        # 1e-10 here to avoid NaN --> has not other effect since norm is 0 <==> tensor is identically 0
        parallel_magnitude = torch.sum(delta_pq * delta_a_norm, dim=1)      # (N,) signed scalar projection
        orthogonal_magnitude = torch.sqrt(
            torch.clamp(torch.sum(delta_pq ** 2, dim=1) - parallel_magnitude ** 2, min=0.0),
        )  # (N,), follows from pythagorean theorem

        # Ratio: what fraction of the movement is toward the answer?
        parallel_fraction = parallel_magnitude / (displacement + 1e-10)  # (N,), in [-1, 1]

        # sanity check: parallel_fraction and chord_sim should equal (simply from pythagorean theorem)
        #assert torch.allclose(parallel_fraction, chord_sim, atol=1e-4), \
        #    f"Parallel fraction and chord sim do not match: {parallel_fraction} != {chord_sim}"
        # assert removed since result analysis will handle it

        # ---- Metric 5: knn retention ----
        # "Does adding prompt make the query side structure resemble the answer side"
        # knn is NOT applicable to clustering, but let's calculate it anyway
        knn_retention = knn_overlap_score(embeddings_pq.cpu(), embeddings_a.cpu(), k=k)

        # ---- Metric 6: displacement vs. wrong answers ----
        # "Does adding prompt take you away from incorrect answers?"
        # For the k-hardest wrong answers (most similar to the unprompted query),
        # measure how much their similarity changes after prompting.
        # Negative = prompt moved query away from hard negatives (desirable).
        # so, add multiplication by -1 
        # now positive = desirable
        sim_q_all = torch.mm(embeddings_q, embeddings_a.T)
        sim_pq_all = torch.mm(embeddings_pq, embeddings_a.T)
        hard_neg_sim_change = -1*torch.tensor([
            (sim_pq_all[i, hard_neg_indices[i]]
            - sim_q_all[i, hard_neg_indices[i]]).mean().item()
            for i in range(N)
        ])  # (N,)


        # ---- Metric 7: angle between delta_pq and delta_(closest k wrong targets) ----
        # For each query's k-nearest wrong answers, compute the chord vector
        # from the query to that wrong answer, then measure cosine with delta_pq.
        # Positive = prompt pushes toward hard negatives (undesirable).
        # Negative = prompt pushes away from hard negatives (desirable).
        # so again, multiply by -1
        all_embeddings = embeddings_a
        hard_neg_angulation = -1 * torch.tensor([
            cos(
                delta_pq[i].unsqueeze(0).expand(k_hard, -1),    # (k_hard, D)
                all_embeddings[hard_neg_indices[i]] - embeddings_q[i]  # (k_hard, D)
            ).mean().item()
            for i in range(N)
        ])

        def stats(t):
            """Extract summary statistics"""
            if isinstance(t, torch.Tensor):
                arr = t.detach().cpu().numpy().reshape(-1)
            else:
                arr = t
            return {"mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75))}

        results[f"prompt{i}"] = {
            "prompt_text": p if p != "" else "empty",           # prompt text, with "" redirected to "empty"
            "example_text": prompts_and_corpus[0],              # example text as a sanity check
            "chord_similarity":     stats(chord_sim),           # angulation (same as par. fraction)
            "sim_q_a":              stats(sim_qa),              # baseline similarity
            "sim_pq_a":             stats(sim_pqa),             # prompted similarity
            "sim_improvement":      stats(sim_improvement),     # delta
            "displacement":         stats(displacement),        # how far prompt moved q
            "parallel_magnitude":   stats(parallel_magnitude),  # movement toward answer (signed)
            "orthogonal_magnitude": stats(orthogonal_magnitude),# movement sideways
            "parallel_fraction":    stats(parallel_fraction),   # fraction toward answer
            "knn_retention":        stats(knn_retention),       # how much structure we gain
            "hard_neg_sim_change":  stats(hard_neg_sim_change), # sim change to hard negatives
            "hard_neg_angulation":  stats(hard_neg_angulation), # angle toward hard negatives
        }

    return results

    


if __name__=="__main__":
    options = parser.parse_args()
    # dowload dataset and preprocess
    lang = None # initialize
    # clustering/classification does not have queries: we construct separately
    corpus, _, labels = download_dataset(options.data_name, split_to_select=options.split, downsample=options.num_examples)
    prompts = get_prompts(options.data_name)
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

    print(f"Sanity check\n{corpus[0]=}\n{labels[0]}")
    # this time, no need to create correspondence
    results = calculate_metrics_for_clustering(options.model_name, corpus, labels, prompts, options.template, k = options.k, batch_size=options.batch_size)
    
    model_name_safe = options.model_name.replace("/", "__")
    data_safe_name = options.data_name.replace("/","__")
    specific_prompts = "_specific_prompts" if options.use_lang_specific_prompts else ""
    save_path = f"{options.save_prefix}/{model_name_safe}/{data_safe_name}{specific_prompts if lang is not None else ''}/{options.split}/{options.template}_template"
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/prompt_geometry.json', 'w') as f:
        json.dump(results, f, indent=2)