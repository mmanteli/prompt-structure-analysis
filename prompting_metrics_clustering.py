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


def calculate_metrics_for_clustering(model_name, corpus, labels, prompts, template, k=10, batch_size=8):
    """
    model_name: path or huggingface alias
    corpus: texts (all function as both query and corpus in clustering)
    labels: cluster id for each text
    prompts: prompts to iterate over
    template: prompting template
    k: neighborhood size
    """
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print("Model loaded.")

    # For clustering, we are using cluster centers as embeddings_a
    # to avoid recalculation as much as possible, construct label to id mappings
    labels_arr = np.array(labels)
    unique_labels = np.unique(labels_arr)
    N = len(corpus)
    C = len(unique_labels)
    label_to_center_idx = {l: i for i, l in enumerate(unique_labels)}

    # Check that we have at least 2 examples per cluster
    # otherwise, we'd be comparing one point against itself
    cluster_sizes = {l: int(np.sum(labels_arr == l)) for l in unique_labels}
    for l, size_ in cluster_sizes.items():
        assert size_ > 1, f"Cluster {l} has only 1 member — leave-one-out is undefined"

    # Embed unprompted texts: the baseline
    embeddings_q = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)

    # Calculate unprompted cluster centers, keep them as a single matrix
    # and set them also as pairs (like in other tasks)
    # we use these also for hard neg selection for metrics 6 and 7
    baseline_centers = {
        l: torch.mean(embeddings_q[np.where(labels_arr == l)[0]], dim=0)
        for l in unique_labels
    }
    baseline_center_matrix = torch.stack([baseline_centers[l] for l in unique_labels])
    # we can use the id mapping to access this while keeping it as one tensor
    # now create the embeddings_a (fixed anchor) from these: simply assign values so that
    # they correspond to each embeddings_q
    embeddings_a_fixed = torch.stack([baseline_centers[l] for l in labels_arr])

    # for each query, calculate vector to un prompted correct cluster center and similarity to it
    #delta_a_fixed = embeddings_a_fixed - embeddings_q   # we do not need this actually
    sim_qa_fixed  = cos(embeddings_q, embeddings_a_fixed)

    # Okay, but now each text is contained in it's own cluster center
    # introduces bias in metrics other than 6 & 7
    # we need to drop the text itself from the cluster
    # very simply: calculate the mean==center of the whole cluster
    baseline_cluster_sums = {
            l: embeddings_q[np.where(labels_arr == l)[0]].sum(dim=0)
            for l in unique_labels
        }
    # and then substract the individual query embedding and normalize with |cluster|-1 to get the mean 
    # unlike with embeddings_a_fixed, each query will now have different cluster center
    # than others, because it will be omitted from it's own cluster
    # we actually measure if the prompt moved the text towards others in the same cluster
    embeddings_a_fixed_loo = torch.stack([
            (baseline_cluster_sums[labels_arr[i]] - embeddings_q[i])
            / (cluster_sizes[labels_arr[i]] - 1)
            for i in range(N)
        ])
    # again, calculate the similarity
    sim_qa_loo = cos(embeddings_q, embeddings_a_fixed_loo)

    # Select hard negatives for 6 & 7, the most likely clusters to be mistakenly put to
    # "Which other cluster centers are most confusable before prompting?""
    # here we can use the baseline_center_matrix instead of embeddings_a
    # simpler calculation
    k_hard = min(k, C - 1)
    sim_q_baseline = torch.mm(embeddings_q, baseline_center_matrix.T)
    hard_neg_cluster_idxs = []  # these will be used to select the rows to use in 6 &7
    for i in range(N):
        own_center_idx = label_to_center_idx[labels_arr[i]]
        sims_i = sim_q_baseline[i].clone()
        sims_i[own_center_idx] = -float('inf')  # mask own cluster (not a negative)
        top_k = torch.topk(sims_i, k_hard).indices
        hard_neg_cluster_idxs.append(top_k)

    # Now, iterate over prompts and calculate ~fairly~ similarly
    results = {}
    for prompt_num, p in enumerate(prompts):
        prompts_and_corpus = [get_detailed_instruct(p, c, template=template) for c in corpus]
        print(f"Example of what is embedded:\n----\n{prompts_and_corpus[0]}\n----\n")
        embeddings_pq = model.encode(prompts_and_corpus, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)

        # To get the cluster centers AFTER prompt has been added
        # we need to again do leave-one-out
        # because otherwise it is biased towards the correct answer
        # same as above: calculate sums, subtract the text itself, divide by |cluster|-1
        prompted_cluster_sums = {
            l: embeddings_pq[np.where(labels_arr == l)[0]].sum(dim=0)
            for l in unique_labels
        }
        embeddings_a_loo = torch.stack([
            (prompted_cluster_sums[labels_arr[i]] - embeddings_pq[i])
            / (cluster_sizes[labels_arr[i]] - 1)
            for i in range(N)
        ])

        # Chord vector from unprompted to prompted embedding
        delta_pq = embeddings_pq - embeddings_q

        # ---- Metric 1: Angulation toward answer ----
        #  "Does the prompt move the text in the direction of prompted (likely more correct) cluster center?"
        # -> we have to use embeddings_a_loo
        delta_a_loo = embeddings_a_loo - embeddings_q
        chord_sim = cos(delta_a_loo, delta_pq)

        # ---- Metric 2: Direct similarity improvement ----
        # "How much did similarity increase with the cluster center"
        # both query and cluster center move with prompt
        # for both, use loo version to avoid bias
        sim_pqa_loo = cos(embeddings_pq, embeddings_a_loo)
        sim_improvement = sim_pqa_loo - sim_qa_loo 

        # ---- Metric 3: Displacement ----
        # "How much did the prompt move the query"
        # simply just the norm of delta_pq = embeddings_pq - embeddings_q
        # -> severity of the effect the prompt introduces
        displacement = torch.linalg.norm(delta_pq, dim=1)

        # ---- Metric 4: Parallel vs. orthogonal decomposition ----
        # "How much of the movement is towards the 'correct answer'?"
        # we need to use embeddings_a_loo and delta_a_loo
        delta_a_norm = delta_a_loo / (torch.linalg.norm(delta_a_loo, dim=1, keepdim=True) + 1e-10)
        parallel_magnitude = torch.sum(delta_pq * delta_a_norm, dim=1)
        orthogonal_magnitude = torch.sqrt(
            torch.clamp(torch.sum(delta_pq ** 2, dim=1) - parallel_magnitude ** 2, min=0.0)
        ) 
        parallel_fraction = parallel_magnitude / (displacement + 1e-10)

        # ---- Metric 5: kNN retention ----
        # "Does prompting preserve the unprompted neighborhood structure?"
        # this measures a different thing than for retrieval, but as there is no way to apply this in other ways
        # this it the most natural extra question to ask
        knn_retention = knn_overlap_score(
            embeddings_pq.detach().cpu(),
            embeddings_q.detach().cpu(),
            k=min(k, N - 1)
        )

        # ---- Metric 6: similarity change to hard negative clusters ----
        # Here we use the fixed centers
        # change reflects purely how the text moved, not how the centers moved.
        sim_pq_baseline = torch.mm(embeddings_pq, baseline_center_matrix.T)
        hard_neg_sim_change = -1 * torch.tensor([
            (sim_pq_baseline[i, hard_neg_cluster_idxs[i]]
             - sim_q_baseline[i, hard_neg_cluster_idxs[i]]).mean().item()
            for i in range(N)
        ])

        # ---- Metric 7: angle toward hard negative cluster centers ----
        # Similarly here: use the fixed centers
        hard_neg_angulation = -1 * torch.tensor([
            cos(
                delta_pq[i].unsqueeze(0).expand(k_hard, -1),                        
                baseline_center_matrix[hard_neg_cluster_idxs[i]] - embeddings_q[i]  
            ).mean().item()
            for i in range(N)
        ])


        # NOTE: sim_qa_fixed and sim_qa_loo are computed from unprompted embeddings only
        # identical always, but results are easier to read if we include them every time
        results[f"prompt{prompt_num}"] = {
            "prompt_text":          p if p != "" else "empty",
            "example_text":         prompts_and_corpus[0],
            "chord_similarity":     stats(chord_sim),
            "sim_q_a_fixed":        stats(sim_qa_fixed),
            "sim_q_a_loo":          stats(sim_qa_loo),
            "sim_pq_a":             stats(sim_pqa_loo),
            "sim_improvement":      stats(sim_improvement),
            "displacement":         stats(displacement),
            "parallel_magnitude":   stats(parallel_magnitude),
            "orthogonal_magnitude": stats(orthogonal_magnitude),
            "parallel_fraction":    stats(parallel_fraction),
            "knn_retention":        stats(knn_retention),
            "hard_neg_sim_change":  stats(hard_neg_sim_change),
            "hard_neg_angulation":  stats(hard_neg_angulation),
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
    # all texts are queries, and cluster centers are answers
    results = calculate_metrics_for_clustering(options.model_name, corpus, labels, prompts, options.template, k = options.k, batch_size=options.batch_size)
    
    model_name_safe = options.model_name.replace("/", "__")
    data_safe_name = options.data_name.replace("/","__")
    specific_prompts = "_specific_prompts" if options.use_lang_specific_prompts else ""
    save_path = f"{options.save_prefix}/{model_name_safe}/{data_safe_name}{specific_prompts if lang is not None else ''}/{options.split}/{options.template}_template"
    print(f"Saving to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/prompt_geometry.json', 'w') as f:
        json.dump(results, f, indent=2)