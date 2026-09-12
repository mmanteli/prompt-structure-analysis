from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np
import torch
import jsonargparse
import random
from scipy.spatial.distance import pdist, squareform
from utils.dataset_handling import download_dataset
from utils.prompts import get_prompts, get_detailed_instruct

cos = torch.nn.CosineSimilarity()
# set random behaviour for replication
seed = 42
random.seed(seed)
np.random.seed(seed)

def report(msg):
    print(msg, flush=True)

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
parser.add_argument('--nn', type=int, default=10,
                    help="which neighborhood size to use (for knn_ret)")
parser.add_argument('--batch_size', type=int, default=16,
                    help="batch size for embedding")
parser.add_argument('--num_examples', type=int|bool, default=5000,
                    help="For largest datasets, number of examples to downsample to, set to False for no downsampling")
parser.add_argument('--embedding_prefix', type=str|bool, default=False,
                    help="prefix to save embedings to, works similar to --save_prefix")
parser.add_argument('--save_prefix', type=str, default="results_metrics",
                    help="Saving path; model_name, data_name, prompt_type and k added in script")

def _as_2d_float(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")
    return a

def pairwise_distances(X, metric="cosine"):
    """Make a pairwise distance matrix for input X"""
    X = _as_2d_float(X)
    return squareform(pdist(X, metric=metric))

def knn_indices_from_distance_matrix(D, k:int):
    """
    Return indices of k nearest neighbors for each row i, excluding self.
    D: NxN pairwise distances.
    Output: Nxk integer array of indices.
    """
    D = np.asarray(D)
    N = D.shape[0]
    if D.shape != (N, N):
        raise ValueError("D must be sqaure")
    if not (1 <= k <= N - 1):
        raise ValueError(f"k value must be in [1, {N-1}], got {k}")
    idx = np.argsort(D, axis=1)
    # exclude self -> d(self, self) == 0, always the first index
    idx = idx[:, 1:k+1]
    return idx

def _jaccard_overlap(a, b):
    """Jaccard overlap for two 1D integer arrays (as sets)."""
    sa = set(map(int, a))
    sb = set(map(int, b))
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0  # return 1 for no sets -> will not happen if k>0


def knn_overlap_score(X, Y, k=10, metric="cosine", mode= "jaccard"):
    """
    Calculate the average neighborhood preservation score, 0 to 1
      - "recall": |N_k^X(i) ∩ N_k^Y(i)| / k
      - "jaccard": Jaccard(N_k^X(i), N_k^Y(i))
    This function does not return the mean but individual values.
    """
    assert mode in ["recall", "jaccard"], f"Mode needs to be recall or jaccard, now {mode}"
    assert X.shape[0] == Y.shape[0], f"X and Y must have the same number of instances, X.shape={X.shape}, Y.shape={Y.shape}"

    DX = pairwise_distances(X, metric=metric)
    DY = pairwise_distances(Y, metric=metric)

    NX = knn_indices_from_distance_matrix(DX, k)
    NY = knn_indices_from_distance_matrix(DY, k)

    if mode == "recall":
        scores = []
        for i in range(X.shape[0]):
            scores.append(len(set(NX[i]) & set(NY[i])) / k)
        return scores   # MEAN removed here, see stats()

    else:
        return [_jaccard_overlap(NX[i], NY[i]) for i in range(X.shape[0])]   # MEAN removed here, see stats()


def download_squad_paraphrases(split_to_select="dev", **kwargs):
    report("Downloading SQuAD from local")
    assert split_to_select in ["train", "dev", "test"], "--split given incorrectly to squad"
    with open(f"/scratch/project_462001491/jmnybl/squad_v1.1/train-splits/train-{split_to_select}.json") as file:
        data = json.load(file)
    return data, "question"

def download_arcchallenge_paraphrases(**kwargs):
    report("Dowloading ARCChallenge from local")
    data = []
    with open(f"/scratch/project_462001491/jmnybl/arcchallenge-mteb/arcchallenge_test_queries.json") as file:
        data = json.load(file)
    return data, "query"

def download_tatoeba_paraphrases(lang = None, **kwargs):
    assert lang is not None, "Give language to go with Tatoeba"
    report("Dowloading Tatoeba paraphrases from local")
    with open(f"/scratch/project_462001491/jmnybl/tatoeba-mteb/tatoeba_test_{lang}.json") as file:
        data = json.load(file)
    return data, "query"


def read_distractors(data_name, **kwargs):
    if "tatoeba" in data_name.lower():
        return download_tatoeba_paraphrases(**kwargs)
    if "arcchallenge" in data_name.lower():
        return download_arcchallenge_paraphrases(**kwargs)
    if "squad" in data_name.lower():
        return download_squad_paraphrases(**kwargs)
    return None, None  # for datasets without



def find_relevant_doc_id(query_id, qrels):
    if isinstance(qrels, dict):
        return ([qrels[query_id]], [1]) if query_id in qrels else ([], [])
    indices_of_query_ids = np.where(np.array(qrels["query_id"]) == query_id)[0]
    associated_corpus_values = np.array(qrels["corpus_id"])[indices_of_query_ids]
    associated_corpus_scores = np.array(qrels["score"])[indices_of_query_ids]
    # sort these to have the best match at the top
    indices_that_sort = np.argsort(associated_corpus_scores)[::-1]
    return (associated_corpus_values[indices_that_sort].tolist(), associated_corpus_scores[indices_that_sort].tolist())

def create_one_to_one_correspondence(corpus, queries, qrels):
    """
    For each query, find the best match in the corpus.
    Return them in order, and additionally, return the 'unused'/'filler' corpus texts
    """
    # if the dataset is already sorted in this way (some dataset download scripts return this format)
    if len(corpus) == len(queries) and (qrels == {f"q{k}":f"c{k}" for k in range(len(queries))} or qrels == {k:k for k in range(len(queries))}):
        return corpus["text"], queries["text"], None, None
    questions = []
    targets = []
    found_ids = set()
    # loop over queries
    for line in queries:
        q_id, q_text = line["_id"], line["text"]
        # find the relevant answers based on the query id
        # this funtion returns the best match at the top of the list
        relevant_ids, assoc_scores = find_relevant_doc_id(q_id, qrels)
        #print(f"{relevant_ids=}, {assoc_scores=}")
        most_relevant_id = relevant_ids[0]
        found = [l for l in corpus if l["_id"] == most_relevant_id]
        assert len(found) == 1, f"Duplicate ids OR no match in corpus, {most_relevant_id=} resulted in {found=}"
        c_id, c_text = found[0]["_id"], found[0]["text"]
        questions.append(q_text)
        targets.append(c_text)
        found_ids.add(most_relevant_id)  # here we could also choose all relevant ids
    filler_targets = corpus.filter(lambda example: example["_id"] not in found_ids)["text"]
    # check for overlap: sometimes there may be docs that are identical but with different ids
    overlap = set(targets)&set(filler_targets)
    filler_targets = [f for f in filler_targets if f not in overlap]
    # find the texts with the same exact answer: masked in hard_negatives
    unique_text_ids = np.unique(targets)    # unique answers
    same_answer_id_dict = {k: np.where(np.array(targets) == k)[0] for k in unique_text_ids} # [0] to remove tuple
    same_answer_id_list_per_targets = [same_answer_id_dict[t] for t in targets] # which ids share the same answer
    # ok now just rmove the index itself (we do not want to match to self)
    same_answer_id_list_per_question = [[d for d in same_answer_id_list_per_targets[i] if d!=i] for i in range(len(questions))]
    # ^^ these are ids of the same answers: use them to mask the other answers when we look for closest false pos!
    return targets, questions, filler_targets, same_answer_id_list_per_question



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



def calculate_metrics(model_name, prompts, template, queries, answers, distractors, wrong_answers, k=10, batch_size=8, same_answers_mask=None):
    """
    Calculate retrieval metrics.
    model_name: path or huggingface alias
    prompts: prompts to iterate over
    template: "Instruct-Query" or "simple", tells get_detailed_instruct() what to do
    queries: preprocessed queries
    anwers: proprocessed corpus
        NOTE: queries and corpus need to have 1 to 1 correspondence, i.e. no qrels here
    distractors: for each query, a list of texts that are used to distract retrieval
    wrong_answers: "leftovers" from corpus, texts that do not correpond to a query.
    k = number of neighbors considered knn
    batch_size= batch size for embedding
    """
    report("Downloading model...")
    model = SentenceTransformer(model_name, trust_remote_code=True).to('cuda')
    report("Model loaded.")
    # embed the "ground truth values": regular queries and targets
    embeddings_q = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size).float()
    embeddings_a = model.encode(answers, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size).float()
    if len(distractors)>0:
        embeddings_d = model.encode(distractors, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size).float()
    # if the dataset has filler/wrong answers, answers with no question that answers then, embed them as well
    # and create a "all" embeddings variable: the ids still match to queries bc we only append
    # we use embeddings_a_all in any calculation where it is not strictly necessary to have 1-to-1 QA pairs
    if wrong_answers:
        # yes filler, so concatenate them at the end
        embeddings_wa = model.encode(wrong_answers, convert_to_tensor=True,
                                     normalize_embeddings=True, batch_size=batch_size).float()
        embeddings_all_a = torch.concat((embeddings_a, embeddings_wa))
    else:
        # no filler
        embeddings_all_a = embeddings_a

    # calculate everything we can calculate without using the prompt
    # chord vector from query to answer
    delta_q2a = embeddings_a - embeddings_q

    # baseline similarity: how similar are q and a without any prompt?
    sim_qa = cos(embeddings_q, embeddings_a)  # this is a vector(len(queries))
    # same for euclidean
    sim_qa_euc = torch.linalg.norm(delta_q2a, dim=1)

    # hard negatives (most likely false positives) here we can use the "wrong" answers (if they exist)
    N_pool = embeddings_all_a.shape[0]
    N_q = embeddings_q.shape[0]
    k_hard = 1 #min(k, N_pool - 1)  # opted to use 1 always
    # add the filler
    sim_q_all = torch.mm(embeddings_q, embeddings_all_a.T)  # here we use embeddings_all_a, hence no cos()
    # For each query, find indices of k nearest *wrong* answers
    # since embeddings_q and embeddings_a are in order, and we just append embeddings_wa
    # we can still just mask the "diagonal" (i) (correct match)
    # but then just search the larger area in torch.topk
    hard_neg_indices = []
    for i in range(N_q):
        sims_i = sim_q_all[i].clone()
        # mask out the correct pair (i == i)
        sims_i[i] = -float('inf')
        # mask out duplicates
        if same_answers_mask and same_answer_mask[i]: # first checks if the variable is none, second if for current i we have a mask
            sims_i[same_answers_mask[i]] = -float('inf')
        hard_neg_indices.append(torch.topk(sims_i, k_hard).indices)

    results = {}
    for prompt_num, p in enumerate(prompts):
        # apply template and embed the prompt+query
        prompts_and_queries = [get_detailed_instruct(p, q, template=template) for q in queries]
        # sanity check printout: see that template is filled correctly
        report(f"Sanity: Example of what is embedded:\n----\n{prompts_and_queries[0]}\n----\n")
        embeddings_pq = model.encode(prompts_and_queries, convert_to_tensor=True,
                                        normalize_embeddings=True, batch_size=batch_size).float()

        # Chord vector from query to prompted query ( to compare with delta_a)
        # delta_q2a = embeddings_a - embeddings_q
        delta_q2pq = embeddings_pq - embeddings_q
        delta_pq2a = embeddings_a - embeddings_pq
        # delta_q2a <= delta_q2pq + delta_pq2a

        # Metric 1: similarities in all directions:
        # sim_qa and sim_qa_euc already calculated
        # now for the prompt
        # sim_qa = cos(embeddings_q, embeddings_a)
        sim_pqa = cos(embeddings_pq, embeddings_a)
        sim_pqq = cos(embeddings_pq, embeddings_q)
        # sim_qa_euc = torch.linalg.norm(delta_q2a, dim=1)
        sim_pqa_euc = torch.linalg.norm(delta_pq2a, dim=1)
        sim_pqq_euc = torch.linalg.norm(delta_q2pq, dim=1)
        naive_sim_improvement = sim_pqa - sim_qa  # how much did the query increase similarity

        # metric 2: angulation to answer
        # cosine similarity between the chord
        # "Does the prompt move the query in the same direction as the answer?"
        chord_sim = cos(delta_q2a, delta_q2pq)   # close to 1 = yes, close to -1 = no, close to 0 = orthogonal

        # metric 3: knn retention
        # this is not perfectly applicable because of duplicates
        # measures if the paraphrases are all close together ->
        # the answers are the same embedding so they definitely are
        knn_retention = knn_overlap_score(embeddings_pq.detach().cpu(), embeddings_a.detach().cpu(), k=k)

        # similarity: prompted query to all answers
        sim_pq_all = torch.mm(embeddings_pq, embeddings_all_a.T)

        # Metric 4: similarity change from hard negative answers
        # this time we do not need to mask the paraphrases
        # as the indices are preclaculated
        # multiply by -1 because if the sim increased (>0)
        # that's bad, we want it the other way round
        # max() over the sim changes
        #print(f"{(sim_pq_all[i, hard_neg_indices[i]] - sim_q_all[i, hard_neg_indices[i]]).shape=}")
        hard_neg_sim_change_max = -1*torch.tensor([
            (sim_pq_all[i, hard_neg_indices[i]]
            - sim_q_all[i, hard_neg_indices[i]]).max().item()
            for i in range(N_q)
        ])
        #report(f"{hard_neg_sim_change_max.shape=}")
        # mean() over sim changes
        hard_neg_sim_change_mean = -1*torch.tensor([
            (sim_pq_all[i, hard_neg_indices[i]]
            - sim_q_all[i, hard_neg_indices[i]]).mean().item()
            for i in range(N_q)
        ])

        # metric 5: the same as above, but angles
        hard_neg_angulation_max = -1 * torch.tensor([
            cos(
                delta_q2pq[i].unsqueeze(0).expand(k_hard, -1),    # expand duplicates this k_hard times
                embeddings_all_a[hard_neg_indices[i]] - embeddings_q[i],  # -> k_hard vectors beween q[i] and a[i]
            ).max().item()
            for i in range(N_q)
        ])

        hard_neg_angulation_mean = -1 * torch.tensor([
            cos(
                delta_q2pq[i].unsqueeze(0).expand(k_hard, -1),    # same here
                embeddings_all_a[hard_neg_indices[i]] - embeddings_q[i],  # and here
            ).mean().item()
            for i in range(N_q)
        ])


        if len(distractors)>0:
            # metric 6: displacement from Q* = paraphrases
            # same as 4, but we compare to query side with distractor indices
            # these are already in order, so no need to have different indices, just diag-diag
            sim_q_with_distractors= torch.mm(embeddings_q, embeddings_d.T)
            sim_pq_with_distractors = torch.mm(embeddings_pq, embeddings_d.T)
            paraphrase_neg_sim_change_max = -1*torch.tensor([
                (sim_pq_with_distractors[i, i] - sim_q_with_distractors[i, i]).max().item()
                for i in range(N_q)
            ])

            paraphrase_neg_sim_change_mean = -1*torch.tensor([
                (sim_pq_with_distractors[i,i] - sim_q_with_distractors[i,i]).mean().item()
                for i in range(N_q)
            ])
        else:
            paraphrase_neg_sim_change_mean, paraphrase_neg_sim_change_max = [-1], [-1]

        results[f"prompt{prompt_num}"] = {
            "prompt_text": p if p != "" else "empty",       # prompt text, with "" redirected to "empty"
            "example_text": prompts_and_queries[0],         # example text as a sanity check
            "sim_q2a":                  stats(sim_qa),      # baseline similarity
            "sim_q2pq":                 stats(sim_pqq),         # similarity between query and prompted query
            "sim_pq2a":                 stats(sim_pqa),         # prompted similarity
            "sim_q2a_euc":              stats(sim_qa_euc),      # same but euclidean
            "sim_q2pq_euc":             stats(sim_pqq_euc),     # 
            "sim_pq2a_euc":             stats(sim_pqa_euc),     #
            "chord_similarity":         stats(chord_sim),                   # angulation toward answer
            "sim_improvement":          stats(naive_sim_improvement),       # how much did we move toward answer
            "knn_retention":            stats(knn_retention),               # how much structure we gain
            "hard_neg_sim_change_mean": stats(hard_neg_sim_change_mean),    # sim change to hard negatives
            "hard_neg_sim_change_max":  stats(hard_neg_sim_change_max),     # sim change to hard negatives
            "hard_neg_angulation_mean": stats(hard_neg_angulation_mean),     # angle toward hard negatives
            "hard_neg_angulation_max":  stats(hard_neg_angulation_max),     # angle toward hard negatives
            "paraphrase_neg_sim_change_max": stats(paraphrase_neg_sim_change_max),  # sim change to paraphrases
            "paraphrase_neg_sim_change_mean": stats(paraphrase_neg_sim_change_mean),# sim change to paraphrases
        }

    return results


if __name__=="__main__":
    options = parser.parse_args()
    # dowload dataset and preprocess
    lang=None
    # if lang is given with column notation
    if ":" in options.data_name:
        options.data_name, lang = options.data_name.split(":")

    # download the dataset with data_name
    report("Downloading the corpus")
    corpus, queries, qrels = download_dataset(options.data_name, split_to_select=options.split, lang=lang)
    
    # create one-to-one correspondence: each query is in the same index with its answer
    report("Creating one-to-one")
    targets, questions, filler_targets, same_answer_mask = create_one_to_one_correspondence(corpus, queries, qrels)
    report(f"Found {len(questions)=}, {len(targets)=} and {(len(filler_targets) if filler_targets else filler_targets)=}")
    report(f"Sanity: {questions[0]=} {targets[0]=}")

    # read distractors:
    report("Reading distractors")
    distractors_raw, question_field = read_distractors(options.data_name, split_to_select=options.split, lang=lang)
    distractors = []
    if distractors_raw:
        # check we have one for each query:
        for i, par in enumerate(distractors_raw):
            q_original, q_par = par[question_field], par["generated_paraphrase"]
            q = questions[i]
            assert q.rstrip() == q_original.rstrip(), f"Unable to match queries and distractors\n{q} == {q_original}"
            distractors.append(q_par)
        report(f"Sanity: {questions[0]=} {distractors[0]=}")
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

    # save the results
    model_safe_name = options.model_name.replace("/", "__")
    data_safe_name = options.data_name.replace("/","__")
    if lang is not None:
        data_safe_name += f"_{lang}" # bring this back now
    specific_prompts = ""
    save_path = f"{options.save_prefix}/{model_safe_name}/{data_safe_name}{specific_prompts if lang is not None else ''}/{options.split}/{options.template}_template"
    
    report("Calculating")
    # calculate the metrics and eval results
    results = calculate_metrics(options.model_name,
                                prompts,
                                options.template,
                                questions,
                                targets,
                                distractors,
                                filler_targets,
                                k = options.nn,
                                batch_size=options.batch_size,
                                same_answers_mask=same_answer_mask)

    # save the results
    os.makedirs(save_path, exist_ok=True)
    report(f"Saving to {save_path}")
    with open(f'{save_path}/prompt_geometry_{options.nn}nn_1_distractor_and_1_false_positive.json', 'w') as f:
        json.dump(results, f, indent=2)
