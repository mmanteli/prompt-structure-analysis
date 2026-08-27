from sentence_transformers import SentenceTransformer
import datasets
import os
import json
import numpy as np
import torch
import jsonargparse
import sys
import random
import pickle
from evaluate_prompts import find_relevant_doc_id, write_embeddings
from scipy.spatial.distance import pdist, squareform
from utils.dataset_handling import download_dataset
from utils.prompts import get_prompts, get_detailed_instruct
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
             
        

def unit_test():
    """Function to test the angulation similarity measures."""
    # check that multidim calculations work, i.e. dimensions match
    Q = torch.Tensor([[1.,0.,0.], [0.5,0.,0.], [0.,2.,2.]])
    A = torch.Tensor([[1.,1.,0.], [0.,0.,1.], [0.,1.,1.]])
    # Q1->A1: 45 degree, should be 0.7071
    # Q2->A2: 90 degree, should be 0
    # Q3->A3: same direction, should be 1
    for i, j in zip(cos(Q,A), torch.tensor([0.7071, 0.0, 1.0])):
        assert np.isclose(i,j, atol=1e-4), f"Test 1 not passed, {i}, {j}"
    # subtracting query --> measuring angle wrt. vector A-Q, not wrt. origin
    # i.e. see the angle of Q->P compared to Q->A
    Q = torch.Tensor([[2.,4.], [2.,4.], [-4.,2.]])
    A = torch.Tensor([[6.,4.], [6.,4.], [-2.,4.]])
    P = torch.Tensor([[4.,5.], [2.,2.],[-4.,3.]])
    A__Q = A-Q
    P__Q = P-Q
    # A__Q1 & P__Q1: 2/np.sqrt(5) = 0.89442
    # A__Q2 & P__Q2: 90 degrees, so should be 0
    # A__Q3 & P__Q3: 45 degrees, from triangle = adj/hypot = (np.sqrt(2)/2)/1 = 1/ np.sqrt(2) = 0.7071
    for i,j in zip(cos(A__Q, P__Q), torch.Tensor([0.8944, 0.0, 0.7071])):
        assert np.isclose(i,j, atol=1e-4), f"Test 2 not passed, {i}-{j}"
    print("Success: All tests passed.")

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


def create_one_to_one_correspondence(corpus, queries, qrels):
    """
    For each query, find the best match in the corpus.
    Return them in order, and additionally, return the 'unused'/'filler' corpus texts
    """
    # if the dataset is already sorted in this way (some dataset download scripts return this format)
    if len(corpus) == len(queries) and qrels == {k:k for k in range(len(queries))}:
        return corpus["text"], queries["text"], None
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
            "q75": float(np.percentile(arr, 75)),
            "full": str(arr.tolist())}


def calculate_metrics(model_name, queries, answers, prompts, template, wrong_answers=None, k=10, batch_size=8, embeddings=None):
    """
    model_name: path or huggingface alias
    queries: preprocessed queries
    anwers: proprocessed corpus
        NOTE: queries and corpus need to have 1 to 1 correspondence, i.e. no qrels here
    prompts: prompts to iterate over
    wrong_answers: "leftovers" from corpus, texts that do not correpond to a query. These
        will be added in the "negative" metrics
    k = number of neighbors considered. for kNN, it is on the query and target side (1 to 1)
        while in the negative metrics, it is on the target side
    batch_size= batch size for embedding
    """
    # load the model or precalculated results
    calculate_embeddings = True
    if embeddings is not None and os.path.exists(embeddings):
        calculate_embeddings = False # we already have results
    if calculate_embeddings:  
        report("Downloading model...")
        model = SentenceTransformer(model_name, trust_remote_code=True)
        report("Model loaded.")
        # embed the "ground truth values": regular queries and targets
        embeddings_q = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
        embeddings_a = model.encode(answers, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
        if embeddings:  # now we can still save the embeddings if we want to
            report(f"Writing embeddings to {embeddings}")
            write_embeddings(file=embeddings, key="corpus", data={"text":answers}, embeddings=embeddings_a)
            write_embeddings(file=embeddings, key="queries", data={"text":queries}, embeddings=embeddings_q)
        # if the dataset has filler/wrong answers, answers with no question that answers then, embed them as well
        # and create a "all" embeddings variable: used in the negative metrics
        if wrong_answers:
            # yes filler, so concatenate them at the end
            embeddings_wa = model.encode(wrong_answers, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
            if embeddings:
                write_embeddings(file=embeddings, key="wrong_answers", data={"text":wrong_answers}, embeddings=embeddings_wa)
            embeddings_all_a = torch.concat((embeddings_a, embeddings_wa))
        else:
            # no filler
            embeddings_all_a = embeddings_a
    else:  # we read embeddings, not calculate them
        print("Reading precalculated embeddings")
        prec = read_from_pickle(embeddings, "corpus")
        assert prec["data"]["text"][0] == answers[0], f'{prec["data"]["text"][0]}!={answers[0]}'# here sanity check
        embeddings_a = prec["data"]["embeddings"]
        prec = read_from_pickle(embeddings, "queries")
        embeddings_q = prec["data"]["embeddings"]
        if wrong_answers:
            prec = read_from_pickle(embeddings, "wrong_answers")
            embeddings_wa = prec["data"]["embeddings"]
            embeddings_all_a = torch.concat((embeddings_a, embeddings_wa))
        else:
            # no filler
            embeddings_all_a = embeddings_a



    # calculate everything we can calculate without using the prompt
    # chord vector from query to answer
    delta_a = embeddings_a - embeddings_q 

    # baseline: how similar are q and a without any prompt?
    sim_qa = cos(embeddings_q, embeddings_a)

    # negatives for metrics 6 & 7: here we can use the "wrong" answers (if they exist)
    N_pool = embeddings_all_a.shape[0]
    N_q = embeddings_q.shape[0]
    k_hard = min(k, N_pool - 1)
    # add the filler
    sim_q_all = torch.mm(embeddings_q, embeddings_all_a.T)  # here we use embeddings_all_a
    # For each query, find indices of k nearest *wrong* answers
    # since embeddings_q and embeddings_a are in order, and we just append embeddings_wa
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
        # sanity check printout: see that template is filled correctly
        print(f"Example of what is embedded:\n----\n{prompts_and_queries[0]}\n----\n")
        if calculate_embeddings:
            embeddings_pq = model.encode(prompts_and_queries, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
            if embeddings:
                write_embeddings(file=embeddings, key=p, data={"text": prompts_and_queries}, embeddings=embeddings_pq)
        else:  # just read
            prec = read_from_pickle(embeddings, p)
            assert prec["data"]["text"] == prompts_and_queries, "mismatch between precalculated and dataset"
            embeddings_pq = prec["data"]["embeddings"]

        # Chord vector from query to prompted query ( to compare with delta_a)
        delta_pq = embeddings_pq - embeddings_q

        # ---- Metric 1: Angulation toward answer ----
        # cosine similarity between the two chord vectors
        # "Does the prompt move the query in the same direction as the answer?"
        chord_sim = cos(delta_a, delta_pq)

        # ---- Metric 2: Direct similarity improvement ----
        # "Does adding the prompt make pq closer to a than q was?"
        sim_pqa = cos(embeddings_pq, embeddings_a)
        sim_improvement = sim_pqa - sim_qa

        # ---- Metric 3: Direct distance ----
        # "How far did the prompt move the query?"
        displacement = torch.linalg.norm(delta_pq, dim=1)

        # ---- Metric 4: Parallel vs orthogonal decomposition ----
        # Project delta_pq onto the direction of delta_a
        # out of the total movement caused by the prompt,
        # how much is *toward the answer* vs *sideways*?
        delta_a_norm = delta_a / (torch.linalg.norm(delta_a, dim=1, keepdim=True) + 1e-10)
        # 1e-10 here to avoid NaN --> has not other effect since norm is 0 <==> tensor is identically 0
        parallel_magnitude = torch.sum(delta_pq * delta_a_norm, dim=1)      # signed scalar projection
        orthogonal_magnitude = torch.sqrt(
            torch.clamp(torch.sum(delta_pq ** 2, dim=1) - parallel_magnitude ** 2, min=0.0),
        )  # follows from pythagorean theorem

        # Ratio: what fraction of the movement is toward the answer?
        parallel_fraction = parallel_magnitude / (displacement + 1e-10)  # (N,), in [-1, 1]

        # sanity check: parallel_fraction and chord_sim should equal (simply from pythagorean theorem)
        #assert torch.allclose(parallel_fraction, chord_sim, atol=1e-4), \
        #    f"Parallel fraction and chord sim do not match: {parallel_fraction} != {chord_sim}"
        # assert removed since this has float-accuraty issues, and result analysis will handle it anyway

        # ---- Metric 5: knn retention ----
        # "Does adding prompt make the query side structure resemble the answer side"
        knn_retention = knn_overlap_score(embeddings_pq.detach().cpu(), embeddings_a.detach().cpu(), k=k)

        # ---- Metric 6: displacement vs. wrong answers ----
        # "Does adding prompt take you away from incorrect answers?"
        # For the k-hardest wrong answers (most similar to the unprompted query),
        # measure how much their similarity changes after prompting.
        # Negative = prompt moved query away from hard negatives (desirable).
        # so, add multiplication by -1 
        # now positive = desirable
        sim_pq_all = torch.mm(embeddings_pq, embeddings_all_a.T)
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
        all_embeddings = embeddings_all_a  # (N+M, D)
        hard_neg_angulation = -1 * torch.tensor([
            cos(
                delta_pq[i].unsqueeze(0).expand(k_hard, -1),    # (k_hard, D)
                all_embeddings[hard_neg_indices[i]] - embeddings_q[i]  # (k_hard, D)
            ).mean().item()
            for i in range(N_q)
        ])


        results[f"prompt{prompt_num}"] = {
            "prompt_text": p if p != "" else "empty",           # prompt text, with "" redirected to "empty"
            "example_text": prompts_and_queries[0],             # example text as a sanity check
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
    lang=None
    # if lang is given with column notation
    if ":" in options.data_name:
        options.data_name, lang = options.data_name.split(":")

    # download the dataset with data_name
    # this returns queries and corpus, both datasets.Dataset, and 
    # qrels which may be a datasets.Dataset or a dict{query_id:corpus_is}
    corpus, queries, qrels = download_dataset(options.data_name, lang=lang, split_to_select=options.split, downsample=options.num_examples)
    # download prompts
    # this returns a list of possible instructions to use on the query side
    if options.use_lang_specific_prompts:
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
    # create 1-to-1 correspondence (i.e. sort by qrels)
    # and return "filler targets" == texts in corpus that do not correspond to a query
    # naming of questions, targets to not confuse with the unprocessed corpus and queries
    targets, questions, filler_targets = create_one_to_one_correspondence(corpus, queries, qrels)
    print(f"Sanity check: These should be a pair:\n{questions[0]=}\n{targets[0]}")
    
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
    # calculate the metrics
    results = calculate_metrics(options.model_name, questions, targets, prompts, options.template, wrong_answers=filler_targets, k = options.k, batch_size=options.batch_size, embeddings=embeddings_path)
    
    # save the results
    os.makedirs(save_path, exist_ok=True)
    report(f"Saving to {save_path}")
    with open(f'{save_path}/prompt_geometry.json', 'w') as f:
        json.dump(results, f, indent=2)