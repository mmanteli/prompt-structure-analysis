from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np
import torch
import jsonargparse
import random
from evaluate_prompts import find_relevant_doc_id, calculate_non_rank_metrics
from prompting_metrics import knn_overlap_score
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

def create_one_to_one_correspondence(corpus, queries, qrels):
    """
    Separate questions, distractors, targets, and answers that answer no questions.
    - questions == list of queries which have an answer (in all datasets just all queries)
    - targets == list of answers for the above
    - distractors == for a question, the other questions that have the same answer
    - filler == answers that have no associated question
    We separate them like this so that in evaluation we can simply use the same ids for Q and A sides.
    """
    questions = []
    targets = []
    found_ids = set()   # this is used to filter the filler in the end
    # loop over queries
    for line in queries:
        q_id, q_text = line["_id"], line["text"]
        # find the relevant answers based on the query id from qrels
        # this funtion returns the best match at the top of the list
        relevant_ids, assoc_scores = find_relevant_doc_id(q_id, qrels)
        # our datasets are binary anyway, so this list is always just one long
        # so choosing index 0, but this structure ensures easy modification later
        most_relevant_id = relevant_ids[0]
        # find all corpus texts that have the most_relevant_id
        found = [l for l in corpus if l["_id"] == most_relevant_id]
        assert len(found) == 1, f"Duplicate ids in corpus, {most_relevant_id=} resulted in {found=}"
        #c_id, c_text = found[0]["_id"], found[0]["text"]
        c_text = found[0]["text"]
        questions.append(q_text)
        targets.append(c_text)
        found_ids.add(most_relevant_id)

    # We now know that indices of questions and targets match questions[0]=>targets[0]
    # Next, look for the distractors == paraphrases of the query
    # we approximate them as questions that have the same answer
    unique_text_ids = np.unique(targets)    # unique answers
    distractor_id_dict = {k: np.where(np.array(targets) == k)[0] for k in unique_text_ids} # [0] to remove tuple
    distractor_id_list_per_targets = [distractor_id_dict[t] for t in targets] # which ids share the same answer
    # ok now just rmove the index itself (we do not want to match to self)
    distractor_id_list_per_question = [[d for d in distractor_id_list_per_targets[i] if d!=i] for i in range(len(questions))]
    # ^^ these are ids of the paraphrase questions
    distractor_text_list_per_question = [np.array(questions)[i] for i in distractor_id_list_per_question]
    # ^^ their associated texts
    report("----------Sanity check----------")
    report(f"1. These should be a pair:\n{questions[0]=}\n{targets[0]}")
    report(f"2. For Q={questions[0]} the distractors (near paraphrases) are\n{distractor_text_list_per_question[0]}")
    report("--------Sanity check end--------")
    # last, the corpus texts that did not correspond to any answer (they are just filler)
    unmatched_targets = corpus.filter(lambda example: example["_id"] not in found_ids)
    # return all
    return  targets, questions, distractor_id_list_per_question, unmatched_targets["text"]


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
            "full": str(arr),
            }

def calculate_scores(k, query_embeddings, corpus_embeddings, distractors, additional_mask=None, prompt_text=None):
    """
    Calculate retrieval metrics for corpus and query embeddings
    query_embeddings = matrix of query embeddings
    corpus_embeddings = matrix of corpus embeddings (in the same order as queries, with possible tail
        of non relevant answers)
    distractors = ids of duplicate answers/paraphrase questions
    additional_mask = ids of other things we need to mask 
    """
    # calculate similarity matrix
    sims = query_embeddings @ corpus_embeddings.T
    sims = sims.detach().cpu().numpy()
    # argsort sims to get best matches
    # here the additional ":" is needed together with axis=1, see sanity_check_sorting()
    #sims = np.argsort(sims, axis=1)[:, ::-1]
    # actually, do this later because we need to mask per query!
    # initialize data collection
    recall_at_k = []
    mrr_at_k = []
    ndcg_at_k = []
    f1_at_1 = []  # this is the metric for bitext mining (binary task, hence "at_1")
    precision_at_k = []
    rprecision_at_k = []
    for i, sim_line in enumerate(sims):
        # mask sim_line:
        # we do not want the duplicate answers
        sim_line[distractors[i]] = -float('inf')
        # if given, mask the duplicate question as well
        if additional_mask:
            sim_line[additional_mask[i]] = -float('inf')
        sim_line_sorted = np.argsort(sim_line)[::-1]
        # Calculation: first, find the relevant ids
        relevant_ids, associated_scores = [i], [1] # it's i and 1, thet are in order already
        # from these, calculate the ideal_cumulative_gain (used to normalize discounted cumulative gain)
        ideal_cumulative_gain = np.sum([(2**s-1)/np.log2(rank+1+1.) for rank, s in enumerate(np.sort(associated_scores)[::-1][:k])])
        # +1+1 since the rank is zero indexed here -> one +1 to fix rank and other is in the formula
        # Next, find best k matches
        # again, they are already in order
        most_similar_docs = sim_line_sorted[:k]
        found_ids = most_similar_docs #[j for j in most_similar_docs]

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
        current_ndcg_at_k = 0 if ideal_cumulative_gain == 0 else discounted_cumulative_gain/ideal_cumulative_gain 
        # ^^0 if nothing was to be discovered
        # collect results
        ndcg_at_k.append(current_ndcg_at_k)
        mrr_at_k.append(mrr_)
        recall_at_k.append(rec_)
        f1_at_1.append(f1_)
        precision_at_k.append(prec_)
        rprecision_at_k.append(rprec_)

    return {"prompt_text": prompt_text,
            f"mrr@{k}": stats(mrr_at_k),
            f"recall@{k}": stats(recall_at_k),
            f"ndcg@{k}": stats(ndcg_at_k),
            f"F1": stats(f1_at_1),
            f"precision@{k}": stats(precision_at_k),
            f"rprecision@{k}": stats(rprecision_at_k),
            }


def calculate_metrics(model_name, prompts, template, queries, answers, distractors, wrong_answers, k=10, batch_size=8):
    """
    Calculate retrieval metrics.
    model_name: path or huggingface alias
    prompts: prompts to iterate over
    template: "Instruct-Query" or "simple", tells get_detailed_instruct() what to do
    queries: preprocessed queries
    anwers: proprocessed corpus
        NOTE: queries and corpus need to have 1 to 1 correspondence, i.e. no qrels here
    distractors: ids of near paraphrase queries, e.g. for queries[0], distractors[0] = [1,2,3]
        if first 4 queries are paraphrases
    wrong_answers: "leftovers" from corpus, texts that do not correpond to a query.
    k = number of neighbors considered in all calculations
    batch_size= batch size for embedding
    """
    report("Downloading model...")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    report("Model loaded.")
    # embed the "ground truth values": regular queries and targets
    embeddings_q = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
    embeddings_a = model.encode(answers, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size)
    # if the dataset has filler/wrong answers, answers with no question that answers then, embed them as well
    # and create a "all" embeddings variable: the ids still match to queries bc we only append
    # we use embeddings_a_all in any calculation where it is not strictly necessary to have 1-to-1 QA pairs
    if wrong_answers:
        # yes filler, so concatenate them at the end
        embeddings_wa = model.encode(wrong_answers, convert_to_tensor=True,
                                     normalize_embeddings=True, batch_size=batch_size)
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
    k_hard = min(k, N_pool - 1)
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
        # mask the identical answers (located in the answer side of distractors)
        sims_i[distractors[i]] = -float('inf')
        hard_neg_indices.append(torch.topk(sims_i, k_hard).indices)
    assert all([len(set(hard_neg_indices[qi]) & set(distractors[qi])) == 0 for qi in range(N_q)]), \
        f"hard neg indices calculation problem, example: {set(hard_neg_indices[0])=} & {set(distractors[0])=}"

    # last thing:
    # to calculate similarity change from k-nearest distractors
    # we need to know which distractors are k-closest
    # above, we need to mask all (otherwise we mark duplicate answers as hard negs)
    # but here, we simply select the ones that are k-closest to vanilla query
    # explanation: calculate similarity between one query embedding and its distractor embeddings (by a@b.T)
    # then select indices that are the most similar topk(a@b.T, k).indices
    # then take top k (with the possibility that there might be less than k)
    #distractors_original = distractors
    report(f"Distractors before filtering: {distractors[0]=}")
    full_distractors = distractors   # save these for eval!!
    distractors = [(np.array(distractors[i])[torch.argsort(embeddings_q[i]@embeddings_q[distractors[i],:].T, descending=True)[:k].cpu()]) for i in range(N_q)]
    report(f"Distractors after filtering: {distractors[0]=}")
    # calculate metrics per prompt
    results = {}
    results_vanilla_eval = {}
    results_distractor_eval = {}
    for prompt_num, p in enumerate(prompts):
        # apply template and embed the prompt+query
        prompts_and_queries = [get_detailed_instruct(p, q, template=template) for q in queries]
        # sanity check printout: see that template is filled correctly
        report(f"Sanity: Example of what is embedded:\n----\n{prompts_and_queries[0]}\n----\n")
        embeddings_pq = model.encode(prompts_and_queries, convert_to_tensor=True,
                                        normalize_embeddings=True, batch_size=batch_size)

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
        #report(f"{hard_neg_angulation_max.shape=}")
        hard_neg_angulation_mean = -1 * torch.tensor([
            cos(
                delta_q2pq[i].unsqueeze(0).expand(k_hard, -1),    # same here
                embeddings_all_a[hard_neg_indices[i]] - embeddings_q[i],  # and here
            ).mean().item()
            for i in range(N_q)
        ])

        # metric 6: displacement from Q* = paraphrases
        # same as 4, but we compare to query side with distractor indices
        sim_q_itself = torch.mm(embeddings_q, embeddings_q.T)
        sim_pq_with_prompted = torch.mm(embeddings_pq, embeddings_q.T)
        paraphrase_neg_sim_change_max = -1*torch.tensor([
            (sim_pq_with_prompted[i, distractors[i]] - sim_q_itself[i, distractors[i]]).max().item()
            for i in range(N_q) if isinstance(distractors[i], np.int64) or len(distractors[i]) > 0
        ])

        paraphrase_neg_sim_change_mean = -1*torch.tensor([
            (sim_pq_with_prompted[i, distractors[i]] - sim_q_itself[i, distractors[i]]).mean().item()
            for i in range(N_q) if isinstance(distractors[i], np.int64) or len(distractors[i]) > 0
        ])


        # evaluation here at the same time
        results_vanilla_eval[f"prompt{prompt_num}"] = calculate_scores(k, 
                                                                        embeddings_pq, 
                                                                        embeddings_all_a, 
                                                                        distractors=full_distractors,   # here we want everything masked
                                                                        prompt_text=p if p != "" else "empty")
        additional_mask = [len(embeddings_all_a)+i for i in range(N_q)]  # masks each duplicate question
        results_distractor_eval[f"prompt{prompt_num}"] = calculate_scores(k,
                                                        embeddings_pq,
                                                        torch.cat([embeddings_all_a, embeddings_q]),
                                                        distractors=full_distractors,  # here again, because we add them in embeddings_q
                                                        additional_mask=additional_mask,
                                                        prompt_text=p if p != "" else "empty"
                                                       )
        
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

    return results, results_vanilla_eval, results_distractor_eval




if __name__=="__main__":
    options = parser.parse_args()
    # dowload dataset and preprocess
    lang=None
    # if lang is given with column notation
    if ":" in options.data_name:
        options.data_name, lang = options.data_name.split(":")

    # download the dataset with data_name
    corpus, queries, qrels = download_dataset(options.data_name,
                                                lang=lang,
                                                split_to_select=options.split,
                                                downsample=options.num_examples)
    # download prompts
    # this returns a list of possible instructions to use on the query side
    if options.use_lang_specific_prompts:
        assert lang is not None, "Give language for language specific prompts"
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

    # preprocess the data
    # create 1-to-1 correspondence (i.e. sort by qrels)
    # and return "filler targets" == texts in corpus that do not correspond to a query
    # distractors == near-paraphrase questions
    # naming of questions, targets to not confuse with the unprocessed corpus and queries
    targets, questions, distractors, filler_targets = create_one_to_one_correspondence(corpus, queries, qrels)


    # save the results
    model_safe_name = options.model_name.replace("/", "__")
    data_safe_name = options.data_name.replace("/","__")
    if lang is not None:
        data_safe_name += f"_{lang}" # bring this back now
    specific_prompts = "_lang_specific" if options.use_lang_specific_prompts else ""
    save_path = f"{options.save_prefix}/{model_safe_name}/{data_safe_name}{specific_prompts if lang is not None else ''}/{options.split}/{options.template}_template"
    # calculate the metrics
    results, results_vanilla_eval, results_distractor_eval = calculate_metrics(options.model_name,
                                                                                prompts,
                                                                                options.template,
                                                                                questions,
                                                                                targets,
                                                                                distractors,
                                                                                filler_targets,
                                                                                k = options.k,
                                                                                batch_size=options.batch_size)

    # save the results
    os.makedirs(save_path, exist_ok=True)
    report(f"Saving to {save_path}")
    with open(f'{save_path}/prompt_geometry_with_paraphrase_distractors_k{options.k}.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(f'{save_path}/prompt_eval_k{options.k}.json', 'w') as f:
        json.dump(results_vanilla_eval, f, indent=2)
    with open(f'{save_path}/prompt_eval_with_paraphrase_distractors_k{options.k}.json', 'w') as f:
        json.dump(results_distractor_eval, f, indent=2)
