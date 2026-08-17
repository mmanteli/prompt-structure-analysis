import datasets
import numpy as np

def calculate_non_rank_metrics(found_ids, relevant_ids):
    """
    Calculate non-rank related metrics. 
    Both lists assumed to be sorted, found_ids by similarity and relevant_ids by relevance
    """
    tp = sum(1 for fid in found_ids if fid in relevant_ids)
    # recall is normal
    recall = tp / len(relevant_ids)
    # precision is artificially deflated: if there are 2 relevant docs
    # but we set k=10 for 10 found ids
    # even if we found the two relevant at the top
    # precision will be low
    # hence, also r-precision
    precision = tp / len(found_ids)
    tp_ = sum(1 for fid in found_ids[:len(relevant_ids)] if fid in relevant_ids)
    rprecision = tp_ / len(relevant_ids)
    # for F1, we are interested in the top1 match
    # this is for tasks with one to one binary qrels
    #f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1_at_1 = 1 if found_ids[0] == relevant_ids[0] else 0
    return f1_at_1, recall, precision, rprecision


def sanity_check_logic(k=2):
    """Check the logic of metric calculation"""
    # construct examples for the sanity check
    corpus = datasets.Dataset.from_dict({"_id":["c14","c21","c33","c41"], "text":["a","b","c","d"]})
    queries = datasets.Dataset.from_dict({"_id": ["q1", "q2", "q4"], "text":["correct is mostly a, but d is okay", "correct is c", "correct is a+b+c"]})
    qrels = datasets.Dataset.from_dict({"query_id":["q1", "q1", "q2", "q4", "q4", "q4"], "corpus_id":["c14", "c41", "c33", "c14","c21","c33"], "score": [1, 0.9, 1, 1,0.99,0.9]})
    # similarities: normally, this would be emb(queries)@emb(corpus).T
    # Perfect match for q1 (a is highest, d second highes) and second best for q2 (b is highest, followed by c, which is correct)
    # for the last on, the correct ones are correct but the last to are in wrong order
    sims = np.array([[0.8, -0.3, 0.3, 0.6], [-0.2, 0.92, 0.90, 0.1], [0.9, 0.6, 0.7, -0.2]])
    # argsort to get most similar indices
    sims = np.argsort(sims, axis=1)[:, ::-1]
    recall_at_k = []
    mrr_at_k = []
    ndcg_at_k = []
    f1_at_1 = []
    precision_at_k = []
    rprecision_at_k = []
    for (i, query), sim_line in zip(enumerate(queries), sims):
        print("In Q:", query["_id"],", text=", query["text"])
        relevant_ids, associated_scores = find_relevant_doc_id(query["_id"], qrels)
        most_similar_docs = sim_line[:k]
        found_ids = [corpus["_id"][j] for j in most_similar_docs]
        print(f"{relevant_ids=}, {associated_scores=}")
        ideal_cumulative_gain = np.sum([(2**s-1)/np.log2(rank+1+1.) for rank, s in enumerate(np.sort(associated_scores)[::-1][:k])])   # UP TO k
        # if everything was perfect: highest score at rank1, second at rank2
        # also +1+1 since the rank is zero indexed here -> one +1 to fix rank and other is in the formula
        discounted_cumulative_gain = 0
        mrr_ = 0
        # we can already calculate some results with no rank information
        #rec_ = sum(1 for fid in found_ids if fid in relevant_ids) / len(relevant_ids)
        f1_, rec_, prec_, rprec_= calculate_non_rank_metrics(found_ids, relevant_ids)
        for rank_, found_id in enumerate(found_ids):
            rank = rank_+ 1 # fix zero indexing
            print(f"{found_id} found in rank {rank}")
            if found_id in relevant_ids:
                mrr_ = 1/(rank) if mrr_==0 else mrr_  # again, only first match counts, so only set at highest rank
                found_score = associated_scores[relevant_ids.index(found_id)]
                discounted_cumulative_gain += (2**found_score-1)/np.log2(rank+1)
        current_ndcg_at_k = 1 if ideal_cumulative_gain == 0 else discounted_cumulative_gain/ideal_cumulative_gain # 1 for "nothing relevant was to be discovered"
        ndcg_at_k.append(current_ndcg_at_k)
        mrr_at_k.append(mrr_)
        recall_at_k.append(rec_)
        f1_at_1.append(f1_)
        precision_at_k.append(prec_)
        rprecision_at_k.append(rprec_)
    print("Final results")
    print("RECALL", np.mean(recall_at_k))
    print("MRR", np.mean(mrr_at_k))
    print("NDCG", np.mean(ndcg_at_k))
    print("F1", np.mean(f1_at_1))
    print("R-PRECISION", np.mean(rprecision_at_k))

def find_relevant_doc_id(query_id, qrels):
    if isinstance(qrels, dict):
        return (qrels[query_id], 1) if query_id in qrels else (qrels[f"ARC-Challenge-q-{query_id}"],1)
    indices_of_query_ids = np.where(np.array(qrels["query_id"]) == query_id)[0]
    associated_corpus_values = np.array(qrels["corpus_id"])[indices_of_query_ids]
    associated_corpus_scores = np.array(qrels["score"])[indices_of_query_ids]
    # sort these to have the best match at the top
    indices_that_sort = np.argsort(associated_corpus_scores)[::-1]
    return (associated_corpus_values[indices_that_sort].tolist(), associated_corpus_scores[indices_that_sort].tolist())



def sanity_check_one_to_one_correspondence():
    """
    For each query, find the best match in the corpus.
    Return them in order, and additionally, return the 'unused' corpus texts
    """
    corpus = datasets.Dataset.from_dict({"_id":["c14","c21","c33","c41"], "text":["text a","text b","text c","text d"]})
    queries = datasets.Dataset.from_dict({"_id": ["q1", "q2", "q4"], "text":["correct is mostly a, but d is okay", "correct is c", "correct is d"]})
    qrels = datasets.Dataset.from_dict({"query_id":["q1", "q1", "q2", "q4",], "corpus_id":["c14", "c41", "c33", "c41",], "score": [1, 0.9, 1, 1]})
    # if the dataset is already sorted in this way
    if len(corpus) == len(queries) and qrels == {k:k for k in range(len(queries))}:
        return corpus["text"], queries["text"], None
    questions = []
    targets = []
    found_ids = set()
    # loop over queries
    for line in queries:
        q_id, q_text = line.values()
        print(f"Now in {q_id=}, {q_text=}")
        # find the relevant answers based on the query id
        relevant_ids, assoc_scores = find_relevant_doc_id(q_id, qrels)
        print(f"{relevant_ids=}, {assoc_scores=}")
        # best match is at the top of the list
        most_relevant_id = relevant_ids[0]
        c_id, c_text = [l for l in corpus if l["_id"] == most_relevant_id][0].values()
        print(f"Found {c_id=}, {c_text=}")
        questions.append(q_text)
        targets.append(c_text)
        found_ids.add(most_relevant_id)  # here we could also choose all relevant ids
    unmatched_targets = corpus.filter(lambda example: example["_id"] not in found_ids)
    print(f"{unmatched_targets['text']=}")
    return  targets, questions, unmatched_targets["text"]


if __name__=="__main__":
    #sanity_check_logic(k=3)
    sanity_check_one_to_one_correspondence()