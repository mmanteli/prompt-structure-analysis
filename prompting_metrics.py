from sentence_transformers import SentenceTransformer
import datasets
import os
import json
import numpy as np
import torch
from sklearn.preprocessing import normalize
import sys
from prompts import prompts_arcchallenge, prompts_summeval, prompts_tatoeba
# this contains simply lists and dictionaries that help select the correct prompts

cos = torch.nn.CosineSimilarity()

def load_dataset(path):
    if os.path.exists(path):
        return datasets.load_from_disk(path)
    return datasets.load_dataset(path)

def get_detailed_instruct(prompt: str, query: str) -> str:
    return f'Instruct: {prompt}\nQuery: {query}'

class Testmodel:
    """
    Class for generating random embeddings for cases where GPU is not available.
    Ununsed at this point.
    """

    def __init__(self, dim=3):
        self.dim=dim
    def encode(self, text, convert_to_tensor=True, normalize_embeddings=False):
        embedding = np.random.random((len(text), self.dim))
        if normalize_embeddings:
            embedding = normalize(embedding, axis=1)
            assert embedding.shape[1] == self.dim, f"{embedding.shape[1]} != {self.dim}"
            assert np.isclose(np.linalg.norm(embedding[0,:]), 1.0), f"Norm after normalizing: {np.linalg.norm(embedding[0,:])}"
        if convert_to_tensor:
            return torch.Tensor(embedding)
        return embedding


def unit_test():
    """Function to test the similarity measures."""
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


def looper_old(model_name, data_name):

    # read the data
    data_path = {"arcchallenge": "/flash/project_462000883/datasets/arcchallenge",
                "tatoeba:fin-eng": "/flash/project_462000883/datasets/tatoeba:en-fi",
                "summeval-2": "/flash/project_462000883/datasets/summeval-2"}[data_name]
    ds = load_dataset(data_path)
    print(f"Dataset loaded:\n{ds}")
    # download the model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    #model = Testmodel()   # <- for testing
    print("Model loaded.")

    # which columns to read from each dataset
    columns = {"arcchallenge": ('query', 'document'),
               "tatoeba:fin-eng": ('english', 'non_english'),
               "summeval-2": ('summary','text')}[data_name]

    # embed queries, no need to repeat
    split = "test"
    queries = ds[split][columns[0]]
    embeddings_q = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
    answers = ds[split][columns[1]]
    embeddings_a = model.encode(answers, convert_to_tensor=True, normalize_embeddings=True)

    # substract query from everything to calculate the angles from it instead of the origin.
    embeddings_a__q = embeddings_a - embeddings_q
    assert embeddings_a__q.shape == embeddings_a.shape, "Shape of embeddings_a__q does not match"

    # get the prompts to try
    prompts_to_try = {"arcchallenge": prompts_arcchallenge,
                      "tatoeba:fin-eng": prompts_tatoeba,
                      "summeval-2": prompts_summeval}[data_name]

    # actual calculation and result collecting
    results={}
    for i, p in enumerate(prompts_to_try):
        # embed prompt and template(prompt, query)
        prompts_and_queries = [get_detailed_instruct(p,q) for q in queries]
        embeddings_pq = model.encode(prompts_and_queries, convert_to_tensor=True, normalize_embeddings=True)
        # again, map query the origin:
        embeddings_pq__q = embeddings_pq - embeddings_q
        assert embeddings_pq__q.shape == embeddings_pq.shape, "Shape of embeddings_pq__q does not match"
        # calculate cosine distance
        similarity = cos(embeddings_a__q, embeddings_pq__q)
        assert similarity.shape[0] == len(prompts_and_queries), f"similarity dimensions do not match {similarity.shape} == {len(prompts_and_queries)} with number of instances"
        results[f"prompt{i}"] = {"mean": str(np.mean(similarity.detach().cpu().numpy().reshape(-1))), 
                                 "std": str(np.std(similarity.detach().cpu().numpy().reshape(-1))),
                                 "prompt_text": p}

    comp ="query" 
    model_name = model_name.replace("/", "__")
    os.makedirs(f"results_angle/{model_name}/{data_name}", exist_ok=True)
    with open(f'results_angle/{model_name}/{data_name}/prompt_data_cosine_query_to_{comp}.json', 'w') as f:
        json.dump(results, f)


def looper(model_name, data_name):

    data_path = {"arcchallenge": "/flash/project_462000883/datasets/arcchallenge",
                "tatoeba:fin-eng": "/flash/project_462000883/datasets/tatoeba:en-fi",
                "summeval-2": "/flash/project_462000883/datasets/summeval-2"}[data_name]
    ds = load_dataset(data_path)
    print(f"Dataset loaded:\n{ds}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print("Model loaded.")

    columns = {"arcchallenge": ('query', 'document'),
               "tatoeba:fin-eng": ('english', 'non_english'),
               "summeval-2": ('summary','text')}[data_name]

    split = "test" if "tatoeba" in data_name else "fit"
    queries = ds[split][columns[0]]
    answers = ds[split][columns[1]]

    embeddings_q = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
    embeddings_a = model.encode(answers, convert_to_tensor=True, normalize_embeddings=True)

    # Chord vector from query to answer
    delta_a = embeddings_a - embeddings_q  # (N, D)

    # Baseline: how similar are q and a without any prompt?
    sim_qa = cos(embeddings_q, embeddings_a)  # (N,)

    prompts_to_try = {"arcchallenge": prompts_arcchallenge,
                      "tatoeba:fin-eng": prompts_tatoeba,
                      "summeval-2": prompts_summeval}[data_name]

    results = {}
    for i, p in enumerate(prompts_to_try):
        prompts_and_queries = [get_detailed_instruct(p, q) for q in queries]
        embeddings_pq = model.encode(prompts_and_queries, convert_to_tensor=True, normalize_embeddings=True)

        # Chord vector from query to prompted query
        delta_pq = embeddings_pq - embeddings_q  # (N, D)

        # ---- Metric 1: Your original metric ----
        # Cosine similarity between the two chord vectors
        # "Does the prompt move the query in the same direction as the answer?"
        chord_sim = cos(delta_a, delta_pq)  # (N,)

        # ---- Metric 2: Direct similarity improvement ----
        # "Does adding the prompt make pq closer to a than q was?"
        sim_pqa = cos(embeddings_pq, embeddings_a)  # (N,)
        sim_improvement = sim_pqa - sim_qa           # (N,)

        # ---- Metric 3: How far did the prompt move the query? ----
        displacement = torch.norm(delta_pq, dim=1)  # (N,)

        # ---- Metric 4: Parallel vs orthogonal decomposition ----
        # Project delta_pq onto the direction of delta_a
        # This tells you: of the total movement caused by the prompt,
        # how much is *toward the answer* vs *sideways*?
        delta_a_norm = delta_a / (torch.norm(delta_a, dim=1, keepdim=True) + 1e-10)
        parallel_magnitude = torch.sum(delta_pq * delta_a_norm, dim=1)      # (N,) signed scalar projection
        orthogonal_magnitude = torch.sqrt(
            torch.clamp(torch.sum(delta_pq ** 2, dim=1) - parallel_magnitude ** 2, min=0.0)
        )  # (N,)

        # Ratio: what fraction of the movement is toward the answer?
        parallel_fraction = parallel_magnitude / (displacement + 1e-10)  # (N,), in [-1, 1]

        def stats(t):
            """Helper to extract summary statistics"""
            arr = t.detach().cpu().numpy().reshape(-1)
            return {"mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75))}

        results[f"prompt{i}"] = {
            "prompt_text": p,
            "chord_similarity":     stats(chord_sim),          # your original metric
            "sim_q_a":              stats(sim_qa),              # baseline similarity
            "sim_pq_a":             stats(sim_pqa),             # prompted similarity
            "sim_improvement":      stats(sim_improvement),     # delta
            "displacement":         stats(displacement),        # how far prompt moved q
            "parallel_magnitude":   stats(parallel_magnitude),  # movement toward answer (signed)
            "orthogonal_magnitude": stats(orthogonal_magnitude),# movement sideways
            "parallel_fraction":    stats(parallel_fraction),   # fraction toward answer
        }

    model_name_safe = model_name.replace("/", "__")
    os.makedirs(f"results_angle2/{model_name_safe}/{data_name}", exist_ok=True)
    with open(f'results_angle2/{model_name_safe}/{data_name}/prompt_geometry.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__=="__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # run "unit test"
            unit_test()
            exit()
        else:
            print("Usage: no params (default: prompt comparison) OR one param: test: unit test, answer/A: answer comparison, prompt/P: prompt comparison.")

    data_names = ["arcchallenge", "tatoeba:fin-eng", "summeval-2"]
    models = ["BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B","intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large-instruct", "minishlab/potion-base-8M", "google/embeddinggemma-300m"]
    
    for d in data_names:
        for m in models:
            print(f"Starting model {m} on {d}")
            looper(m, d)




