from sentence_transformers import SentenceTransformer
import datasets
import os
import json
import numpy as np
import torch
import jsonargparse
import sys
from utils.prompts import get_prompts_arcchallenge, get_prompts_summeval, get_prompts_tatoeba, get_prompts_webfaq
# this contains simply lists and dictionaries that help select the correct prompts

cos = torch.nn.CosineSimilarity()


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
parser.add_argument('--save_prefix', type=str, default="results_metrics",
                    help="Saving path, model_name and k added in script")


def load_dataset(path):
    if os.path.exists(path):
        return datasets.load_from_disk(path)
    return datasets.load_dataset(path)

def get_detailed_instruct(prompt, query, template="Instruct-Query"):
    if template == "simple":
        return f"{prompt}. {query}"
    return f'Instruct: {prompt}\nQuery: {query}'


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



def looper(model_name, data_name, options):
    """
    Main logic: embed query and answer/target sides, and compare them to
    the embedding of template(prompt, query).
    """
    data_name_for_saving=data_name
    # handle the language
    lang=None
    if ":" in data_name:
        data_name, lang = data_name.split(":")
    # load data
    if data_name == "tatoeba":
        if "-" in lang:   # tatoeba uses different lang scheme
                lang_to_search_tatoeba = "en-"+lang.split("-")[0][:2]
        else:
            raise AttributeError("Give tatoeba lang in fin-eng, deu-eng, etc. format")
        print(f"Reading /flash/project_462000883/datasets/tatoeba:{lang_to_search_tatoeba}")
        data_path = f"/flash/project_462000883/datasets/tatoeba:{lang_to_search_tatoeba}"
    elif data_name == "webfaq":
        print(f"Reading /flash/project_462000883/datasets/web-faq-bitext:{lang}")
        data_path = f"/flash/project_462000883/datasets/web-faq-bitext:{lang}"
    elif data_name == "arcchallenge":
        print("Reading /flash/project_462000883/datasets/arcchallenge")
        data_path = "/flash/project_462000883/datasets/arcchallenge"
    elif data_name == "summeval-2":
        print("Reading /flash/project_462000883/datasets/summeval-2")
        data_path = "/flash/project_462000883/datasets/summeval-2"
    ds = load_dataset(data_path)
    print(f"Dataset loaded:\n{ds}")
    # which columns to read
    columns = {"arcchallenge": ('query', 'document'),
               "tatoeba": ('english', 'non_english'),
               "summeval-2": ('summary','text'),
               "webfaq": ("question2", "answer2")}[data_name]

    split = options.split
    queries = ds[split][columns[0]]
    answers = ds[split][columns[1]]

    # load the model
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print("Model loaded.")
    # embed the "ground truth values": regular queries and targets
    embeddings_q = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
    embeddings_a = model.encode(answers, convert_to_tensor=True, normalize_embeddings=True)

    # Chord vector from query to answer
    delta_a = embeddings_a - embeddings_q  # (N, D)

    # Baseline: how similar are q and a without any prompt?
    sim_qa = cos(embeddings_q, embeddings_a)  # (N,)

    # select prompts to use
    if options.use_lang_specific_prompts:
        print(f"Trying to resolve lang specific prompts with {data_name} {lang}")
        if data_name == "tatoeba":
            prompts_to_try = get_prompts_tatoeba(lang=lang)
        elif data_name == "webfaq":
            prompts_to_try = get_prompts_webfaq(lang=lang)
        else:
            raise AttributeError(f"{data_name} does not have a lang-specific prompt. Also you should not see this, something's wrong.")
    else:
        prompts_to_try={"arcchallenge": get_prompts_arcchallenge(),
                        "tatoeba": get_prompts_tatoeba(),
                        "summeval-2": get_prompts_summeval(),
                        "webfaq": get_prompts_webfaq()}[data_name]

    print(f"Example: first prompt is {prompts_to_try}")
    # calculate per prompt
    results = {}
    for i, p in enumerate(prompts_to_try):
        # apply template and embed
        prompts_and_queries = [get_detailed_instruct(p, q, template=options.template) for q in queries]
        #print(f"Example of prompt+query to be encoded:\n{prompts_and_queries[0]}")
        embeddings_pq = model.encode(prompts_and_queries, convert_to_tensor=True, normalize_embeddings=True)
        #embeddings_p = model.encode([p for _ in prompts_and_queries],convert_to_tensor=True, normalize_embeddings=True)
        # ^^ TODO add comparison to query side later?

        # Chord vector from query to prompted query
        delta_pq = embeddings_pq - embeddings_q  # (N, D)

        # ---- Metric 1: Angulation toward answer ----
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
        #parallel_fraction = parallel_magnitude / displacement if displacement > 0 else 0.0

        # sanity check: these should equal (simply from pythagorean theorem)
        # if not: check values of deltas and displacement, one of these might be 0.0
        # removed this because the +1e-10 thing has an impact: saving both and checking later!
        #assert torch.allclose(parallel_fraction, chord_sim, atol=1e-4), \
        #    f"Parallel fraction and chord sim do not match: {parallel_fraction} != {chord_sim}"

        def stats(t):
            """Extract summary statistics"""
            arr = t.detach().cpu().numpy().reshape(-1)
            return {"mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "q25": float(np.percentile(arr, 25)),
                    "q75": float(np.percentile(arr, 75))}

        results[f"prompt{i}"] = {
            "prompt_text": p,
            "chord_similarity":     stats(chord_sim),           # angulation (same as par. fraction)
            "sim_q_a":              stats(sim_qa),              # baseline similarity
            "sim_pq_a":             stats(sim_pqa),             # prompted similarity
            "sim_improvement":      stats(sim_improvement),     # delta
            "displacement":         stats(displacement),        # how far prompt moved q
            "parallel_magnitude":   stats(parallel_magnitude),  # movement toward answer (signed)
            "orthogonal_magnitude": stats(orthogonal_magnitude),# movement sideways
            "parallel_fraction":    stats(parallel_fraction),   # fraction toward answer
        }

    model_name_safe = model_name.replace("/", "__")
    specific_prompts = "_specific_prompts" if options.use_lang_specific_prompts else ""
    save_path = f"{options.save_prefix}/{model_name_safe}/{data_name_for_saving}{specific_prompts if lang is not None else ''}/{options.split}/{options.template}_template"
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/prompt_geometry.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__=="__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # run "unit test"
            unit_test()
            sys.exit()
        else:
            pass
            #print("Usage: no params (default: prompt comparison) OR one param: \
            #    test: unit test, answer/A: answer comparison, prompt/P: prompt comparison.")
    # parse arguments
    options = parser.parse_args()
    # set default values for data and model
    if options.data_name is None:
        if options.use_lang_specific_prompts:
            options.data_name = [
                            "tatoeba:fin-eng",
                            "tatoeba:fra-eng",
                            "webfaq:deu",
                            "webfaq:eng"]
        else:
            options.data_name = ["arcchallenge",
                            "summeval-2",
                            "tatoeba:fin-eng",
                            "tatoeba:fra-eng",
                            "webfaq:deu",
                            "webfaq:eng"]
    elif isinstance(options.data_name, str):
        options.data_name = [options.data_name]
    if options.model_name is None:
        options.model_name = ["BAAI/bge-m3",
              "Qwen/Qwen3-Embedding-0.6B",
              "intfloat/multilingual-e5-small",
              "intfloat/multilingual-e5-large-instruct",
              "minishlab/potion-base-8M",
              "google/embeddinggemma-300m"]
    elif isinstance(options.model_name, str):
        options.model_name = [options.model_name]

    # loop over models and datasets
    for d in options.data_name:
        for m in options.model_name:
            print(f"Starting model {m} on {d}")
            looper(m, d, options)
