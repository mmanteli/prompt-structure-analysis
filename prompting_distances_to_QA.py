from sentence_transformers import SentenceTransformer
import datasets
import os
import json
import numpy as np
import torch
from sklearn.preprocessing import normalize
import sys
cos = torch.nn.CosineSimilarity()
os.environ["HF_TOKEN"] = "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"




"""prompts_to_try = ["Given a web search query, retrieve relevant passages that answer the query",
                "Retrieve relevant passages that answer the query",
                "Retrieve relevant passages",
                "Find relevent documents",
                "Find relevant documents that answer the query",
                "Answer the question",
                "Retrieve a corresponding summary",
                "Translate the sentence", 
                "This is a completely random sentence",
                "DKJfbs goinegja ebgkdgn"]"""
prompts_arcchallenge = [
    # Good, on-task prompts (varying length)
    "Given a science question, retrieve the passage that best answers it",
    "Retrieve the passage that correctly answers the multiple choice science question",
    "Find the document that contains the answer to the given science question",
    "Given a challenging science question, retrieve a passage that provides the correct answer and explains the underlying concept",
    "Retrieve passages that answer elementary and middle school science questions spanning biology, chemistry, physics, and earth science",
    "Find the most relevant scientific passage that directly answers the question",
    "Given a multiple choice question about science, retrieve the passage most likely to contain the correct answer",
    "Retrieve a passage answering the question",
    "Find relevant science passages",
    "Answer the science question",
    "Retrieve the correct answer",

    # Slightly off or vague prompts
    "Find documents related to the topic",
    "Retrieve a relevant passage",
    "Given a question, find a related document",
    "Search for information about the query",
    "Look up the answer",
    "Find something useful",
    "Get the relevant text",
    "Retrieve",

    # Wrong task prompts
    "Translate the following sentence into French",
    "Summarize the given paragraph into two sentences",
    "Given two sentences, determine if they are semantically similar",
    "Classify the sentiment of the given review as positive or negative",
    "Generate a creative story based on the prompt",
    "Retrieve duplicate questions from the forum",
    "Find the most similar product review",
    "Given a code snippet, retrieve the documentation",

    # Nonsense / keysmash prompts
    "asdfjkl qpwoeiru zxcvbnm",
    "hgJKSbf oiawnef LKJHDS kdjfbs",
    "!!!! ??? ### @@@",
]

prompts_tatoeba = [
    # Well-Fitting Instructions
    "Retrieve the corresponding translation",
    "Find the equivalent sentence in another language",
    "Retrieve a parallel sentence",
    "Given an English sentence, find its translation",
    "Match the English sentence to its translated counterpart in the target language",
    "Identify the sentence in the target language that is a direct translation of the given English source sentence",
    "Retrieve",
    "Find translation",
    "Given the following English text, retrieve the passage that is a faithful and accurate translation of it into a non-English language",
    "Translate",

    # Related but Suboptimal Instructions
    "Find a sentence that has similar meaning",
    "Retrieve a semantically similar passage",
    "Find a paraphrase of the given sentence",
    "Retrieve the most thematically related passage from the document corpus",
    "Find a sentence that discusses the same topic",
    "Given a query sentence, retrieve passages that are topically relevant",
    "Identify the passage that best captures the intent of the following sentence",
    "Retrieve the passage that contains overlapping vocabulary with the source sentence",

    # Wrong Task Instructions
    "Classify the sentiment of the given review as positive or negative",
    "Summarize the following paragraph in two sentences",
    "Answer the following question based on the provided context",
    "Extract all named entities from the following sentence and label them as PERSON, ORGANIZATION, or LOCATION",
    "Determine whether the following two statements are contradictory, entailed, or neutral",
    "Generate a creative short story based on the following prompt",
    "Given the following code snippet, identify and fix the bug",
    
    # Nonsense / keysmash prompts
    "asdfjkl qpwoeiru zxcvbnm",
    "hgJKSbf oiawnef LKJHDS kdjfbs",
    "!!!! ??? ### @@@",
]

prompts_summeval = [
    # Well-Fitting Instructions
    "Given a summary, retrieve the full document it was derived from",
    "Match the short abstract to the longer document it summarises",
    "Retrieve the document whose key points are captured in the given summary",
    "Given the following brief summary, find the source document that contains the information it describes",
    "Retrieve",
    "Given a condensed version of a text, retrieve the original document from which it was summarised",
    "Identify the passage that the following summary is an abridged version of",

    # Related but Suboptimal Instructions
    "Retrieve a document that discusses the same subject matter",
    "Find a passage that is thematically related to the given text",
    "Retrieve the most topically similar document",
    "Find a passage that covers overlapping key concepts",
    "Given a short text, retrieve a longer passage on the same topic",
    "Identify the document that shares the most content words with the given passage",
    "Retrieve a passage that is semantically close to the given description",
    "Find a document that would be relevant to someone interested in this topic",

    # Wrong Task Instructions, first 3 are the wrong way round
    "Find summary",
    "Retrieve the document that best summarises the given topic",
    "Find the passage that serves as a concise summary of the source text",
    "Translate the following sentence into French",
    "Answer the following question using the provided context",
    "Classify the sentiment of the following customer review",
    "Extract all named entities from the text and categorise them as PERSON, LOCATION, or ORGANIZATION",
    "Determine whether the following two passages are contradictory, entailed, or neutral",
    "Generate a creative short story inspired by the following prompt",
    "Given the following code, identify the bug and suggest a fix",

    # Nonsense / keysmash prompts
    "asdfjkl qpwoeiru zxcvbnm",
    "hgJKSbf oiawnef LKJHDS kdjfbs",
    "!!!! ??? ### @@@",
]


def load_dataset(path):
    if os.path.exists(path):
        return datasets.load_from_disk(path)
    else:
        return datasets.load_dataset(path)

def get_detailed_instruct(prompt: str, query: str) -> str:
    return f'Instruct: {prompt}\nQuery: {query}'

class Testmodel():
    def __init__(self, dim=3):
        self.dim=dim
    def encode(self, text, convert_to_tensor=True, normalize_embeddings=False):
        embedding = np.random.random((len(text), self.dim))
        if normalize_embeddings:
            embedding = normalize(embedding, axis=1)
            assert np.isclose(embedding[0,:].shape, self.dim), f"{embedding[0,:].shape} != {self.dim}"
            assert np.isclose(np.linalg.norm(embedding[0,:]), 1.0), f"Norm after normalizing: {np.linalg.norm(embedding[0,:])}"
        if convert_to_tensor:
            return torch.Tensor(embedding)
        return embedding



def compare_q_p_and_qp(embeddings_pq, embeddings_q, embedding_p, alpha=0.5, d="cosine"):
    """Function that calculates the midpoint between Query and Prompt,
    and compares embedding of template(prompt, query) to that. """
    distances = []
    for q, pq in zip(embeddings_q, embeddings_pq):
        midpoint = torch.lerp(input=q, end=embedding_p, weight=alpha)
        if d=="euclidean":
            distance = torch.dist(pq, midpoint)
        elif d=="cosine":
            distance = cos(pq.reshape(1,-1), midpoint.reshape(1,-1))
        else:
            raise NotImplementedError
        distances.append(distance.detach().cpu().numpy())
    return np.array(distances)

def compare_q_a_and_qp(embeddings_pq, embeddings_q, embeddings_a, alpha=0.5, d="cosine"):
    """Function that calculates the midpoint between Query and Answer/target,
    and compares embedding of template(prompt, query) to that. """
    distances = []
    for q, pq, a in zip(embeddings_q, embeddings_pq, embeddings_a):
        midpoint = torch.lerp(input=q, end=a, weight=alpha)
        if d=="euclidean":
            distance = torch.dist(pq,midpoint)
        elif d=="cosine":
            distance = cos(pq.reshape(1,-1), midpoint.reshape(1,-1))
        else:
            raise NotImplementedError
        distances.append(distance.detach().cpu().numpy())
    return np.array(distances)
def unit_test():
    # first, test that midpoints equal average and that results in 0
    q1, q2 = torch.Tensor(np.random.random((1,5))), torch.Tensor(np.random.random((1,5)))
    p = torch.Tensor(np.random.random((1,5)))
    pq1, pq2 = (q1+p)/2, (q2+p)/2
    assert np.allclose(compare_q_p_and_qp([pq1, pq2], [q1, q2], p, d="euclidean"),0., atol=1e-4), "Test 1: euclidean not passed"
    assert np.allclose(compare_q_p_and_qp([pq1, pq2], [q1, q2], p, d="cosine"), 1., atol=1e-5), f"Test 1: cosine not passed"   
    # increased atol here since rtol is effectively useless because we compare to 0
    # second, that the geometry works with known distances
    q = torch.Tensor([-2,0])
    p = torch.Tensor([2,2])
    pq1 = torch.Tensor([1,-1])
    pq2 = torch.Tensor([-2,1])
    pq3 = torch.Tensor([0,1])
    for i,j in zip(compare_q_p_and_qp([pq1, pq2, pq3], [q,q,q], p, d="euclidean"), (np.sqrt(5), 2, 0)):
        assert np.isclose(i,j), "Test 2: euclidean not passed"
    for i,j in zip(compare_q_p_and_qp([pq1, pq2, pq3], [q,q,q], p, d="cosine"), (-0.7071067811865475, 0.8944271909999159, 1)):
        assert np.isclose(i,j), f"Test 2: cosine not passed: {i}!={j}"
    print("Success: All tests passed.")
    

def looper(model_name, data_name):
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # run "unit test"
            unit_test()
            exit()
        elif sys.argv[1] == "answer" or sys.argv[1] == "A":
            comp = "answer"
        elif sys.argv[1] == "prompt" or sys.argv[1] == "P":
            comp = "prompt"
        else:
            print("Usage: no params (default: prompt comparison) OR one param: test: unit test, answer/A: answer comparison, prompt/P: prompt comparison.")
    else:
        comp="prompt"


    data_path = {"arcchallenge": "/flash/project_462000883/datasets/arcchallenge",
                "tatoeba:fin-eng": "/flash/project_462000883/datasets/tatoeba:en-fi",
                "summeval-2": "/flash/project_462000883/datasets/summeval-2"}[data_name]
    ds = load_dataset(data_path)
    print(ds)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    #model = Testmodel()
    print("Model loaded.")
    columns = {"arcchallenge": ('query', 'document'),
               "tatoeba:fin-eng": ('english', 'non_english'),
               "summeval-2": ('summary','text')}[data_name]

    # embed queries, no need to repeat
    split = "test" if "tatoeba" in data_name else "fit"
    queries = ds[split][columns[0]]
    embeddings_q = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
    if comp=="answer":
        answers = ds[split][columns[1]]
        embeddings_a = model.encode(answers, convert_to_tensor=True, normalize_embeddings=True)

    prompts_to_try = {"arcchallenge": prompts_arcchallenge,
                      "tatoeba:fin-eng": prompts_tatoeba,
                      "summeval-2": prompts_summeval}[data_name]
    # result collecting
    results={}
    for i, p in enumerate(prompts_to_try):
        # embed prompt and template(prompt, query)
        prompts_and_queries = [get_detailed_instruct(p,q) for q in queries]
        embeddings_pq = model.encode(prompts_and_queries, convert_to_tensor=True, normalize_embeddings=True)
        dists = {}
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            if comp=="prompt":
                embedding_p = model.encode([p], convert_to_tensor=True, normalize_embeddings=True)
                distances = compare_q_p_and_qp(embeddings_pq, embeddings_q, embedding_p, alpha=alpha)
                dists[alpha] = {"mean": str(np.mean(distances.reshape(-1))), "std": str(np.std(distances.reshape(-1)))}
            else:
                distances = compare_q_a_and_qp(embeddings_pq, embeddings_q, embeddings_a, alpha=alpha)
                dists[alpha] = {"mean": str(np.mean(distances.reshape(-1))), "std": str(np.std(distances.reshape(-1)))}
        #print(f"Prompt: {p}")
        #print(distances.shape)
        #print(f"{np.mean(distances.reshape(-1))} (+- {np.std(distances.reshape(-1))})")
        #print(dists)
        results[f"prompt{i}"] = dists | {"prompt_text": p}
    #print(results)
    model_name = model_name.replace("/", "__")
    os.makedirs(f"results/{model_name}/{data_name}", exist_ok=True)
    with open(f'results/{model_name}/{data_name}/prompt_data_cosine_query_to_{comp}.json', 'w') as f:
        json.dump(results, f)
    
if __name__=="__main__":
    data_name = "summeval-2"
    models = ["BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B","intfloat/multilingual-e5-small", "intfloat/multilingual-e5-large-instruct", "minishlab/potion-base-8M", "google/embeddinggemma-300m"]
    for m in models:
        print(f"Starting model {m}")
        looper(m, data_name)




