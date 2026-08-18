import random
import datasets
import os
seed = 42
random.seed(seed)

path_to_data=os.environ["DATAPATH"]
if path_to_data[-1] != "/":
    path_to_data += "/"

def report(msg):
    # for quick flushing
    print(f'{msg}', flush=True)

def downsample_with_seed(ds, rows=5000):
    print(ds)
    if len(ds) > rows:
        report("Downsampling the dataset...")
        # generate a list of random indices
        random_indices = random.sample(range(len(ds)), rows)
        # select the rows using the random indices
        ds = ds.select(random_indices)
    return ds

def sanity_check_arcchallenge(corpus, qrels, queries, split="test"):
    report("---------------SANITY CHECK----------------")
    report("--see that the given example makes sense---")
    report(queries[split][0])
    query_id = queries[split][0]["_id"]
    qrel = qrels.filter(lambda example: example["query-id"]==query_id)
    assert len(qrel) == 1, "More than one relevant document found. Should not be possible for binary relevance."
    corpus_id = qrels[split][0]["corpus-id"]
    target = corpus.filter(lambda example: example["_id"] == corpus_id)
    report(target[split][0])
    report("-------------SANITY CHECK END--------------")


def create_binary_relevance_map_arcchallenge(qrels, split="test"):
    qrels_dict = {}
    for d in qrels[split]:
        q = d["query-id"]
        c = d['corpus-id']
        if d['score'] == 1:
            qrels_dict[q] = c
    return qrels_dict

def download_arcchallenge_from_hub():
    report("Downloading ARCChallenge from the hf-hub")
    split_to_select = "test"   # this is the only choice
    # download all 3 files for retrieval
    corpus = datasets.load_dataset("mteb/ARCChallenge", "corpus")
    qrels = datasets.load_dataset("mteb/ARCChallenge", "qrels")
    queries = datasets.load_dataset("mteb/ARCChallenge", "queries")
    sanity_check_arcchallenge(corpus, qrels, queries)
    return corpus[split_to_select], \
           queries[split_to_select], \
           qrels[split_to_select].rename_column("query-id","query_id").rename_column("corpus-id", "corpus_id")

def download_arcchallenge(split_to_select="test", downsample=False):
    report(f"Downloading ARCChallenge ({split_to_select}) from local")
    ds = datasets.load_from_disk(path_to_data+"arcchallenge")
    if downsample:
        ds[split_to_select] = downsample_with_seed(ds[split_to_select], rows=downsample)
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["document"]})
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["query"]})
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, queries, qrels_map



def download_tatoeba_from_hub(lang=None, split_to_select="test", downsample=False):
    report(f"Downloading Tatoeba:{lang} ({split_to_select}) from hf-hub")
    assert lang is not None, f"{lang=} give a language"
    ds = datasets.load_dataset("mteb/tatoeba-bitext-mining", lang)
    if downsample:
        ds[split_to_select] = downsample_with_seed(ds[split_to_select], rows=downsample)
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["sentence1"]}) # non-english
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["sentence2"]}) # english
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, queries, qrels_map

def download_tatoeba(lang=None, split_to_select="test", downsample=False):
    report(f"Downloading Tatoeba:{lang} ({split_to_select}) from local")
    lang_ = "en-" + lang.split("-")[0][:2]
    if lang_ == "en-bu":   # bad, i know
        lang = "en-bg"
    ds = datasets.load_from_disk(f"{path_to_data}tatoeba:{lang_}")
    if downsample:
        ds[split_to_select] = downsample_with_seed(ds[split_to_select], rows=downsample)
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["non_english"]}) # non-english
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["english"]})  # english
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, queries, qrels_map

def download_webfaq(lang=None, split_to_select="test", downsample=False):
    report(f"Downloading webfaq:{lang} ({split_to_select}) from local")
    ds = datasets.load_from_disk(f"{path_to_data}web-faq-bitext:{lang}")
    if downsample:
        ds[split_to_select] = downsample_with_seed(ds[split_to_select], rows=downsample)
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["answer2"]}) # 2's contain the target lang
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["question2"]})
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, queries, qrels_map

def download_summeval(split_to_select = "test", downsample=False):
    report(f"Downloading summeval ({split_to_select}) from local")
    #'text','summary'
    ds = datasets.load_from_disk(path_to_data+"summeval-2")
    if downsample:
        ds[split_to_select] = downsample_with_seed(ds[split_to_select], rows=downsample)
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["text"]})
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["summary"]})
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, queries, qrels_map

def download_stsbench(split_to_select="test", downsample=False):
    print(f"Donwnloading mteb/stsbenchmark-sts ({split_to_select})")
    ds = datasets.load_dataset("mteb/stsbenchmark-sts")[split_to_select]
    return ds["sentence2"], ds["sentence1"], ds["score"]

def download_redditclustering(split_to_select="test", downsample=False, subsplit=0):
    report(f"Downloading mteb/reddit-clustering ('test') with index {subsplit}")
    ds = datasets.load_dataset("mteb/reddit-clustering")["test"][subsplit]
    return ds["sentences"], None, ds["labels"]

def download_dataset(data_name, **kwargs):
    if data_name.lower() == "arcchallenge":
        return download_arcchallenge(**kwargs)
    if data_name.lower() == "summeval" or data_name.lower() == "summeval-2":
        return download_summeval(**kwargs)
    if data_name.lower() == "tatoeba":
        return download_tatoeba(**kwargs)
    if data_name.lower() == "webfaq":
        return download_webfaq(**kwargs)
    if data_name == "mteb/ARCChallenge" or data_name.lower() == "arcchallenge":
        return download_arcchallenge_from_hub(**kwargs)
    if data_name == "mteb/tatoeba-bitext-mining":
        return download_tatoeba_from_hub(**kwargs)
    if data_name == "mteb/stsbenchmark-sts":
        return download_stsbench(**kwargs)
    if data_name == "mteb/reddit-clustering":
        return download_redditclustering(**kwargs)
    raise NotImplementedError(f"Unable to download a dataset with arguments {data_name=} {kwargs}")

if __name__ == "__main__":
    data_name="mteb/ARCChallenge"
    corpus, queries, qrels = download_dataset(data_name)
    assert isinstance(corpus, datasets.Dataset)
    assert isinstance(queries, datasets.Dataset)
    assert isinstance(qrels, datasets.Dataset) or isinstance(qrels, dict)
    #print(f"{corpus=}")
    #print(f"{queries=}")
    #print(f"{qrels=}")
    #print("")
    print(qrels[0])
    print(queries[0])
    print(corpus[2])
    print("\n\n")

    data_name="mteb/reddit-clustering"
    corpus, _, labels = download_dataset(data_name, subsplit=0)
    assert isinstance(corpus, list), f"{type(corpus)=}"
    assert isinstance(labels, list), f"{type(labels)=}"
    print(f"{corpus[0]=}")
    print(f"{labels[0]=}")
    print("\n\n")

    data_name="mteb/stsbenchmark-sts"
    corpus, queries, scores = download_dataset(data_name)
    print(corpus[0])
    print(queries[0])
    print(scores[0])
    print("\n\n")

    data_name="mteb/tatoeba-bitext-mining"
    lang="fin-eng"
    corpus, queries, qrels = download_dataset(data_name, lang=lang)
    print(corpus[0])
    print(queries[0])