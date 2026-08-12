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
           qrels[split_to_select], \
           queries[split_to_select], \
           create_binary_relevance_map_arcchallenge(qrels, split=split_to_select)

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
    return corpus, None, queries, qrels_map



def download_tatoeba_from_hub(lang, split_to_select="test", downsample=False):
    report(f"Downloading Tatoeba:{lang} ({split_to_select}) from hf-hub")
    ds = datasets.load_dataset("mteb/tatoeba-bitext-mining", lang)
    if downsample:
        ds[split_to_select] = downsample_with_seed(ds[split_to_select], rows=downsample)
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["sentence1"]}) # non-english
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["sentence2"]}) # english
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, None, queries, qrels_map

def download_tatoeba(lang, split_to_select="test", downsample=False):
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
    return corpus, None, queries, qrels_map

def download_webfaq(lang, split_to_select="test", downsample=False):
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
    return corpus, None, queries, qrels_map

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
    return corpus, None, queries, qrels_map


def download_dataset(data_name, split, lang, local=True, num_examples=5000):
    if local:
        if data_name.lower() == "arcchallenge":
            return download_arcchallenge(split_to_select=split, downsample=num_examples)
        if data_name.lower() == "summeval" or data_name.lower() == "summeval-2":
            return download_summeval(split_to_select=split, downsample=num_examples)
        assert lang is not None, f"{data_name} requires setting lang, now {lang=}"
        if data_name.lower() == "tatoeba":
            return download_tatoeba(lang, split_to_select=split, downsample=num_examples)
        if data_name.lower() == "webfaq":
            return download_webfaq(lang, split_to_select=split, downsample=num_examples)
    else:
        if data_name == "mteb/ARCChallenge" or data_name.lower() == "arcchallenge":
            return download_arcchallenge_from_hub()
        if data_name == "Tatoeba":
            return download_tatoeba_from_hub(lang, split_to_select=split, downsample=num_examples)
        raise NotImplementedError(f"Unable to download a dataset with arguments {data_name=} {split=} {lang=}, {local=}")

if __name__ == "__main__":
    data_name="mteb/ARCChallenge"
    lang=None
    split = "fit"
    local=False
    corpus, qrels, queries, qrels_dict = download_dataset(data_name, split, lang, local=local, num_examples=40)
    print(corpus)
    print(queries)
    print(qrels_dict)
    print("")
    print(queries[qrels_dict])
    print(corpus[5])