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



# --------------------------------------RETRIEVAL-------------------------------------- #
def download_webfaq_from_hub(lang=None, **kwargs):
    report("Downloading WebFAQRetrieval from hf-hub")
    assert lang is not None, "Language must be defined for WebFAQ, give as mteb/WebFAQRetrieval:\{lang\}"
    split_to_select = "test" # no others exist
    corpus = datasets.load_dataset("mteb/WebFAQRetrieval", f"{lang}-corpus", revision="f64f483ad0f31d2e78209d524c14a4a867965959")
    qrels = datasets.load_dataset("mteb/WebFAQRetrieval", f"{lang}-qrels", revision="f64f483ad0f31d2e78209d524c14a4a867965959")
    queries = datasets.load_dataset("mteb/WebFAQRetrieval", f"{lang}-queries", revision="f64f483ad0f31d2e78209d524c14a4a867965959")
    return corpus[split_to_select], \
           queries[split_to_select], \
           qrels[split_to_select].rename_column("query-id","query_id").rename_column("corpus-id", "corpus_id")

def download_webfaq(lang=None, split_to_select="test", downsample=False,**kwargs):
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

def download_squad_from_hub(split_to_select="test", **kwargs):
    report(f"Downloading SQuAD ({split_to_select}) from hf-hub")
    ds = datasets.load_dataset("rajpurkar/squad", split=split_to_select)
    corpus = ds["test"]["context"]
    queries = ds["test"]["question"]
    qrels_dict = {k:k for k in range(len(corpus))}
    return corpus, queries, qrels_map


def download_arcchallenge_from_hub(**kwargs):
    report("Downloading ARCChallenge from the hf-hub")
    split_to_select = "test"   # this is the only choice
    # download all 3 files for retrieval
    corpus = datasets.load_dataset("mteb/ARCChallenge", "corpus", revision="61b42fe57d9a44e30f47b9b878b664a95472ec80")
    qrels = datasets.load_dataset("mteb/ARCChallenge", "qrels", revision="61b42fe57d9a44e30f47b9b878b664a95472ec80")
    queries = datasets.load_dataset("mteb/ARCChallenge", "queries", revision="61b42fe57d9a44e30f47b9b878b664a95472ec80")
    return corpus[split_to_select], \
           queries[split_to_select], \
           qrels[split_to_select].rename_column("query-id","query_id").rename_column("corpus-id", "corpus_id")

def download_arcchallenge(split_to_select="test", downsample=False, **kwargs):
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


def download_nq_from_hub(**kwargs):
    report("Downloading Natural Questions from the hf-hub")
    split_to_select = "test"   # this is the only choice
    # download all 3 files for retrieval
    corpus = datasets.load_dataset("mteb/nq", "corpus", revision="b774495ed302d8c44a3a7ea25c90dbce03968f31")
    qrels = datasets.load_dataset("mteb/nq", "default", revision="b774495ed302d8c44a3a7ea25c90dbce03968f31")
    queries = datasets.load_dataset("mteb/nq", "queries", revision="b774495ed302d8c44a3a7ea25c90dbce03968f31")
    return corpus[split_to_select], \
           queries[split_to_select], \
           qrels[split_to_select].rename_column("query-id","query_id").rename_column("corpus-id", "corpus_id")

# -------------------------------------CLASSIFICATION------------------------------------- #

def download_multihate_from_hub(lang=None, split_to_select="test", **kwargs):
    report(f"Downloading mteb/multi-hatecheck {lang=}, {split_to_select=} from hf-hub")
    assert split_to_select == "test", "Only test split for mteb/multi-hatecheck"
    assert lang is not None, "Must give lang for mteb/multi-hatecheck, --data_name=mteb/multi-hatecheck:\{lang\}"
    ds = datasets.load_dataset("mteb/multi-hatecheck", split="test", revision="8f95949846bb9e33c6aaf730ccfdb8fe6bcfb7a9")
    # select lang
    ds_filtered = ds.filter(lambda row: row["lang"] == lang)
    del ds
    # classification functions similarly to clustering, hence only return corpus and labels
    return ds_filtered["text"], None, ds_filtered["is_hateful"]


# -------------------------------------BITEXT MINING------------------------------------- #

def download_tatoeba_from_hub(lang=None, split_to_select="test", **kwargs):
    report(f"Downloading Tatoeba:{lang} ({split_to_select}) from hf-hub")
    assert lang is not None, f"{lang=} give a language"
    ds = datasets.load_dataset("mteb/tatoeba-bitext-mining", lang, revision="69e8f12da6e31d59addadda9a9c8a2e601a0e282")
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["sentence1"]}) # non-english
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["sentence2"]}) # english
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    return corpus, queries, qrels_map

def download_tatoeba(lang=None, split_to_select="test", downsample=False,**kwargs):
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


# ------------------------------------SUMMARIZATION------------------------------------ #

def download_summeval_from_hub(split_to_select = "test", **kwargs):
    report(f"Downloading SummEvalSummarization.v2 from hub")
    assert split_to_select=="test", "Only test split available for Summeval"
    ds = datasets.load_dataset("mteb/summeval", revision="cda12ad7615edc362dbf25a00fdd61d3b1eaf93c")
    queries = []
    corpus = []
    scores = []
    for line in ds[split_to_select]:
        for machine_summary, associated_score in zip(line["machine_summaries"], line["relevance"]):
            queries.append(line["human_summaries"][random]) # this is a list: summarisation eval works like this
            corpus.append(machine_summary)
            scores.append(associated_score)
    return corpus, queries, scores


# this test-version was formatted as retrieval, above the "correct STS style with scores"
def download_summeval(split_to_select = "test", downsample=False, **kwargs):
    report(f"Downloading summeval ({split_to_select}) from local")
    # Summeval has been preprocessed by selecting the most relevant summary
    ds = datasets.load_from_disk(path_to_data+"summeval-2")
    if downsample:
        ds[split_to_select] = downsample_with_seed(ds[split_to_select], rows=downsample)
    corpus = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["summary"]})
    queries = datasets.Dataset.from_dict({"_id": range(len(ds[split_to_select])),
                                         "text":ds[split_to_select]["text"]})
    qrels_map = {k:k for k in range(len(corpus))} # they are in the same order
    del ds
    # queries = original text, get the prompt "Summarize the given paragraph into a short paragraph."
    # corpus = the summaries
    return corpus, queries, qrels_map

# ---------------------------------PAIR CLASSIFICATION---------------------------------- #

def download_rte_from_hub(split_to_select="test", lang=None, **kwargs):
    report(f"Downloading RTE3 ({split_to_select}, {lang}) from hub")
    assert lang is not None, "Give language with RTE_ --data_name=mteb/RTE3:lang"
    assert split_to_select == "test", "Only test split available for RTE3"
    ds = datasets.load_dataset("mteb/RTE3", lang, revision="54ea0052267265f4906dd77b0a3d041d301a5ee6")[split_to_select]
    # RTE can be thought of as both STS and classification
    # since RTE's entailment and mteb abstask prompt hint to STS
    # it will be treated as such (despite the name of label-column)
    return ds["sentence1"], ds["sentence2"], ds["labels"]


# -----------------------------------------STS------------------------------------------ #

def download_stsbench_from_hub(split_to_select="test", **kwargs):
    print(f"Donwnloading mteb/stsbenchmark-sts ({split_to_select})")
    ds = datasets.load_dataset("mteb/stsbenchmark-sts", revision="b0fddb56ed78048fa8b90373c8a3cfc37b684831")[split_to_select]
    # in sts-format tasks, instead of qrels, we get a score
    return ds["sentence2"], ds["sentence1"], ds["score"]



# --------------------------------------CLUSTERING-------------------------------------- #
def download_redditclustering_from_hub(split_to_select="test", subsplit=0, **kwargs):
    report(f"Downloading mteb/reddit-clustering ('{split_to_select}') with index {subsplit}")
    assert split_to_select == "test", "Only test split available for mteb/reddit-clustering"
    ds = datasets.load_dataset("mteb/reddit-clustering", revision="24640382cdbf8abc73003fb0fa6d111a705499eb")["test"][subsplit]
    # this is a clustering task: no queries, corpus works as both corpus and queries
    return ds["sentences"], None, ds["labels"]

def download_arxivclustering_from_hub(split_to_select="test", subsplit=0, **kwargs):
    report(f"Downloading mteb/arxiv-clustering-s2s ('{split_to_select}') with index {subsplit}")
    assert split_to_select == "test", "Only test split available for mteb/arxiv-clustering-s2s"
    ds = datasets.load_dataset("mteb/arxiv-clustering-s2s", revision="f910caf1a6075f7329cdf8c1a6135696f37dbd53")["test"][subsplit]
    # this is a clustering task: no queries, corpus works as both corpus and queries
    return ds["sentences"], None, ds["labels"]

def download_dataset(data_name, **kwargs):
    # first, local installs (for smaller test runs)
    if data_name.lower() == "arcchallenge":
        return download_arcchallenge(**kwargs)
    if data_name.lower() == "summeval" or data_name.lower() == "summeval-2":
        return download_summeval(**kwargs)
    if data_name.lower() == "tatoeba":
        return download_tatoeba(**kwargs)
    if data_name.lower() == "webfaq":
        return download_webfaq(**kwargs)
    # actual downloads
    if data_name == "mteb/ARCChallenge":
        return download_arcchallenge_from_hub(**kwargs)
    if data_name == "mteb/WebFAQRetrieval":
        return download_webfaq_from_hub(**kwargs)
    if data_name == "mteb/tatoeba-bitext-mining":
        return download_tatoeba_from_hub(**kwargs)
    if data_name == "mteb/stsbenchmark-sts":
        return download_stsbench_from_hub(**kwargs)
    if data_name == "mteb/reddit-clustering":
        return download_redditclustering_from_hub(**kwargs)
    if data_name == "mteb/arxiv-clustering-s2s":
        return download_arxivclustering_from_hub(**kwargs)
    if data_name == "mteb/multi-hatecheck":
        return download_multihate_from_hub(**kwargs)
    if data_name == "mteb/RTE3":
        return download_rte_from_hub(**kwargs)
    if data_name == "rajpurkar/squad":
        return download_squad_from_hub(**kwargs)
    if data_name == "mteb/nq":
        return download_nq_from_hub(**kwargs)
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