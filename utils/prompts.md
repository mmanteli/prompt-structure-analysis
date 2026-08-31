# Prompt selection

Here we describe how we chose the set of prompt in our experiments.
First, we select the "fallback" prompts from ``mteb/abstasks/<task category>.py``, when available.
Then, we select the task specific prompts when available.
After this, we generate prompts with Claude, and select 2-6 per task category depending on the number of "official" prompts available.


## Abstask fallback prompts

Some task categories has a fall-back, asbtask prompt. 

### [Clustering abstask](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/clustering.py#L148)
    "Identify categories in user passages."  
### [Classification abstask](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/classification.py#L119)
    "Classify user passages."
### [Pair Classification abstask](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/pair_classification.py#L53)
    "Retrieve text that are semantically similar to the given text."
### [Retrieval abstask](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/retrieval.py#L101)
    "Retrieve text based on user query."
### [STS abstask](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/sts.py#L78)
    "Retrieve semantically similar text."
### [Summarization abstask](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/text/summarization.py#L51)
    "Given a news summary, retrieve other semantically similar summaries."
### [Bitext-mining abstask](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/abstasks/text/bitext_mining.py#L54)
    "Retrieve parallel sentences."

## Task-specific prompts

Some tasks have their own prompts.

### [ARCChallenge](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/retrieval/eng/arc_challenge_retrieval.py#L42)
    "Retrieve the answer to the question.",
### [RedditClustering](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/clustering/eng/reddit_clustering.py#L50)
    "Identify the topic or theme of Reddit posts based on the titles"
### [Arxiv clustring](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/clustering/eng/arxiv_clustering_s2s.py#L37)
    "Identify the main and secondary category of Arxiv papers based on the titles"

## Per-task generated prompts

### Retrieval

Generated with Claude. 

    "Given a question, retrieve the passage that best answers it",
    "Retrieve",
    "Find the most relevant passage that directly answers the question",
    "Given a question, find a related document"

### Bitext mining (these will have the correct lang added to them, or "in {lang}" removed)

Bitext-mining has no specific prompt for Tatoeba, and no abstask prompt. These are generated with Claude.

    f"Retrieve the corresponding translation in {lang}.",
    f"Given an English sentence, find its translation in {lang}.",
    f"Retrieve parellel sentences in {lang}",
    f"Translate to {lang}.",
    "Find a sentence that has similar meaning.",

### Summarization

Summarisation also lacks both specific prompts for SummEvalSummarization.v2 and the abstask promp. These are generated with Claude.

    "Summarize the given paragraph into a short paragraph.",
    "Match the text to the short abstract.",
    "Retrieve a related summary."


### Pair-classification

RTE3 has no specific prompt, but pair-classification has the abs prompt.

    "Classify texts as entailment or contradiction."
    "Determine whether the following two passages are contradictory, entailed, or neutral."
    "Classify based on semantic similarity."

### Clustering and classification

Generated with Claude.

    "Classify the following text into a meaningful cluster based on its content.",
    "Group documents related to the same topic.",
    "Cluster."

### STS

Generated with Claude.

    "Retrieve a similar passage.",
    "Represent this sentence for a natural language understanding task.",
    "Group passages based on semantic similarity."

## No corresponding task

Examples taken from [here](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/pair_classification/eng/sprint_duplicate_questions_pc.py#L30), [here](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/retrieval/code/coreb_retrieval.py#L52) and [here](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/tasks/reranking/eng/ecommerce_product_relevance_reranking.py#L36), as well as [here](https://huggingface.co/datasets/mteb/amazon_reviews_multi), 

    "Retrieve duplicate questions from the forum.",
    "Retrieve the most relevant problem description for the given code implementation."
    "Rerank products by relevance to the e-commerce query.",
    "Classify the sentiment of the given review as positive or negative",

Prompt that are not about embedding tasks

    "Generate a creative story based on the prompt",
    "Extract all named entities from the text and categorise them as PERSON, LOCATION, or ORGANIZATION",


## General and vague prompts

Vague prompts generated with Claude.

    "Search for related information",
    "Find documents related to the topic",
    "Find something useful",
    "Retrieve the passage that contains overlapping vocabulary with the source sentence",

## Keysmash

Intentionally nonsense prompts.

    "asdfjkl qpwoeiru zxcvbnm",
    "hgJKSbf oiawnef LKJHDS kdjfbs",
    "!!!! ??? ### @@@",

# Finetuning prompt

    "Given a question, retrieve Wikipedia passages that answer the question."

