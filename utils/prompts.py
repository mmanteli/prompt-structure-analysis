from iso639 import languages

def get_prompts_arcchallenge():
    return [
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

def get_prompts_tatoeba(lang=None):
    if lang is None:
        return [
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
    # map lang abreviation to text
    if "-" in lang:
        l, eng_ = lang.split("-")
        if eng_ != "eng":
            raise AttributeError(f"Language to tatoeba prompt given incorrectly, is {lang}, should be fra-eng, deu-eng")
    else:
        l = lang   # assuming parsed already
    try:
        lang = languages.get(part2t=l).name
    except KeyError as err:
        raise KeyError(f"Cannot resolve {l} with iso639 in tatoeba prompts") from err
    return [
        # Well-Fitting Instructions
        f"Retrieve the corresponding translation in {lang}",
        f"Find the equivalent sentence in {lang}",
        f"Retrieve a parallel sentence in {lang}",
        f"Given an English sentence, find its translation in {lang}",
        f"Match the English sentence to its translated counterpart in {lang}",
        f"Identify the sentence in {lang} that is a direct translation of the given English source sentence",
        "Retrieve",
        f"Find translation in {lang}",
        f"Given the following English text, retrieve the passage that is a faithful and accurate translation of it into {lang}",
        f"Translate to {lang}",

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

def get_prompts_summeval():
    return [
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


def get_prompts_webfaq(lang=None):
    if lang is None:
        return [
        # Good, on-task prompts (varying length and specificity)
        "Given a question, retrieve the passage that correctly answers it",
        "Retrieve the passage that answers the following question",
        "Find the document that contains the answer to this question",
        "Given a question, retrieve the most relevant passage that directly provides the answer",
        "Retrieve a passage that answers the given question accurately and completely",
        "Find the best-matching answer passage for the following question",
        "Given a question, find the paragraph from which the answer can be extracted",
        "Given a question, retrieve the passage that answers it",
        "Retrieve the passage containing the answer to the question",
        "Find the answer passage for the given question",
        "Retrieve the most relevant passage that answers the query",
        "Answer the question",

        # Slightly off or vague prompts
        "Find documents related to the topic of the question",
        "Retrieve a relevant passage",
        "Given a query, find a related document",
        "Search for information",
        "Look up the answer to the question",
        "Find something relevant",
        "Get a passage that might be useful",
        "Retrieve",

        # Wrong task prompts
        "Translate the following sentence from French into English",
        "Summarize the given paragraph into a few sentences",
        "Given two sentences, determine if they are semantically similar",
        "Classify the sentiment of the given review as positive or negative",
        "Generate a fluent continuation of the following paragraph",
        "Given a premise and a hypothesis, determine the entailment relationship",
        "Retrieve duplicate questions from the forum",

        # Nonsense / keysmash prompts
        "asdfjkl qpwoeiru zxcvbnm",
        "hgJKSbf oiawnef LKJHDS kdjfbs",
        "!!!! ??? ### @@@",
    ]
    try:
        lang = languages.get(part2t=lang).name
    except KeyError as err:
        raise KeyError(f"Cannot resolve {lang} with iso639 in webfaq prompts.") from err
    return [
        # Good, on-task prompts (varying length and specificity)
        f"Given a question in {lang}, retrieve the passage that correctly answers it",
        f"Retrieve the passage that answers the following {lang} question",
        f"Find the document that contains the answer to this {lang}-language question",
        f"Given a question written in {lang}, retrieve the most relevant passage that directly provides the answer",
        f"Retrieve a passage in {lang} that answers the given question accurately and completely",
        f"Find the best-matching answer passage for the following question in {lang}",
        f"Given a {lang} question, find the paragraph from which the answer can be extracted",
        "Given a question, retrieve the passage that answers it",
        "Retrieve the passage containing the answer to the question",
        "Find the answer passage for the given question",
        "Retrieve the most relevant passage that answers the query",
        "Answer the question",

        # Slightly off or vague prompts
        "Find documents related to the topic of the question",
        "Retrieve a relevant passage",
        "Given a query, find a related document",
        f"Search for information in {lang}",
        "Look up the answer to the question",
        "Find something relevant",
        "Get a passage that might be useful",
        "Retrieve",

        # Wrong task prompts
        f"Translate the following sentence from {lang} into English",
        "Summarize the given paragraph into a few sentences",
        "Given two sentences, determine if they are semantically similar",
        "Classify the sentiment of the given review as positive or negative",
        "Generate a fluent continuation of the following paragraph",
        "Given a premise and a hypothesis, determine the entailment relationship",
        "Retrieve duplicate questions from the forum",

        # Nonsense / keysmash prompts
        "asdfjkl qpwoeiru zxcvbnm",
        "hgJKSbf oiawnef LKJHDS kdjfbs",
        "!!!! ??? ### @@@",
    ]


def get_prompts_sts():
    return [
        # Good, on-task prompts (varying length)
        "Represent this sentence for measuring semantic similarity with another sentence.",
        "Encode this sentence so that sentences with similar meanings are mapped to nearby points in the embedding space.",
        "Generate a representation of the following sentence that captures its core meaning for comparison with other sentences.",
        "Represent this sentence such that its similarity to other sentences can be assessed on a continuous scale from unrelated to equivalent.",
        "Embed the following sentence for the task of determining how closely it paraphrases another sentence.",
        "Retrieve semantically similar sentences.",
        "Represent this text for evaluating the degree of semantic equivalence between sentence pairs.",
        "Given the following sentence, produce an embedding useful for comparing meaning across sentences.",
        "Classify how similar the meaning of this sentence is to another sentence.",
        "Represent this sentence for a natural language understanding task.",

        # Slightly off or vague prompts
        "Find documents related to the topic",
        "Retrieve a similar passage",
        "Search for related information",
        "Look up the answer",
        "Find something useful",
        "Get the relevant text",
        "Retrieve",

        # Wrong task prompts
        "Translate the following sentence into French",
        "Summarize the given paragraph into two sentences",
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


def get_prompts_redditclustering():
    return [
        "Identify the main topic of this text for grouping similar documents together.",
        "Represent this sentence for clustering with other semantically similar sentences.",
        "Summarize the core theme of the following passage so it can be grouped with related texts.",
        "Generate an embedding of this document that captures its primary subject for clustering purposes.",
        "What category or theme does this text belong to?",
        "Classify the following text into a meaningful cluster based on its content.",
        "Represent this paragraph in a way that highlights its topical similarity to other paragraphs.",
        "Determine the underlying subject matter of this text for organizing it alongside similar content.",
        "Embed this text so that documents discussing the same topic are placed near each other.",
        "Extract the key concept from this text to facilitate grouping it with thematically related documents.",
        
        # Slightly off or vague prompts
        "Find documents related to the topic",
        "Retrieve a similar passage",
        "Search for related information",
        "Look up the answer",
        "Find something useful",
        "Get the relevant text",
        "Retrieve",

        # Wrong task prompts
        "Translate the following sentence into French",
        "Summarize the given paragraph into two sentences",
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

def get_prompts(data_name, **kwargs):
    if "arcchallenge" in data_name.lower():
        return get_prompts_arcchallenge()
    if "tatoeba" in data_name.lower():
        return get_prompts_tatoeba(**kwargs)
    if "reddit" in data_name.lower():
        return get_prompts_redditclustering()
    if "webfaq" in data_name.lower():
        return get_prompts_webfaq(**kwargs)
    if "summeval" in data_name.lower():
        return get_prompts_summeval()
    if "sts" in data_name.lower():
        return get_prompts_sts()
    raise NotImplementedError(f"Prompts for {data_name} not implemented or cannot resolve data name")

def get_detailed_instruct(prompt, query, template="Instruct-Query"):
    """Given prompt, query, and template, return a filled template"""
    if prompt == "NO_PROMPT":
        # for "NO_PROMPT" return only the query (a baseline for metrics)
        return query
    # for "EMPTY" return empty template (baseline for including the word "Instruct")
    if template == "simple":
        return f"{prompt if prompt != 'EMPTY' else ''}. {query}"
    if template == "Instruct-Query":
        return f"Instruct: {prompt if prompt != 'EMPTY' else ''}\nQuery: {query}"
    raise NotImplementedError(f"{template=} not implemented")