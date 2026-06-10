#!/bin/bash

# first, run the structural analysis
# loopable parameters are: 
# --split=[fit, test], 
# --template=["Instruct-Query", "simple"]
# --use_lang_specific_prompts (include or not)


# second, run the evaluation
# loopable parameters are:
# --model_name = ["BAAI/bge-m3",
#              "Qwen/Qwen3-Embedding-0.6B",
#              "intfloat/multilingual-e5-small",
#              "intfloat/multilingual-e5-large-instruct",
#              "minishlab/potion-base-8M",
#              "google/embeddinggemma-300m"]
# --data_name = ["arcchallenge",
#                "summeval-2",
#                "tatoeba:fin-eng",
#                "tatoeba:fra-eng",
#                "webfaq:deu",
#                "webfaq:eng"]
# --split=[fit, test], 
# --template=["Instruct-Query", "simple"]
# --use_lang_specific_prompts (include or not)