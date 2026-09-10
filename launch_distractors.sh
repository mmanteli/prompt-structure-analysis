#!/bin/bash

squeue_has_space() {
    local max_jobs="${1:-190}"
    local n
    n=$(squeue --me -h 2>/dev/null | wc -l) || return 3
    (( n < max_jobs ))
}

wait_for_space() {
    while ! squeue_has_space; do
        echo "Queue full, waiting 30s..."
        sleep 30
    done
}


DATASETS=(
    "mteb/ARCChallenge"
    "squad"
    "mteb/tatoeba-bitext-mining:ara-eng"
    "mteb/tatoeba-bitext-mining:cmn-eng"
    "mteb/tatoeba-bitext-mining:deu-eng"
    "mteb/tatoeba-bitext-mining:fin-eng"
    "mteb/tatoeba-bitext-mining:fra-eng"
    "mteb/tatoeba-bitext-mining:spa-eng"
    "mteb/tatoeba-bitext-mining:vie-eng"
    "mteb/tatoeba-bitext-mining:tur-eng"
    "mteb/tatoeba-bitext-mining:zho-eng"
)


MODELS=(
    "BAAI/bge-m3"
    "Qwen/Qwen3-Embedding-0.6B"
    "Qwen/Qwen3-Embedding-4B"
    #"/flash/project_462001491/models/v1-20260828-095152/checkpoint-18000"
    "intfloat/multilingual-e5-large-instruct"
    "nvidia/llama-embed-nemotron-8b"
    "microsoft/harrier-oss-v1-0.6b"
    "nvidia/NV-Embed-v2"
    "google/embeddinggemma-300m"
    "codefuse-ai/F2LLM-v2-8B"
    "Octen/Octen-Embedding-8B"
    #"jinaai/jina-embeddings-v5-text-small"
)



# ── First: structural analysis ──────────────────────────────────────

split="test"
k="[1,2,5,10]"

for dataset in "${DATASETS[@]}"; do   
    for model in "${MODELS[@]}"; do
        for template in "Instruct-Query"; do
            CMD=(python prompt_structure_with_distractors.py \
                    --model=$model \
                    --split=$split \
                    --nn=10 \
                    --data_name=$dataset \
                    --template="$template" \
                    --save_prefix="results" \
                    --batch_size=4)

            wait_for_space
            echo "STRUCTURE ${model}:${dataset}:${split}_${template}_@${k}"
            #sbatch --job-name="distractors/${model_safe_name}/${data_safe_name}:${split}_${template}_10nn" -t 0:59:59 slurm_run_command_gpu.sh "${CMD[@]}"
        done
    done
done

for dataset in "${DATASETS[@]}"; do   
    for model in "${MODELS[@]}"; do
        for template in "Instruct-Query"; do
            CMD=(python prompt_eval_with_distractors.py \
                    --model=$model \
                    --split=$split \
                    --k=$k \
                    --data_name=$dataset \
                    --template="$template" \
                    --save_prefix="results" \
                    --batch_size=4)

            wait_for_space
            echo "EVAL ${model}:${dataset}:${split}_k12510"
            #sbatch --job-name="distractors_eval/${model_safe_name}/${data_safe_name}:${split}_${template}_k12510" -t 2:59:59 slurm_run_command_gpu.sh "${CMD[@]}"
        done
    done
done

exit 0
