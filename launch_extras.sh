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
)

MODELS=(
    "BAAI/bge-m3"  # 1024
    "Qwen/Qwen3-Embedding-0.6B"   # 1024
    "Qwen/Qwen3-Embedding-4B"   # 2560
    "/scratch/project_462001491/jmnybl/final_embedding_model_checkpoints/v2-20260909-final/final-finetuned-model"
    "/scratch/project_462001491/jmnybl/final_embedding_model_checkpoints/v2-20260909-final/checkpoint-18754"
    "intfloat/multilingual-e5-large-instruct"   #1024
    "microsoft/harrier-oss-v1-0.6b"   # 1024
    "google/embeddinggemma-300m"   # 768
    "codefuse-ai/F2LLM-v2-4B"     # 2560
    "Octen/Octen-Embedding-8B"   # 4096
    "/scratch/project_462001491/jmnybl/checkpoint-19478-tatoeba"
)




for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        model_safe_name="${model//\//_}"
        data_safe_name="${dataset//\//_}"
        CMD=(python calculate_distractor_to_query_distances.py $model $dataset)
        wait_for_space
        echo "${CMD[@]}"
        sbatch --job-name="distractors_extra/${model_safe_name}" -t 2:59:59 slurm_run_command_gpu.sh "${CMD[@]}"
    done
done
