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


k=10

MODELS=(
    #"BAAI/bge-m3"
    "Qwen/Qwen3-Embedding-0.6B"
    "intfloat/multilingual-e5-large-instruct"
    #"nvidia/llama-embed-nemotron-8b"
    "microsoft/harrier-oss-v1-0.6b"
    #"nvidia/NV-Embed-v2"
    "google/embeddinggemma-300m"
)

DATASETS=(
    "mteb/reddit-clustering"
)

for split in test; do
    for template in "Instruct-Query"; do
        for model in "${MODELS[@]}"; do
            # sanitise model and dataset names for job names / log files
            safe_model="${model//\//_}"

            for data in "${DATASETS[@]}"; do
                safe_data="${data//:/_}"

                CMD=(python evaluate_prompts_clustering.py \
                    --k=$k \
                    --data_name="$data" \
                    --model_name="$model" \
                    --split="$split" \
                    --template="$template"\
                    --batch_size=8 \
                    --embedding_prefix="embeddings_eval" \
                    --save_prefix="prompt_eval_results")

                wait_for_space
                #echo "eval_${safe_model}_${safe_data}_${split}_${template}"
                sbatch --job-name="eval_${safe_model}_${safe_data}_${split}_${template}" \
                       -t 02:59:59 \
                       slurm_run_command_gpu.sh "${CMD[@]}"

            done
        done
    done
done