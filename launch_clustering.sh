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


MODELS=(
    #"BAAI/bge-m3"
    "Qwen/Qwen3-Embedding-0.6B"
    "intfloat/multilingual-e5-large-instruct"
    #"nvidia/llama-embed-nemotron-8b"
    "microsoft/harrier-oss-v1-0.6b"
    #"nvidia/NV-Embed-v2"
    "google/embeddinggemma-300m"
)


is_lang_specific_data() {
    local d="$1"
    for skip in "${NON_LANG_DATASETS[@]}"; do
        [[ "$d" == "$skip" ]] && return 1
    done
    return 0
}

for dataset in "mteb/reddit-clustering"; do   
    for model in "${MODELS[@]}"; do
        for template in "Instruct-Query"; do
            CMD=(python prompting_metrics_clustering.py \
                --model=$model \
                --data_name=$dataset \
                --split="test" \
                --subsplit=0 \
                --template="$template" \
                --save_prefix="prompt_structure_results" \
                --embedding_prefix="embeddings_struct" \
                --batch_size=4)

            wait_for_space
            echo "${model}:${dataset}:${split}_${template}"
            sbatch --job-name="prompt_metrics_${model}_${dataset}:${split}_${template}" -t 4:39:59 slurm_run_command_gpu.sh "${CMD[@]}"
        done
    done
done


for dataset in "mteb/multi-hatecheck"; do   
    for model in "${MODELS[@]}"; do
        for template in "Instruct-Query"; do
            for lang in "eng" "fra"; do
                CMD=(python prompting_metrics_clustering.py \
                    --model=$model \
                    --data_name="${dataset}:${lang}" \
                    --split="test" \
                    --template="$template" \
                    --save_prefix="prompt_structure_results" \
                    --embedding_prefix="embeddings_struct" \
                    --batch_size=4)

                wait_for_space
                echo "${model}:${dataset}:${split}_${template}"
                sbatch --job-name="prompt_metrics_${model}_${dataset}:${split}_${template}" -t 4:39:59 slurm_run_command_gpu.sh "${CMD[@]}"
            done
        done
    done
done

exit 0
