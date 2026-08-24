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

# datasets that are NOT language-specific (skip --use_lang_specific_prompts)
NON_LANG_DATASETS=("mteb/ARCChallenge" "summeval-2")

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

# ── First: structural analysis ──────────────────────────────────────

for dataset in "mteb/ARCChallenge" "webfaq:eng" "webfaq:deu" "mteb/tatoeba-bitext-mining:fin-eng" "mteb/tatoeba-bitext-mining:fra-eng"; do   
    for model in "${MODELS[@]}"; do
        for template in "Instruct-Query"; do
            CMD=(python prompting_metrics.py \
                --model=$model \
                --data_name=$dataset \
                --split="test" \
                --template="$template" \
                --save_prefix="prompt_structure_results" \
                --embedding_prefix="embeddings_struct" \
                --batch_size=4)

            wait_for_space
            echo "${model}:${dataset}:${split}_${template}"
            sbatch --job-name="prompt_metrics_${model}_${dataset}:${split}_${template}" -t 1:39:59 slurm_run_command_gpu.sh "${CMD[@]}"
            
            if is_lang_specific_data $dataset; then
                CMD=(python prompting_metrics.py \
                    --model=$model \
                    --split="test" \
                    --data_name=$dataset \
                    --template="$template" \
                    --save_prefix="prompt_structure_results" \
                    --batch_size=4 \
                    --embedding_prefix="embeddings_struct" \
                    --use_lang_specific_prompts)

                wait_for_space
                echo "${model}:${dataset}:${split}_${template}_lang_specific"
                sbatch --job-name="prompt_metrics_${model}_${dataset}:${split}_${template}_lang_specific" -t 1:39:59 slurm_run_command_gpu.sh "${CMD[@]}"
            fi
        done
    done
done

exit 0
