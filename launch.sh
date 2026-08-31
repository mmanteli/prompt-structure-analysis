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
    "mteb/tatoeba-bitext-mining:fin-eng"
    "mteb/tatoeba-bitext-mining:fra-eng"
    "mteb/tatoeba-bitext-mining:zho-eng"
    "mteb/tatoeba-bitext-mining:ara-eng"
)

# datasets that are NOT language-specific (skip --use_lang_specific_prompts)
NON_LANG_DATASETS=("mteb/ARCChallenge")

MODELS=(
    "BAAI/bge-m3"
    "Qwen/Qwen3-Embedding-0.6"
    "intfloat/multilingual-e5-large-instruct"
    "nvidia/llama-embed-nemotron-8b"
    "microsoft/harrier-oss-v1-0.6b"
    "nvidia/NV-Embed-v2"
    "google/embeddinggemma-300m"
    "codefuse-ai/F2LLM-v2-8B"
    "Octen/Octen-Embedding-8B"
    "jinaai/jina-embeddings-v5-text-small"
)



is_lang_specific_data() {
    local d="$1"
    for skip in "${NON_LANG_DATASETS[@]}"; do
        [[ "$d" == "$skip" ]] && return 1
    done
    return 0
}

# ── First: structural analysis ──────────────────────────────────────

split="test"

for dataset in "${DATASETS[@]}"; do   
    for model in "${MODELS[@]}"; do
        for template in "Instruct-Query"; do
            CMD=(python prompting_metrics.py \
                --model=$model \
                --data_name=$dataset \
                --split=$split \
                --template="$template" \
                --save_prefix="prompt_structure_results" \
                --embedding_prefix="embeddings_struct" \
                --batch_size=4)

            model_safe_name="${model//\//_}"
            data_safe_name="${dataset//\//_}"
            wait_for_space
            echo "${model}:${dataset}:${split}_${template}"
            sbatch --job-name="prompt_metrics/${model_safe_name}_${data_safe_name}:${split}_${template}" -t 0:29:59 slurm_run_command_gpu.sh "${CMD[@]}"
            
            if is_lang_specific_data $dataset; then
                CMD=(python prompting_metrics.py \
                    --model=$model \
                    --split=$split \
                    --data_name=$dataset \
                    --template="$template" \
                    --save_prefix="prompt_structure_results" \
                    --batch_size=4 \
                    --embedding_prefix="embeddings_struct" \
                    --use_lang_specific_prompts)

                wait_for_space
                echo "${model}:${dataset}:${lang}:${split}_${template}_lang_specific"
                sbatch --job-name="prompt_metrics/${model_safe_name}_${data_safe_name}:${split}_${template}_lang_specific" -t 0:29:59 slurm_run_command_gpu.sh "${CMD[@]}"
            fi
        done
    done
done

exit 0
