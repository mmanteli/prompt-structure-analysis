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
NON_LANG_DATASETS=("mteb/ARCChallenge" "summeval-2" "squad" "webfaq:eng")

is_lang_specific_data() {
    local d="$1"
    for skip in "${NON_LANG_DATASETS[@]}"; do
        [[ "$d" == "$skip" ]] && return 1
    done
    return 0
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
    #"codefuse-ai/F2LLM-v2-8B"
    #"Octen/Octen-Embedding-8B"
    #"jinaai/jina-embeddings-v5-text-small"
)

DATASETS=(
    "mteb/ARCChallenge"
    #"squad"
    #"webfaq:eng"
    #"mteb/tatoeba-bitext-mining:fin-eng"
    #"mteb/tatoeba-bitext-mining:fra-eng"
    #"mteb/tatoeba-bitext-mining:zho-eng"
    #"mteb/tatoeba-bitext-mining:ara-eng"
)

for split in test; do
    for template in "Instruct-Query"; do
        for model in "${MODELS[@]}"; do
            # sanitise model and dataset names for job names / log files
            safe_model="${model//\//_}"

            for data in "${DATASETS[@]}"; do
                safe_data="${data//\//_}"

                CMD=(python evaluate_prompts_distractors.py \
                    --k=$k \
                    --data_name="$data" \
                    --model_name="$model" \
                    --split="$split" \
                    --template="$template"\
                    --batch_size=8 \
                    --save_prefix="prompt_eval_results")

                wait_for_space
                #echo "eval_${safe_model}_${safe_data}_${split}_${template}"
                sbatch --job-name="eval_w_distractors/${safe_model}_${safe_data}_${split}_${template}" \
                       -t 02:59:59 \
                       slurm_run_command_gpu.sh "${CMD[@]}"

                if is_lang_specific_data "$data"; then
                    CMD=(python evaluate_prompts_distractors.py \
                        --k=$k \
                        --data_name="$data" \
                        --model_name="$model" \
                        --split="$split" \
                        --template="$template" \
                        --save_prefix="prompt_eval_results" \
                        --batch_size=8 \
                        --use_lang_specific_prompts)

                    wait_for_space
                    echo "eval_${safe_model}_${safe_data}_LANG_SPECIFIC_${split}_${template}"
                    sbatch --job-name="eval_w_distractors/${safe_model}_${safe_data}_${split}_${template}_lang_specific" \
                           -t 02:59:59 \
                           slurm_run_command_gpu.sh "${CMD[@]}"
                fi
            done
        done
    done
done