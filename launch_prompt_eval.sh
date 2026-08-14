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
NON_LANG_DATASETS=("arcchallenge" "summeval-2")

is_lang_specific_data() {
    local d="$1"
    for skip in "${NON_LANG_DATASETS[@]}"; do
        [[ "$d" == "$skip" ]] && return 1
    done
    return 0
}

k=10

MODELS=(
    "BAAI/bge-m3"
    "Qwen/Qwen3-Embedding-0.6B"
    "intfloat/multilingual-e5-small"
    "intfloat/multilingual-e5-large-instruct"
    "minishlab/potion-base-8M"
    "google/embeddinggemma-300m"
)

DATASETS=(
    "arcchallenge"
    "summeval-2"
    "tatoeba:fin-eng"
    "tatoeba:fra-eng"
    "tatoeba:bul-eng"
    "webfaq:deu"
    "webfaq:eng"
    "webfaq:zho"
)

for split in fit; do
    for template in "Instruct-Query" "simple"; do
        for model in "${MODELS[@]}"; do
            # sanitise model and dataset names for job names / log files
            safe_model="${model//\//_}"

            for data in "${DATASETS[@]}"; do
                safe_data="${data//:/_}"

                CMD=(python evaluate_prompts.py \
                    --k=$k \
                    --data_name="$data" \
                    --model_name="$model" \
                    --split="$split" \
                    --template="$template"\
                    --batch_size=8 \
                    --save_prefix="prompt_eval")

                wait_for_space
                sbatch --job-name="eval_${safe_model}_${safe_data}_${split}_${template}" \
                       -t 00:59:59 \
                       slurm_run_command_gpu.sh "${CMD[@]}"

                if is_lang_specific_data "$data"; then
                    CMD=(python evaluate_prompts.py \
                        --k=$k \
                        --data_name="$data" \
                        --model_name="$model" \
                        --split="$split" \
                        --template="$template" \
                        --save_prefix="prompt_eval" \
                        --batch_size=8 \
                        --use_lang_specific_prompts)

                    wait_for_space
                    sbatch --job-name="eval_${safe_model}_${safe_data}_${split}_${template}_lang_specific" \
                           -t 00:49:59 \
                           slurm_run_command_gpu.sh "${CMD[@]}"
                fi
            done
        done
    done
done