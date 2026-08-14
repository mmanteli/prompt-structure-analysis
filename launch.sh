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

# ── First: structural analysis ──────────────────────────────────────

for dataset in "arcchallenge" "summeval-2" "webfaq:deu" "webfaq:eng" "webfaq:zho" "tatoeba:fin-eng" "tatoeba:fra-eng"; do   
    for split in fit; do
        for template in "Instruct-Query" "simple"; do
            CMD=(python prompting_metrics.py \
                --data_name=$dataset \
                --split="$split" \
                --template="$template" \
                --save_prefix="prompt_metrics" \
                --batch_size=4)

            wait_for_space
            echo "${dataset}:${split}_${template}"
            sbatch --job-name="prompt_metrics_${dataset}:${split}_${template}" -t 0:39:59 slurm_run_command_gpu.sh "${CMD[@]}"
            
            if is_lang_specific_data $dataset; then
                CMD=(python prompting_metrics.py \
                    --split="$split" \
                    --data_name=$dataset \
                    --template="$template" \
                    --save_prefix="prompt_metrics" \
                    --batch_size=4 \
                    --use_lang_specific_prompts)

                wait_for_space
                echo "${dataset}:${split}_${template}_lang_specific"
                sbatch --job-name="prompt_metrics_${dataset}:${split}_${template}_lang_specific" -t 0:39:59 slurm_run_command_gpu.sh "${CMD[@]}"
            fi
        done
    done
done

exit 0
