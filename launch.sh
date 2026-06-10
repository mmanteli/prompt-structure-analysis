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

for split in fit test; do
    for template in "Instruct-Query" "simple"; do
        CMD=(python prompting_metrics.py \
            --split="$split" \
            --template="$template")

        wait_for_space
        sbatch --job-name="prompt_metrics_${split}_${template}" \
               -t 02:59:59 \
               slurm_run_command_gpu.sh "${CMD[@]}"

        CMD=(python prompting_metrics.py \
            --split="$split" \
            --template="$template" \
            --use_lang_specific_prompts)

        wait_for_space
        sbatch --job-name="prompt_metrics_${split}_${template}_lang_specific" \
               -t 02:59:59 \
               slurm_run_command_gpu.sh "${CMD[@]}"
    done
done

exit 0
