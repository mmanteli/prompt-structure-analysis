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
    #"mteb/ARCChallenge"
    #"webfaq:eng"
    #"rajpurkar/squad"
    "tatoeba:fra-eng"
)
split="test"


# from different stages of fine-tuning
MODELS=(
    "Qwen/Qwen3-Embedding-0.6B"
    #"/flash/project_462001491/models/v1-20260828-095152/checkpoint-2000"
    #"/flash/project_462001491/models/v1-20260828-095152/checkpoint-4000"
    #"/flash/project_462001491/models/v1-20260828-095152/checkpoint-6000" 
    #"/flash/project_462001491/models/v1-20260828-095152/checkpoint-12000"
    "/flash/project_462001491/models/v1-20260828-095152/checkpoint-18000"
)

for dataset in "${DATASETS[@]}"; do   
    for model in "${MODELS[@]}"; do
        for template in "Instruct-Query"; do
            CMD=(python prompting_metrics.py \
                --model=$model \
                --data_name=$dataset \
                --split=$split \
                --template="$template" \
                --save_prefix="finetuned_results" \
                --batch_size=4)

            model_safe_name="${model//\//_}"
            data_safe_name="${dataset//\//_}"
            wait_for_space
            echo "Running ${model}:${dataset}:${split}_${template}"
            echo "${CMD[@]}"
            sbatch --job-name="finetuned_prompt_metrics/${model_safe_name}_${data_safe_name}:${split}_${template}" -t 0:29:59 slurm_run_command_gpu.sh "${CMD[@]}"
            
            CMD=(python evaluate_prompts.py \
                --model=$model \
                --data_name=$dataset \
                --split=$split \
                --template="$template" \
                --save_prefix="finetuned_results" \
                --batch_size=4)
            echo "${CMD[@]}"
            sbatch --job-name="finetuned_prompt_eval/${model_safe_name}_${data_safe_name}:${split}_${template}" -t 2:29:59 slurm_run_command_gpu.sh "${CMD[@]}"
        done
    done
done

exit 0
