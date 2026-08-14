#!/bin/bash
#SBATCH -A project_462001394
#SBATCH -p small-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 00:29:59
#SBATCH -N 1
#SBATCH -J evaluate
#SBATCH -o logs/%x-%j.out

echo "Running on gpu: $@"

# module setup
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7
export PYTHONPATH=/scratch/project_462001394/amanda/pythonuserbase/lib/python3.11/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462001394/hf_cache
export DATAPATH=/flash/project_462001394/datasets/
export HFKEY=$(cat ../hf_token.txt)

srun "$@"
