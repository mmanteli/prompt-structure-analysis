#!/bin/bash
#SBATCH -A project_462001394
#SBATCH -p small-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH -t 00:59:59
#SBATCH -N 1
#SBATCH -J test
#SBATCH -o logs/%x-%j.out

echo "Running on gpu: $@"
echo $(date +%d/%m/%Y_%H:%M:%S)

# module setup
module purge
module load Local-LAIF lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif

#export PYTHONPATH=/scratch/project_462001394/amanda/pythonuserbase/lib/python3.11/site-packages:$PYTHONPATH
export HF_HOME=/scratch/project_462001394/hf_cache
export DATAPATH=/flash/project_462001394/datasets/
export HFKEY=$(cat /scratch/project_462001491/amanda/prompting/hf_token.txt)

#singularity run $SIF "$@"
singularity run $SIF bash -c 'source ../.venv/bin/activate && "$@"' _ "$@"
echo $(date +%d/%m/%Y_%H:%M:%S)