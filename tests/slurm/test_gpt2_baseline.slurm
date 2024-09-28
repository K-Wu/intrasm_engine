#!/bin/bash
#SBATCH --time=0:10:59
#SBATCH -p a100
#SBATCH --account=bbzc-hydro
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=5
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --job-name=do_dist_training
#SBATCH --output=./out/my_output.%j.out
#SBATCH --error=./out/my_output.%j.err

source ~/.bashrc

# Load any necessary modules (e.g., Anaconda)
export PYTHONPATH=$PYTHONPATH:$MY_TOOLKIT_PATH/../

# Activate your Python environment (if needed)
eval "$(conda shell.bash hook)"
conda activate interleaved_env
which python

nvcc -V

nsys profile --cuda-graph-trace node -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o ./nsys_rep/test_gpt2_baseline_$SLURM_JOB_ID -f true -x true python ../huggingface/test_gpt2_baseline.py
# python ../test_interleaved_linear.py
