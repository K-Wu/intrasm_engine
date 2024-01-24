#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --time=167:59:00
#SBATCH -p a100
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --job-name=partition_graph
#SBATCH --output=my_output/%j.out
#SBATCH --error=my_output/%j.err

# Load any necessary modules (e.g., Anaconda)


# Activate your Python environment (if needed)
eval "$(conda shell.bash hook)"
conda activate gids_osdi24
which python
python --version

# Run your Python script
# MY_TOOLKIT_PATH is set in sbatch_and_tail.sh and is /shell
cd $MY_TOOLKIT_PATH/..
#. ../../cutlass/build/tools/profiler/cutlass_profiler --operation=Gemm
/u/kunwu2/projects/cutlass/build/tools/profiler/cutlass_profiler --operation=Gemm