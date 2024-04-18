#!/bin/bash

# Slurm SBATCH directives must come before any executable command in the script.
# This script requests a node in the GPU partition with access to 1 GPU for 3 days,
# allocates 1 nodes with 32GB of memory each, specifies a job name, and sets output files.

# Request a job time of 2 days
#SBATCH --time=2-12:00:00

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu
#SBATCH --gres=gpu:1

# Request 1 nodes (ensure they're on the same node if needed with -N and -n appropriately)
#SBATCH -N 1
#4 tasks
#SBATCH -n 4

# Request 32GB of memory
#SBATCH --mem=32G

# Specify a job name
#SBATCH -J HelixBig

# Specify output and error files
# %j is replaced by the JobID when the job starts
#SBATCH -o HelixBig-%j.out
#SBATCH -e HelixBig-%j.err

# --- Your executable commands below this line ---


module load cuda

module load anaconda/2023.03-1
source /gpfs/runtime/opt/anaconda/2023.03-1/etc/profile.d/conda.sh

conda activate bhaimnet2
python run_aimnet2.py

