#!/bin/bash
#SBATCH --job-name=healthqa-bench
#SBATCH --output=healthqa-%j.out
#SBATCH --error=healthqa-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1

set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

cd "$SLURM_SUBMIT_DIR"

# (Optional) Load modules / activate environment
# module load anaconda
source venv/bin/activate

# Step 1: Prepare instruction data
python src/prepare_instruct_data.py

# Step 2: Train LoRA model
python src/train_lora.py
