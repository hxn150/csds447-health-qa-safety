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
# TinyLlama on MedQA
echo "[$(date)] Step 2a: Training LoRA (TinyLlama-1.1B-Chat-v1.0 on medqa)..."
python src/train_lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset medqa
echo "[$(date)] Step 2a completed."
echo

# TinyLlama on PubMedQA
# echo "[$(date)] Step 2b: Training LoRA (TinyLlama-1.1B-Chat-v1.0 on pubmedqa)..."
# python src/train_lora.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset pubmedqa
# echo "[$(date)] Step 2b completed."
# echo

# BioGPT-Large on MedQA
echo "[$(date)] Step 2c: Training LoRA (microsoft/BioGPT-Large on medqa)..."
python src/train_lora.py --model microsoft/BioGPT-Large --dataset medqa
echo "[$(date)] Step 2c completed."
echo

# BioGPT-Large on PubMedQA
# echo "[$(date)] Step 2d: Training LoRA (microsoft/BioGPT-Large on pubmedqa)..."
# python src/train_lora.py --model microsoft/BioGPT-Large --dataset pubmedqa
# echo "[$(date)] Step 2d completed."
# echo

echo "[$(date)] healthqa-bench job finished successfully."
