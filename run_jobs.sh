#!/bin/bash

#############################
# Slurm job configuration   #
#############################
# Adjust these settings for your HPC cluster.
#SBATCH --job-name=healthqa-bench
#SBATCH --output=healthqa-%j.out
#SBATCH --error=healthqa-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
## Uncomment and edit the following lines as needed:
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
##SBATCH --account=your_account

set -euo pipefail

echo "[INFO] Starting Health QA safety benchmark job on HPC"
echo "[INFO] Working directory: ${SLURM_SUBMIT_DIR:-$(pwd)}"

# Move to the directory from which the job was submitted
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Activate virtual environment if present
if [ -d ".venv" ]; then
  echo "[INFO] Activating .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[WARN] .venv not found; using system Python"
fi

export TOKENIZERS_PARALLELISM=false

echo "[INFO] Running MedQA (USMLE) benchmarks..."

echo "[INFO] TinyLlama on MedQA"
python3 src/run_bench.py \
  --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset medqa \
  -n 100 \
  --max_workers 2 \
  --judge_model Qwen/Qwen3-0.6B

echo "[INFO] BioGPT-Large on MedQA"
python3 src/run_bench.py \
  --models microsoft/BioGPT-Large \
  --dataset medqa \
  -n 100 \
  --max_workers 2 \
  --judge_model Qwen/Qwen3-0.6B

echo "[INFO] Running PubMedQA (train split) benchmarks..."

echo "[INFO] TinyLlama on PubMedQA train"
python3 src/run_bench.py \
  --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset pubmedqa \
  --split train \
  -n 100 \
  --max_workers 2 \
  --judge_model Qwen/Qwen3-0.6B

echo "[INFO] BioGPT-Large on PubMedQA train"
python3 src/run_bench.py \
  --models microsoft/BioGPT-Large \
  --dataset pubmedqa \
  --split train \
  -n 100 \
  --max_workers 2 \
  --judge_model Qwen/Qwen3-0.6B

echo "[INFO] Running evaluation over all outputs..."
python3 src/evaluate.py --outputs outputs --dest outputs/metrics.csv
python3 src/evaluate_judge.py --outputs outputs --dest outputs/metrics_judge.csv

echo "[INFO] Job completed. Metrics written to outputs/metrics_qwen.csv"