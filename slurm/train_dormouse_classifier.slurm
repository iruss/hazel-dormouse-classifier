#!/bin/bash

#SBATCH --job-name=dormouse-train
#SBATCH --output=logs/dormouse-train_%j.out
#SBATCH --error=logs/dormouse-train_%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# -----------------------------
# Activate environment
# -----------------------------
source ${HOME}/miniforge3/etc/profile.d/conda.sh
conda activate dormouse-classifier

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# -----------------------------
# Move to project directory
# -----------------------------
cd /gpfs01/home/isorus/hazel-dormouse-classifier/

# Ensure log dir exists
mkdir -p logs

# -----------------------------
# Run training
# -----------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started: $(date)"

python train_dormouse_classifier.py

echo "Finished: $(date)"
