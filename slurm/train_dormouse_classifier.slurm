#!/bin/bash

#SBATCH --job-name=dormouse-train
#SBATCH --output=logs/dormouse-train_%j.out
#SBATCH --error=logs/dormouse-train_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# -----------------------------
# Activate environment
# -----------------------------
source ~/.bashrc
conda activate dormouse-classifier

# -----------------------------
# Move to project directory
# -----------------------------
cd /mnt/zt01/your/project/path

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