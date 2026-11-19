#!/bin/bash

#SBATCH --job-name=dormouse-train
#SBATCH --output=logs/dormouse-train_%j.out
#SBATCH --error=logs/dormouse-train_%j.err
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

# ----------------------------------------
# Log Slurm parameters for optimisation
# ----------------------------------------
echo "Config: $CONFIG_FILE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Job ID: $SLURM_JOB_ID"

# ----------------------------------------
# Take in command line args
# ----------------------------------------
CONFIG_FILE=$1

# ----------------------------------------
# Activate environment
# ----------------------------------------
source ${HOME}/miniforge3/etc/profile.d/conda.sh
conda activate dormouse-classifier

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# ----------------------------------------
# Move to project directory
# ----------------------------------------
cd /gpfs01/home/isorus/hazel-dormouse-classifier/

# Ensure log dir exists
mkdir -p logs

# ----------------------------------------
# Run training
# ----------------------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Config: $CONFIG_FILE"

python train_dormouse_classifier.py \
    --config "$CONFIG_FILE"

echo "Finished: $(date)"
