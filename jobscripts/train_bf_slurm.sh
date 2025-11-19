#!/bin/bash
#SBATCH --time=7-00:00:00             # Max runtime: 7 days
#SBATCH --job-name=Rubisco_Train      # Job name
#SBATCH --partition=compchemq          # Partition to use
#SBATCH --mem=100G                    # Memory allocation
#SBATCH --qos=compchem                # QoS for the partition
#SBATCH -N1 --ntasks-per-node=1      # Single node, 1 task
#SBATCH --cpus-per-task=30           # CPU cores allocated
#SBATCH --ntasks-per-socket=1        # CPUs per socket
#SBATCH --gres=gpu:4                 # Number of GPUs requested
#SBATCH --get-user-env               # Load user environment

# Load necessary modules (adjust as per your cluster environment)
module load cuda-12.2.2
module load anaconda-uoneasy/2023.09-0

# Initialize conda and activate your environment
conda init zsh
source ~/.zshrc
conda activate your-conda-env-name       # <-- Replace with your conda env

# Run your training Python script with distributed launcher
torchrun --nproc_per_node=4 src/train_bf.py > train_bf.out

# Optional: commands for saving results or cleanup after job finishes
