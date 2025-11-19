#!/bin/bash
#SBATCH --time=7-00:00:00          # Max runtime: 7 days
#SBATCH --job-name=Rubisco_Inference  # Job name
#SBATCH --partition=compchemq        # Partition to run on
#SBATCH --mem=100G                   # Memory allocation
#SBATCH --qos=compchem               # QoS for the partition
#SBATCH -N1 --ntasks-per-node=1     # Single node, 1 task
#SBATCH --cpus-per-task=30          # Number of CPU cores
#SBATCH --ntasks-per-socket=1       # CPUs per socket
#SBATCH --gres=gpu:2                # Number of GPUs requested
#SBATCH --get-user-env              # Use user's environment

# Load necessary modules (modify based on your cluster's modules)
module load cuda-12.2.2
module load anaconda-uoneasy/2023.09-0

# Initialize conda for the shell & activate environment
conda init zsh
source ~/.zshrc
conda activate your-conda-env-name   # <-- Replace with your conda env name

# Run the inference script with distributed launcher (update script path if needed)
torchrun --nproc_per_node=2 src/thermo_final.py > thermo_final.out

# Optional: Add commands to save results or clean up after job finishes
