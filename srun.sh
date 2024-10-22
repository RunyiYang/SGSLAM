#!/bin/bash
#SBATCH --job-name=MonoGS-Mono-office1
#SBATCH --nodes=hala
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1          # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=16       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=16G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=1-0         # Job timeout
#SBATCH --output=myjob_mono8.log      # Redirect stdout to a log file
#SBATCH --error=myjob_mono8.error     # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email

source ~/.bashrc
eval "$(micromamba shell hook --shell )"
micromamba activate MonoGS

srun python slam.py --config configs/mono/replica/room1.yaml --save_dir room_1_mono


