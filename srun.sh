#!/bin/bash
#SBATCH --job-name=SGSLAM
#SBATCH --nodelist=hala
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1          # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=16       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=16G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=4:00:00         # Job timeout
#SBATCH --output=output_logs/test_mono.log      # Redirect stdout to a log file
#SBATCH --error=output_logs/test_mono.error     # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email

source ~/.bashrc

eval "$(micromamba shell hook --shell bash)"
# mircomamba create -f environment.yml
micromamba activate sgslam

srun python slam.py --config configs/mono/replica/office0.yaml --save_dir try_semantic_no_optimization_colors_precomp


