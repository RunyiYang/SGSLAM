#!/bin/bash
#SBATCH --job-name=Mono
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

eval "$(micromamba shell hook --shell )"
mircomamba create -f environment_new.yml
micromamba activate MonoGS

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
micromamba install nvidia/label/cuda-11.6.1::cuda-toolkit
pip install -r requirements.txt

srun python slam.py --config configs/mono/replica/room2.yaml --save_dir room_depth=F


