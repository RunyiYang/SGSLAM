#!/bin/bash
#SBATCH --job-name=monogs
#SBATCH --output=sbatch_log/prune_baseline_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

JOB_START_TIME=$(date)
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}" 
echo "Running on node: $(hostname)"
echo "Starting on:     ${JOB_START_TIME}" 

source /scratch_net/schusch/qimaqi/miniconda3//etc/profile.d/conda.sh 
conda activate MonoGS_P
evo_config set plot_backend agg
# python slam.py --config configs/rgbd/replica/office0.yaml --eval

# python slam.py --config configs/rgbd/replica/office1.yaml --eval
# python slam.py --config configs/rgbd/replica/office2.yaml --eval
# python slam.py --config configs/rgbd/replica/office3.yaml --eval
# python slam.py --config configs/rgbd/replica/office4.yaml --eval
python slam.py --config configs/rgbd/replica/room0.yaml --eval
python slam.py --config configs/rgbd/replica/room0.yaml --eval
python slam.py --config configs/rgbd/replica/room0.yaml --eval

python slam.py --config configs/rgbd/replica/room1.yaml --eval
python slam.py --config configs/rgbd/replica/room1.yaml --eval
python slam.py --config configs/rgbd/replica/room1.yaml --eval

python slam.py --config configs/rgbd/replica/room2.yaml --eval
python slam.py --config configs/rgbd/replica/room2.yaml --eval
python slam.py --config configs/rgbd/replica/room2.yaml --eval

# python slam.py --config configs/mono/replica/office0.yaml --eval
# python slam.py --config configs/mono/replica/office1.yaml --eval
# python slam.py --config configs/mono/replica/office2.yaml --eval
# python slam.py --config configs/mono/replica/office3.yaml --eval
# python slam.py --config configs/mono/replica/office4.yaml --eval
# python slam.py --config configs/mono/replica/room0.yaml --eval
# python slam.py --config configs/mono/replica/room1.yaml --eval
# python slam.py --config configs/mono/replica/room2.yaml --eval
