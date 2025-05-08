#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --partition=npl-2024
#SBATCH --job-name=Expert_Cluster_Test
#SBATCH --output=/gpfs/u/home/ARUS/ARUSgrsm/Reflexion-SpS/HotPotQA_Tests/OutputFiles/Outputs.txt
#SBATCH --error=/gpfs/u/home/ARUS/ARUSgrsm/Reflexion-SpS/HotPotQA_Tests/OutputFiles/Errors.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=garnes2@rpi.edu

source ~/barn/miniconda3x86/etc/profile.d/conda.sh
conda activate SpS+Reflexion

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=ERROR
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1

srun bash -c '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

  echo "[$(hostname)] Launching ExpertCluster.py with RANK=$RANK, WORLD_SIZE=$WORLD_SIZE"
  python ExpertTest.py 2>&1 | tee /gpfs/u/home/ARUS/ARUSgrsm/Reflexion-SpS/HotPotQA_Tests/OutputFiles/log_rank$RANK.txt
'
