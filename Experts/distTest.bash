#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --partition=npl-2024
#SBATCH --job-name=SpS_Reflexion_Test_distributed
#SBATCH --output=/gpfs/u/home/ARUS/ARUSgrsm/Reflexion-SpS/HotPotQA_Tests/OutputFiles/Outputs.txt
#SBATCH --error=/gpfs/u/home/ARUS/ARUSgrsm/Reflexion-SpS/HotPotQA_Tests/OutputFiles/Errors.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=garnes2@rpi.edu

# Load environment
source ~/barn/miniconda3x86/etc/profile.d/conda.sh
conda activate SpS+Reflexion

# Get master address for process group init
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# For debugging (printed only once on head node)
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# Launch with per-task RANK and WORLD_SIZE set inside each process
srun bash -c '
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  echo "[$(hostname)] Launching task with RANK=$RANK, WORLD_SIZE=$WORLD_SIZE"
  python distTest.py
'
