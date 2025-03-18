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

source ~/barn/miniconda3x86/etc/profile.d/conda.sh
conda activate SpS+Reflexion

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun python distTest.py