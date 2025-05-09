#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --partition=npl-2024
#SBATCH --job-name=Router_Test
#SBATCH --output=/gpfs/u/home/ARUS/ARUSgrsm/HERD/Outputs/Outputs.txt
#SBATCH --error=/gpfs/u/home/ARUS/ARUSgrsm/HERD/Outputs/Errors.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=garnes2@rpi.edu

source ~/barn/miniconda3x86/etc/profile.d/conda.sh
conda activate SpS+Reflexion

srun python /gpfs/u/home/ARUS/ARUSgrsm/HERD/HERD_Prototype_RCOS/router/Router.py