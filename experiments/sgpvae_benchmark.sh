#!/bin/bash
#SBATCH --job-name=gprpm
#SBATCH --output=gprpm_%A.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --time=2-12:00:00
#SBATCH --gres=gpu:1
#

srun --use-gpu -u python poisson_ball.py