#!/bin/bash 
#SBATCH -p v100 -N 1  --cpus-per-gpu=4  --gpus-per-node=1 
python bayesian_optimizer.py