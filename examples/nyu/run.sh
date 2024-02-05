#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output 2self_supervised.out
##change name of the wandb folder


python main.py --weighting EW --arch HPS --dataset_path /home/g054545/LibMTL --gpu_id 0 --scheduler step --epochs 150 --save_path /home/g054545/LibMTL/examples/nyu/results --aug --split 0.75