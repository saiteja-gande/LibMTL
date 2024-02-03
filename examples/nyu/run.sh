#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --output name.out

##change name of the wandb folder

python main.py --weighting EW --arch HPS --dataset_path /home/g054545/LibMTL --gpu_id 0 --scheduler step --epochs 150 --save_path /home/g054545/LibMTL/examples/nyu/wandb --aug