#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output NAPL_NPL-SSL10-f-cross-L75.out
##change name of the wandb folder
## Newpseudolabel, c14-class14, NOIGI-Noignoring index or KLD - KLD loss, SSL1--warmupfor10epochs, L25-labeled25 , apl -averagepseudlabel / NO- no

python main.py --weighting EW --arch HPS --dataset_path /home/g054545/LibMTL --gpu_id 0 --scheduler step --epochs 150