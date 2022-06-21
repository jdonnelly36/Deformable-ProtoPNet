#!/usr/bin/env bash
#SBATCH --job-name=global_analysis  # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jonathan.donnelly@maine.edu     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=12
#SBATCH --output=global_analysis_%j.out
#SBATCH --mem=30gb                     # Job memory request
#SBATCH --time=24:23:45               # Time limit hrs:min:sec
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
source /home/jdonnelly/protoPNet/bin/activate

#for m in m0.12 m0.14 m0.16 m0.18 m0.20 m0.22 m0.24 m0.26 m0.28 m0.30 m0.32 m0.34 m0.36 m0.38 m0.40
#do
MODELDIR='saved_models/resnet50/datasets/CUB_200_2011/train/001/'
MODELNAME='30push0.8564.pth'

srun python3 global_analysis.py -gpuid='0' -modeldir $MODELDIR -model $MODELNAME
#done