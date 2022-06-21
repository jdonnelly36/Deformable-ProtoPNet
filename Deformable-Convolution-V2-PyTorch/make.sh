#!/usr/bin/env bash
#SBATCH --job-name=proto_p_train  # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jonathan.donnelly@maine.edu     # Where to send mail
#SBATCH --output=testing_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=8
#SBATCH --mem=4gb                     # Job memory request
#SBATCH --time=1:00:00               # Time limit hrs:min:sec
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
source /home/jdonnelly/protoPNet/bin/activate
srun python setup.py build install 
