#!/bin/bash
#SBATCH --job-name=DDDM-VC_slurm
#SBATCH --output=latest_log.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ethanwilson@ufl.edu
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=72:00:00

## Usage: call sbatch slurm.sh <command to run>
## ex. sbatch slurm.sh python inference.py --arg1 <arg1>

module load conda

conda activate /blue/ejain/conda/DDDM-VC.conda

date;hostname;pwd

echo "$@"

$@

