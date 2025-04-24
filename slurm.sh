#!/bin/bash
#SBATCH --job-name=DDDM-VC
#SBATCH --output=arihant_log2.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ethanwilson@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
##SBATCH --partition=gpu
##SBATCH --gpus=1
#SBATCH --time=72:00:00

## Usage: call sbatch slurm.sh <command to run>
## ex. sbatch slurm.sh python inference.py --arg1 <arg1>

module load conda
module load ffmpeg

conda activate /blue/ejain/conda/DDDM-VC.conda

date;hostname;pwd

echo "$@"

$@

