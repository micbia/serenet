#!/bin/bash -l
#SBATCH --job-name=preproc
#SBATCH --account=sk014
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00
#SBATCH --output=../logs/preproc.%j.out
#SBATCH --error=../logs/preproc.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.bianco@epfl.ch
#SBATCH --mem 60G

module load daint-gpu
module load gcc/9.3.0

source /project/c31/codes/miniconda3/etc/profile.d/conda.sh
conda activate karabo-env
python preprocess.py
conda deactivate
