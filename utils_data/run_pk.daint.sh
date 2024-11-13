#!/bin/sh
#SBATCH --job-name=pk
#SBATCH --account=c31
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --mem 62G

##SBATCH --output=logs/pk-%A.%j.out
##SBATCH --error=logs/pk-%A.%j.err
##SBATCH --array=0-150
#SBATCH --output=../logs/pk.%j.out
#SBATCH --error=../logs/pk.%j.err

#SBATCH --time=24:00:00
#SBATCH --constraint=gpu

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mbianc@ethz.ch

module load daint-gpu
module load gcc/9.3.0
module load cudatoolkit/10.2.89_3.28-2.1__g52c0314
module load spack-config

# export conda on shell
source /project/c31/codes/miniconda3/etc/profile.d/conda.sh
conda activate karabo-env

python calculate_pk.py

conda deactivate
