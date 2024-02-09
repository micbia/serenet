#!/bin/bash -l
#SBATCH --job-name=serenet
#SBATCH --account=c31
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00

#SBATCH --output=logs/serenet.%j.out
#SBATCH --error=logs/serenet.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.bianco@epfl.ch
#SBATCH --mem 60G

module purge
module load daint-gpu
module load gcc/9.3.0
module load cudatoolkit/10.2.89_3.28-2.1__g52c0314
module load TensorFlow/2.4.0-CrayGNU-21.09

source /project/c31/codes/miniconda3/etc/profile.d/conda.sh
conda activate segunet-env

CONFIG_PATH="$HOME/codes/serenet/config"

python utils_pred/pred_largeFoV.py
#python serenet.py $CONFIG_PATH/net_segunet2.ini
#python serenet.py $CONFIG_PATH/net_RecUnet_lc.ini
#python serenet.py $CONFIG_PATH/net_SegUnet_lc.ini
#python serenet.py $CONFIG_PATH/net_SERENEt_lc.ini
#python pred_segUNet.py
conda deactivate
