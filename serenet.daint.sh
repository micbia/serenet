#!/bin/bash -l
#SBATCH --job-name=serenet
#SBATCH --account=c31
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=logs/serenet.%j.out
#SBATCH --error=logs/serenet.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mbianc@ethz.ch

module purge
module load cray/23.12
module load cray-python/3.11.5
module load cudatoolkit/24.3_12.3

source $HOME/myvenv/serenet-env/bin/activate
export HDF5_USE_FILE_LOCKING='FALSE'

CONFIG_PATH="$HOME/codes/serenet/config"

#python utils_pred/pred_largeFoV.py
#python serenet.py $CONFIG_PATH/net_segunet2.ini
#python serenet.py $CONFIG_PATH/net_RecUnet_lc.ini
#python serenet.py $CONFIG_PATH/net_SegUnet_lc.ini
python serenet.py $CONFIG_PATH/net_SERENEt_lc.ini
#python pred_segUNet.py

