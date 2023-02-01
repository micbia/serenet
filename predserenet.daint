#!/bin/bash -l
#SBATCH --job-name=segunet
#SBATCH --account=sk09
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00
#SBATCH --output=logs/segunet.%j.out
#SBATCH --error=logs/segunet.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.bianco@epfl.ch
#SBATCH --mem 60G

module load daint-gpu
module load gcc/9.3.0
module load cudatoolkit/10.2.89_3.28-2.1__g52c0314
module load TensorFlow/2.4.0-CrayGNU-21.09

CONFIG_PATH="$SCRATCH/output_segunet/outputs/all24-09T23-36-45_128slice"

source $HOME/venvs/segunet-venv/bin/activate
python utils_plot/postpros_plot.py "$CONFIG_PATH/outputs"
python pred_segUNet.py $CONFIG_PATH/net2D_lc_full.ini
deactivate