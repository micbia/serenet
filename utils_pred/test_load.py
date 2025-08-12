import numpy as np, os, tensorflow as tf
from glob import glob

# add SERENEt directory to path
import sys
sys.path.append("/users/mibianco/codes/serenet")
from config.net_config import NetworkConfig
from utils_pred.prediction import Unet2Predict, LoadModel

#path_model = '/store/ska/sk014/serenet/outputs_segunet/segunet24-09T23-36-45_128slice_paper2/'
path_model = '/store/ska/sk014/serenet/outputs_recunet/recunet17-05T09-23-54_128slice/'
#path_model = '/store/ska/sk014/serenet/outputs_recunet/serenet24-04T16-32-44_128slice_priorGT/'
#path_model = '/store/ska/sk014/serenet/outputs_recunet/serenet01-05T17-59-07_128slice_priorTS/'

config_file = glob(path_model+'*.ini')[0]
print(config_file)
conf = NetworkConfig(config_file)
model = LoadModel(config_file)
