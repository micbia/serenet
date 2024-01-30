import sys
sys.path.append("../")
import numpy as np, tools21cm as t2c

from utils_network.data_generator import LightConeGenerator, LightConeGenerator_SERENEt

PATH_TRAIN = '/home/michele/Documents/PhD_Sussex/output/ML/dataset/inputs/dataLC_128_pred_190922/'
train_idx = np.arange(10)
IM_SHAPE = (128, 128)
BATCH_SIZE = 5

dg = LightConeGenerator_SERENEt(path=PATH_TRAIN, data_temp=train_idx, data_shape=IM_SHAPE, batch_size=BATCH_SIZE, shuffle=True)
data = dg.__getitem__(0)
x_input, x_prior, y = data

print(np.shape(data), x_input.shape, x_prior.shape, y.shape)