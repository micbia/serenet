import tensorflow as tf, h5py
from glob import glob
import sys
sys.path.append("../")
from utils_network.networks import Unet, SERENEt, FullSERENEt, Auto1D
from config.net_config import NetworkConfig
from utils_pred.prediction import Unet2Predict, LoadModel

path_model = "/store/ska/sk014/serenet/outputs_recunet/serenet01-05T17-59-07_128slice_priorTS/"

config_file = glob(path_model+'*.ini')[0]
print(config_file)
conf = NetworkConfig(config_file)

hyperpar = {'coarse_dim': conf.COARSE_DIM,
            'dropout': conf.DROPOUT,
            'kernel_size': conf.KERNEL_SIZE,
            'activation': 'relu',
            'final_activation': None,
            'depth': 4}

model = SERENEt(img_shape1=(128, 128, 1), img_shape2=(128, 128, 1), params=hyperpar, path=path_model)
loaded_model = LoadModel(config_file)

#loaded_model.save_weights('%sweights_model-sem21cm_ep%d.h5' %(path_model+'checkpoints/', conf.BEST_EPOCH))

f = h5py.File('%sweights_model-sem21cm_ep%d.h5' %(path_model+'checkpoints/', conf.BEST_EPOCH))

for i, layer in enumerate(model.layers):
    if (len(layer.weights) != 0):
        #print(layer._name)
        layer_name = layer.name
        new_weight = [f[layer_name][layer_name][layer.weights[i_l].name.replace(layer_name+'/', '')][:] for i_l in range(len(layer.weights))]
        model.layers[i].set_weights(new_weight)

f.close()

loaded_model.save('%scheckpoints/model_tf2-sem21cm_ep%d.h5' %(path_model, conf.BEST_EPOCH))
