import numpy as np, os
from sklearn.metrics import r2_score
from glob import glob

# add SERENEt directory to path
import sys
sys.path.append("/users/mibianco/codes/serenet")
from utils_network.networks import Unet, SERENEt
from config.net_config import NetworkConfig
from utils_pred.prediction import Unet2Predict, LoadModel
from utils.other_utils import read_cbin, save_cbin
from tqdm import tqdm

path = '/project/c31/SKA_low_images/lightcones/EOS16/'
path_out = '/scratch/snx3000/mibianco/pred_eos/'
redshift = np.loadtxt(path+'redshift_EOS16_EoR.txt')

dT4pca = read_cbin(path+'dT4pca4_eos16.bin')
mask_xH = read_cbin(path+'xH_eos16.bin')
dT2 = read_cbin(path+'dT2_eos16.bin')

path_model = '/store/ska/sk014/serenet/outputs_segunet/segunet24-09T23-36-45_128slice_paper2/'
#path_model = '/store/ska/sk014/serenet/outputs_recunet/recunet17-05T09-23-54_128slice/'
#path_model = '/store/ska/sk014/serenet/outputs_recunet/serenet24-04T16-32-44_128slice_priorGT/'
#path_model = '/store/ska/sk014/serenet/outputs_recunet/serenet01-05T17-59-07_128slice_priorTS/'

config_file = glob(path_model+'*.ini')[0]
print(config_file)
conf = NetworkConfig(config_file)
model = LoadModel(config_file)     # TODO: on Piz Daint there is an issue with Tensorflow
"""
# --- Custom Load Model ----
hyperpar = {'coarse_dim': conf.COARSE_DIM,
            'dropout': conf.DROPOUT,
            'kernel_size': conf.KERNEL_SIZE,
            'activation': 'relu',
            'final_activation': None,
            'depth': 4}

# load emtpy model
model = SERENEt(img_shape1=(128, 128, 1), img_shape2=(128, 128, 1), params=hyperpar, path=path_model)

# open and load weights stored in my local
f = h5py.File('%sweights_model-sem21cm_ep%d.h5' %(path_model+'checkpoints/', conf.BEST_EPOCH))

# loop on the layers of the empty model and overwrite weights
for i, layer in enumerate(model.layers):
    if (len(layer.weights) != 0):
        #print(layer._name)
        layer_name = layer.name
        new_weight = [f[layer_name][layer_name][layer.weights[i_l].name.replace(layer_name+'/', '')][:] for i_l in range(len(layer.weights))]
        model.layers[i].set_weights(new_weight)

f.close()

#loaded_model.save('%scheckpoints/model_tf2-sem21cm_ep%d.h5' %(path_model, conf.BEST_EPOCH))
# -------------------------
"""

print('calculate overlapping map...')
overlap_mask = np.zeros((dT2.shape[0],dT2.shape[1]))
overlp = 4
for i_x in tqdm(range(0, overlap_mask.shape[0]-128, overlp)):
    for i_y in range(0, overlap_mask.shape[0]-128, overlp):
        overlap_mask[i_x:i_x+128,i_y:i_y+128] += np.ones((128,128))
error = False
fout = '%spredxH_from_dT4pca4_eos16.bin' %path_out
#fout = '%spreddT2_from_dT4pca4_eos16.bin' %path_out
ferr = '%serror_dT4pca4_eos16.bin' %path_out
ftta = '%stta_dT4pca4_eos16.bin' %path_out

print('calcualte prediction...')
if(not os.path.exists(fout)):
    y_pred = np.zeros_like(dT2)
    for i_x in tqdm(range(0, overlap_mask.shape[0]-128, overlp)):
        for i_y in range(0, overlap_mask.shape[0]-128, overlp):
            # get input
            x_input = dT4pca[i_x:i_x+128,i_y:i_y+128,:]
            #TODO: cut also the binary mask and then combine togheter for serenet network

            # get prediction
            y_tta = Unet2Predict(unet=model, lc=x_input, tta=error, seg=False)
            y_pred[i_x:i_x+128,i_y:i_y+128,:] += y_tta.astype(float)
            fout = '%spredictions/predxH_from_dT4pca4_eos16_x%dy%d.bin' %(path_out, i_x, i_y)
            np.save(fout, y_pred)
    #for i_z in range(dT2.shape[-1]):
    #    y_pred[...,i_z] /= overlap_mask

else:
    #y_pred = t2c.read_cbin(fout)
    pass
