from lzma import MODE_FAST
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model

import sys
sys.path.insert(0,'../')
from utils_network.metrics import get_avail_metris
from config.net_config import NetworkConfig

def LoadModel(cfile):
    conf = NetworkConfig(cfile)
    
    path_out = conf.RESUME_PATH
    MODEL_EPOCH = conf.BEST_EPOCH
    METRICS = {m:get_avail_metris(m) for m in np.append(conf.LOSS, conf.METRICS)}
    model_loaded = load_model('%smodel-sem21cm_ep%d.tf' %(path_out+'checkpoints/', MODEL_EPOCH), custom_objects=METRICS)
    #model_loaded = load_model('%smodel-sem21cm_ep%d.h5' %(path_out+'checkpoints/', MODEL_EPOCH), custom_objects=METRICS)

    
    print(' Loaded model:\n %smodel-sem21cm_ep%d.tf' %(conf.RESUME_PATH, MODEL_EPOCH))
    return model_loaded


def UniqueRows(arr):
    """ Remove duplicate row array in 2D data 
            * arr (narray): array with duplicate row
        
        Example:
        >> d = np.array([[0,1,2],[0,1,2],[0,0,0],[0,0,2],[0,1,2]])
        >> UniqueRows(d) 
        
        array([[0, 0, 0],
                [0, 0, 2],
                [0, 1, 2]])
    """
    arr = np.array(arr)

    if(arr.ndim == 2):
        arr = np.ascontiguousarray(arr)
        unique_arr = np.unique(arr.view([('', arr.dtype)]*arr.shape[1]))
        new_arr = unique_arr.view(arr.dtype).reshape((unique_arr.shape[0], arr.shape[1]))
    elif(arr.ndim == 1):
        new_arr = np.array(list(dict.fromkeys(arr)))

    return new_arr


def IndependentOperation():
    ''' How many operation of flip and rotation can a cube have in SegUnet?, 
        each indipendent operation is considered as an additional rappresentation
        point of the same coeval data, so that it can be considered for errorbar 
        in SegUnet '''

    data = np.array(range(4**3)).reshape((4,4,4)) 
    permut_opt = [] 
    
    operations = [lambda a: a, np.fliplr, np.flipud, lambda a: np.flipud(np.fliplr(a)), lambda a: np.fliplr(np.flipud(a))] 
    axis = [0,1,2] 
    angl_rot = [0,1,2,3] 
    
    permut_idx = np.zeros((len(operations)*len(axis)*len(angl_rot), data.size)) 
    permut_tot = {'opt%d' %k:[] for k in range(len(operations)*len(axis)*len(angl_rot))} 
    
    i = 0 
    for iopt, opt in enumerate(operations): 
        cube = opt(data)
        for rotax in axis: 
            for rot in angl_rot: 
                ax_tup = [0,1,2] 
                ax_tup.remove(rotax)         
                permut_idx[i] = np.rot90(cube, k=rot, axes=ax_tup).flatten() 
                #permut_opt.append('opt%d rotax%d rot%d' %(iopt,rotax,rot)) 
                permut_tot['opt%d' %i] = [opt, rotax, rot] 
                i += 1 
    
    idx_iter = [] 
    for j in range(0,permut_idx.shape[0]-1): 
        for k in range(j+1, permut_idx.shape[0]): 
            if (all(permut_idx[j] == permut_idx[k])): 
                idx_iter.append(k) 
    
    idx_iter = np.array(list(UniqueRows(idx_iter))) 
    idx_opt = np.sort(np.array(range(i))[~idx_iter]) 
    
    permut_opt = {'opt%d' %k: permut_tot['opt%d' %val] for k, val in enumerate(idx_opt)} 
    
    return permut_opt


def IndependentOperation_LC():
    operations = [lambda a: a, np.fliplr, np.flipud, lambda a: np.flipud(np.fliplr(a))] 
    angl_rot = [0,1,2,3] 
    
    permut_op = {} 
    i = 0 
    for opt in operations: 
        for rot in angl_rot: 
            permut_op['opt%d' %i] = [opt, rot] 
            i += 1
    return permut_op


def Unet2Predict(unet, lc, tta=False, seg=True):
    if(np.ndim(lc) == 3):
        # for SegU-Net or RecU-Net
        assert lc.shape[0] == lc.shape[1]
        assert lc.shape[0] != lc.shape[2]
        x = np.moveaxis(lc, -1, 0)[...,np.newaxis]
    elif(np.ndim(lc) == 4):
        # for SERENEt architecture
        assert lc[0].shape[0] == lc[0].shape[1]
        assert lc[0].shape[0] != lc[0].shape[2]
        x = (np.moveaxis(lc[0], -1, 0)[...,np.newaxis], np.moveaxis(lc[1], -1, 0)[...,np.newaxis])

    if(tta):
        transf_opts = IndependentOperation_LC()
        if(np.ndim(lc) == 3):
            # for SegU-Net or RecU-Net
            ax_tup = [1, 2] 
            x_tta = np.zeros((len(transf_opts), lc.shape[0], lc.shape[1], lc.shape[2]))
        elif(np.ndim(lc) == 4):
            # for SERENEt architecture
            ax_tup = [0, 1] 
            x_tta = np.zeros((len(transf_opts), lc[0].shape[0], lc[0].shape[1], lc[0].shape[2]))

        for iopt in tqdm(range(len(transf_opts))):
            opt, rot = transf_opts['opt%d' %iopt]

            if(np.ndim(lc) == 3):
                x_pred = unet.predict(np.rot90(opt(x), k=rot, axes=ax_tup), verbose=0)
                transform_x = opt(np.rot90(x_pred.squeeze(), k=-rot, axes=ax_tup))
                x_tta[iopt] = np.moveaxis(transform_x, 0, -1)
            if(np.ndim(lc) == 4):        
                x = (np.moveaxis(np.rot90(opt(lc[0]), k=rot, axes=ax_tup), -1, 0)[...,np.newaxis], np.moveaxis(np.rot90(opt(lc[1]), k=rot, axes=ax_tup), -1, 0)[...,np.newaxis])
                x_pred = unet.predict(x, verbose=0)
                transform_x = opt(np.rot90(np.moveaxis(x_pred.squeeze(), 0, -1), k=-rot, axes=ax_tup))
                x_tta[iopt] = transform_x
    else:
        x_tta = unet.predict(x)
        x_tta = np.moveaxis(x_tta.squeeze(), 0, 2)

    if(seg):
        x_tta = np.clip(x_tta, 0, 1).round().astype(int)

    return x_tta
