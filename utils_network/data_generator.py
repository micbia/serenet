import math, random, numpy as np, os, sys
import pandas as pd

from datetime import datetime
from glob import glob
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """
    Data generator of 3D data (calculate noise cube and smooth data).
    """
    def __init__(self, path='./', data_temp=None, batch_size=None, data_shape=None, shuffle=True):
        """
        Arguments:
         tobs: int
                observational time, for noise calcuation.
        """
        self.path = path
        self.data_temp = data_temp
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.data_size = len(self.data_temp)
        self.on_epoch_end()
        
        #self.astro_par = np.loadtxt('%sparameters/astro_params.txt' %self.path, unpack=True)
        #with open('%sparameters/user_params.txt' %self.path,'r') as f:
        #    self.user_par = eval(f.read())

        #self.redshift = np.loadtxt('%slc_redshifts.txt' %self.path)

    def __len__(self):
        # number of batches
        return int(np.floor(self.data_size//self.batch_size))

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = self.data_temp
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _read_cbin(self, filename, bits=32, order='C', dimensions=3):
        # read binary file, with 3 integers as header
        assert(bits == 32 or bits == 64)
        f = open(filename)
        temp_mesh = np.fromfile(f, count=dimensions, dtype='int32')
        datatype = np.float32 if bits == 32 else np.float64
        data = np.fromfile(f, dtype=datatype, count=np.prod(temp_mesh))
        data = data.reshape(temp_mesh, order=order)
        return data

    def _rescaledata(self, arr, a=0, b=1):
        scaled_arr = (arr - np.min(arr))/(np.max(arr) - np.min(arr)) * (b-a) + a
        return scaled_arr.astype(np.float32)

    # TODO: implement tta with rotation and horzontal/vertical flip
    def _augment_data(self, data, rotate_angle, rot_axis, flip):
        if(len(self.data_shape) == 3):
            if rot_axis == 1:
                ax_tup = (1, 2)
            elif rot_axis == 2:
                ax_tup = (2, 0)
            elif rot_axis == 3:
                ax_tup = (0, 1)
            else:
                raise ValueError('rotate axis should be 1, 2 or 3')
            rotated_data = np.rot90(data, k=rotate_angle, axes=ax_tup)
        elif(len(self.data_shape) == 2):
            rotated_data = np.rot90(data, k=rotate_angle)
        
        flipped_data = np.flip(rotated_data, axis=flip)
        return rotated_data

# -------------------------------------------------------------------
class OneLightConeGenerator(DataGenerator):
    """
    Michele, 21 Sep 2021:
    Data generator of lightcone data meant for SegU-Net or RecU-Net with one input and one output.
    Change the data_type variable for selecting the target.
    """
    def __init__(self, path='./', data_temp=None, batch_size=None, data_shape=None, data_type=['dT4pca4', 'xH'], shuffle=False, tta=False):
        super().__init__(path, data_temp, batch_size, data_shape, shuffle)
        self.data_type = data_type
        self.tta = tta
        self.nr_sample = len(glob(self.path+'data/'+self.data_type[0]+'*'))
        
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.zeros((np.append(self.batch_size, self.data_shape)))
        y = np.zeros((np.append(self.batch_size, self.data_shape)))
        
        idx = 0
        dT = self._read_cbin(filename='%s%s_21cm_i%d_ch600-751.bin' %(self.path+'data/', self.data_type[0], idx), dimensions=3)
        xH = self._read_cbin(filename='%s%s_21cm_i%d_ch600-751.bin' %(self.path+'data/', self.data_type[1], idx), dimensions=3)

        for i, idx in enumerate(indexes):
            # apply manipolation on the LC data
            rseed_freq = np.random.randint(xH.shape[2])
            X[i], y[i] = self._lc_data(x=dT, y=xH, rseed2=rseed_freq)
            
        # add channel dimension
        X = X[..., np.newaxis]
        y = y[..., np.newaxis]

        return X, y

    def _lc_data(self, x, y, rseed2):
        if(len(self.data_shape) == 2):
            # for U-Net on slices
            dT_sampled = x[:, :, rseed2].astype(np.float32)
            xH_sampled = y[:, :, rseed2].astype(np.float32)
        elif(len(self.data_shape) == 3):
            # for 3D U-Net on frequency cube
            freq_size = self.data_shape[2]
            rseed2 = random.randint(freq_size, x.shape[-1]-(freq_size+1)) 

            dT_sampled = x[:, :, rseed2-freq_size//2 : rseed2+freq_size//2]
            xH_sampled = y[:, :, rseed2-freq_size//2 : rseed2+freq_size//2]
        else:
            raise ValueError('wrong dimension for data_shape: %d' %len(self.data_shape))
         
        if(self.data_shape != dT_sampled.shape):
            ix, iy = np.random.randint(low=self.data_shape[0]//2, high=dT_sampled.shape[0]-self.data_shape[0]//2, size=2)

            dT_sampled = dT_sampled[ix-self.data_shape[0]//2: ix+self.data_shape[0]//2, iy-self.data_shape[0]//2: iy+self.data_shape[0]//2]
            xH_sampled = xH_sampled[ix-self.data_shape[0]//2: ix+self.data_shape[0]//2, iy-self.data_shape[0]//2: iy+self.data_shape[0]//2]

        if(self.tta):
            rseed_rot = np.random.randint(0, 3)
            rseed_flip = np.random.choice([0, 1, (0,1)])
            dT_sampled = self._augment_data(data=dT_sampled, rotate_angle=rseed_rot, rot_axis=3, flip=rseed_flip)
            xH_sampled = self._augment_data(data=xH_sampled, rotate_angle=rseed_rot, rot_axis=3, flip=rseed_flip)
        
        #dT_sampled = self._rescaledata(arr=dT_sampled, a=1e-3, b=100.)
        #xH_sampled = self._rescaledata(arr=xH_sampled, a=1e-7, b=1.-1e-7)
        return dT_sampled, xH_sampled

class LightConeGenerator(DataGenerator):
    """
    Michele, 21 Sep 2021:
    Data generator of lightcone data meant for SegU-Net or RecU-Net with one input and one output.
    Change the data_type variable for selecting the target.
    """
    def __init__(self, path='./', data_temp=None, batch_size=None, data_shape=None, data_type=['dT4pca4', 'xH'], shuffle=False):
        super().__init__(path, data_temp, batch_size, data_shape, shuffle)
        self.data_type = data_type
        self.nr_sample = len(glob(self.path+'data/'+self.data_type[0]+'*'))
        
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.zeros((np.append(self.batch_size, self.data_shape)))
        y = np.zeros((np.append(self.batch_size, self.data_shape)))
        for i, idx in enumerate(indexes):
            
            #dT = self._read_cbin(filename='%s%s_21cm_i%d_ch600-751.bin' %(self.path+'data/', self.data_type[0], idx%self.nr_sample), dimensions=3)
            #xH = self._read_cbin(filename='%s%s_21cm_i%d_ch600-751.bin' %(self.path+'data/', self.data_type[1], idx%self.nr_sample), dimensions=3)
            dT = self._read_cbin(filename='%s%s_21cm_i%d.bin' %(self.path+'data/', self.data_type[0], idx%self.nr_sample), dimensions=3)
            xH = self._read_cbin(filename='%s%s_21cm_i%d.bin' %(self.path+'data/', self.data_type[1], idx%self.nr_sample), dimensions=3)

            # apply manipolation on the LC data
            X[i], y[i] = self._lc_data(x=dT, y=xH, rseed2=idx%dT.shape[-1])

        # add channel dimension
        X = X[..., np.newaxis]
        y = y[..., np.newaxis]

        return X, y

    def _lc_data(self, x, y, rseed2):
        if(len(self.data_shape) == 2):
            # for U-Net on slices
            dT_sampled = x[:, :, rseed2].astype(np.float32)
            xH_sampled = y[:, :, rseed2].astype(np.float32)
        elif(len(self.data_shape) == 3):
            # for 3D U-Net on frequency cube
            freq_size = self.data_shape[2]
            rseed2 = random.randint(freq_size, x.shape[-1]-(freq_size+1)) 

            dT_sampled = x[:, :, rseed2-freq_size//2 : rseed2+freq_size//2]
            xH_sampled = y[:, :, rseed2-freq_size//2 : rseed2+freq_size//2]
        else:
            raise ValueError('wrong dimension for data_shape: %d' %len(self.data_shape))
        
        if(self.data_shape != dT_sampled.shape):
            ix, iy = np.random.randint(low=self.data_shape[0]//2, high=dT_sampled.shape[0]-self.data_shape[0]//2, size=2)

            dT_sampled = dT_sampled[ix-self.data_shape[0]//2: ix+self.data_shape[0]//2, iy-self.data_shape[0]//2: iy+self.data_shape[0]//2]
            xH_sampled = xH_sampled[ix-self.data_shape[0]//2: ix+self.data_shape[0]//2, iy-self.data_shape[0]//2: iy+self.data_shape[0]//2]
            
        #dT_sampled = self._rescaledata(arr=dT_sampled, a=1e-3, b=100.)
        #xH_sampled = self._rescaledata(arr=xH_sampled, a=1e-7, b=1.-1e-7)
        return dT_sampled, xH_sampled

        
class LightConeGenerator_SERENEt(DataGenerator):
    """
    Data generator for lightcone data with RecUNet network
    """
    def __init__(self, path='./', data_temp=None, batch_size=None, data_shape=None, data_type='xH', shuffle=False):
        super().__init__(path, data_temp, batch_size, data_shape, shuffle)
        self.data_type = data_type

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X1 = np.zeros((np.append(self.batch_size, self.data_shape)))
        X2 = np.zeros((np.append(self.batch_size, self.data_shape)))
        y = np.zeros((np.append(self.batch_size, self.data_shape)))

        for i, idx in enumerate(indexes):
            # read LC
            dT3 = self._read_cbin(filename='%sdT4pca4_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
            #xH = self._read_cbin(filename='%sxH_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
            xH = self._read_cbin(filename='%spred_dT4pca4_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
            dT2 = self._read_cbin(filename='%sdT2_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
            
            # apply manipolation on the LC data
            X1[i], X2[i], y[i] = self._lc_data(x1=dT3, x2=xH, y1=dT2)

        # add channel dimension
        X1 = X1[..., np.newaxis]
        X2 = X2[..., np.newaxis]
        y = y[..., np.newaxis]
        return X1, X2, y

    def _lc_data(self, x1, x2, y1):
        if(len(self.data_shape) == 2):
            rseed2 = random.randint(0, x1.shape[-1]-1)
            x1_sampled = x1[:, :, rseed2].astype(np.float32)
            x2_sampled = x2[:, :, rseed2].astype(np.float32)
            y1_sampled = y1[:, :, rseed2].astype(np.float32)

        elif(len(self.data_shape) == 3):
            # for 3D U-Net on frequency cube
            freq_size = self.data_shape[2]
            rseed2 = random.randint(freq_size, x1.shape[-1]-(freq_size+1)) 

            x1_sampled = x1[:, :, rseed2-freq_size//2 : rseed2+freq_size//2]
            x2_sampled = x1[:, :, rseed2-freq_size//2 : rseed2+freq_size//2]
            y1_sampled = y1[:, :, rseed2-freq_size//2 : rseed2+freq_size//2]

        #dT_sampled = self._rescaledata(arr=dT_sampled, a=1e-3, b=100.)
        #xH_sampled = self._rescaledata(arr=xH_sampled, a=1e-7, b=1.-1e-7)
        return x1_sampled, x2_sampled, y1_sampled


class LightConeGenerator_FullSERENEt(DataGenerator):
    """
    Data generator fro lightcone data for SERENEt network 
    """
    def __init__(self, path='./', data_temp=None, batch_size=None, data_shape=None, shuffle=False):
        super().__init__(path, data_temp, batch_size, data_shape, shuffle)
        
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        X = np.zeros((np.append(self.batch_size, self.data_shape)))
        y1 = np.zeros((np.append(self.batch_size, self.data_shape)))
        y2 = np.zeros((np.append(self.batch_size, self.data_shape)))

        for i, idx in enumerate(indexes):
            # read LC
            dT3 = self._read_cbin(filename='%sdT3_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
            dT2 = self._read_cbin(filename='%sdT2_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
            xH = self._read_cbin(filename='%sxH_21cm_i%d.bin' %(self.path+'data/', idx), dimensions=3)
            
            # apply manipolation on the LC data
            X[i], y1[i], y2[i] = self._lc_data(x=dT3, y1=dT2, y2=xH)
            
        # add channel dimension
        X = X[..., np.newaxis]
        y1 = y1[..., np.newaxis]
        y2 = y2[..., np.newaxis]

        return X, y1, y2

    def _lc_data(self, x, y1, y2):
        # for U-Net on slices
        rseed2 = random.randint(0, x.shape[-1]-1)
        x_sampled = x[:, :, rseed2].astype(np.float32)
        y1_sampled = y1[:, :, rseed2].astype(np.float32)
        y2_sampled = y2[:, :, rseed2].astype(np.float32)

        #x_sampled = self._rescaledata(arr=dT_sampled, a=1e-3, b=100.)
        #y1_sampled = self._rescaledata(arr=xH_sampled, a=1e-7, b=1.-1e-7)
        return x_sampled, y1_sampled, y2_sampled




