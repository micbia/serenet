import numpy as np, os
import astropy.constants as cst
import astropy.units as u

from astropy.cosmology import Planck18 as cosmo
from tqdm import tqdm

path_input = '/scratch/snx3000/mibianco/sdc3b/run_256Mpc_128_1e10/'

path_sources = path_input+'sources/'
path_pyc2ray = path_input+'result_pyC2Ray_ts025_zf/'
path_21cmfast = path_input+'result_21cmFAST/'

redshift = np.loadtxt(path_input+'redshifts.txt')
boxsize = 256 # cMpc



def r_mfp(z):
    return (cst.c/cosmo.H(z)).to('Mpc') * 0.1 * np.power((1+z)/4, -2.55)

def fake_simulator(pos, shape, z, res):
    cube = np.zeros(shape)
    xx, yy, zz = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), sparse=True)
    
    # radius
    r  = int(round(r_mfp(z).value*(1+z)/res, 0)) # proper to comoving
    r2 = (xx-shape[0]/2)**2+(yy-shape[1]/2)**2+(zz-shape[2]/2)**2

    # Bubble 0
    xx0, yy0, zz0 = int(shape[0]/2), int(shape[1]/2), int(shape[2]/2)
    cube0 = np.zeros(shape)
    cube0[r2<=r**2] = 1
    cube0 = np.roll(np.roll(np.roll(cube0, -xx0, axis=0), -yy0, axis=1), -zz0, axis=2)

    # Bubble 1
    xx1, yy1, zz1 = pos[0], pos[1], pos[2]
    cube = cube+np.roll(np.roll(np.roll(cube0, xx1, axis=0), yy1, axis=1), zz1, axis=2)
    return cube

def fake_simulator2(pos, shape, z, res):
    radius  = int(round(r_mfp(z).value*(1+z)/res, 0)) # proper to comoving
    semisizes = (radius,) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(pos, shape)]
    pos = np.ogrid[grid]
    arr = np.zeros(shape, dtype=float)
    
    for x_i, semisize in zip(pos, semisizes):
        arr += (x_i / semisize) ** 2

    return arr <= 1.0

def Sphere(r=50):
    data = sphere((128, 128, 128), r, (64, 64, 64))
    return data.astype(float)


for i_z in tqdm(range(redshift.size)):
    z = redshift[i_z]

    if not os.path.exists('%ssrcmask_z%.3f.npy' %(path_sources, z)):
        nion = np.load('%snion_z%.3f.npy' %(path_sources, z))
        h_pos = np.array(np.nonzero(nion)).T

        cube = np.zeros_like(nion)
        for pos in h_pos:
            cube += fake_simulator2(pos=pos, shape=nion.shape, z=z, res=2.0)
        np.save('%ssrcmask_z%.3f.npy' %(path_sources, z), cube)