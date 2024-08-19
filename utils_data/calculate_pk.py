import numpy as np
import tools21cm as t2c
import matplotlib.pyplot as plt
import gc

from matplotlib.colors import LogNorm
from custom_calc import compute_cylindrical_power_spectrum
from tqdm import tqdm

#from custom_calc import compute_cylindrical_power_spectrum

path_input = '/store/ska/sk014/serenet/inputs/dataLC_128_train_190922/'
path_out = '/scratch/snx3000/mibianco/output_sdc3/pk/'
data_typ = 'dT2'

box_len = 256. # cMpc
depth_mhz = 20. # MHz
redshifts = np.loadtxt(path_input+'lc_redshifts.txt')

kper_bins = np.linspace(0.0, 0.3535533905932738, 10)
kpar_bins = np.linspace(0.0, 0.2565419940064186, 10)

for i in tqdm(range(2380, 10000)):
    # load input and reionization lightcone
    y_true = t2c.read_cbin('%sdata/%s_21cm_i%d.bin' %(path_input, data_typ, i))
    xHI = t2c.read_cbin('%sdata/xHI_21cm_i%d.bin' %(path_input, i))

    # calculate reionization history
    mean_xHI = xHI.mean(axis=(0,1))

    # decide EoR milestones
    #eor_milestones = np.array([0.25, 0.5, 0.75])
    #indexes = np.array([np.argmin(np.abs(mean_xHI-xh)) for xh in eor_milestones])

    eor_milestones = np.array([8, 9, 10])
    indexes = np.array([np.argmin(np.abs(redshifts-xh)) for xh in eor_milestones])
    np.savetxt('%s_pk_i%d.txt' %(path_out+'ytrue', i), np.hstack((redshifts[indexes][...,None], mean_xHI[indexes][...,None])), fmt='%.3f\t%.5f', header='z\tmean_xHI')

    for j, z in enumerate(redshifts[indexes]):        
        # calculate sub-volume
        dT_sub, box_dims, cut_redshift = t2c.get_lightcone_subvolume(lightcone=y_true, redshifts=redshifts, central_z=z, depth_mhz=depth_mhz, odd_num_cells=False, subtract_mean=False, fov_Mpc=box_len, verbose=False)

        """
        # Create k grid
        kxy_vals = np.fft.fftshift(np.fft.fftfreq(dT_sub.shape[0], d=box_dims[0] / dT_sub.shape[0]))
        kz_vals = np.fft.fftshift(np.fft.fftfreq(dT_sub.shape[2], d=box_dims[2] / dT_sub.shape[2]))
        kx, ky, kz = np.meshgrid(kxy_vals, kxy_vals, kz_vals, indexing='ij')
        k_perp = np.sqrt(kx**2 + ky**2).flatten()
        k_parallel = np.abs(kz).flatten()
        
        # Define k_perp and k_parallel bins
        kper_bins = np.linspace(k_perp.min(), k_perp.max(), num=10)
        kpar_bins = np.linspace(k_parallel.min(), k_parallel.max(), num=10)
        """

        Pk, kper, kpar = t2c.power_spectrum_2d(dT_sub, kbins=[kper_bins, kpar_bins], box_dims=box_dims, nu_axis=2, window='blackmanharris')
        #Pk, kper, kpar = compute_cylindrical_power_spectrum(dT_sub, box_dims=box_dims)
        
        #fout = '%s_pk_i%d_x%d.txt' %(path_out+data_typ, i, eor_milestones[j]*100)
        fout = '%s_pk_i%d_z%d.txt' %(path_out+data_typ, i, eor_milestones[j])
        np.savetxt(fout, Pk)

        if(i % 1000 == 0):
            plt.title('z = %.3f   x$_{HI}$ = %.2f' %(z, mean_xHI[indexes][j]))
            plt.pcolormesh(kper, kpar, Pk.T, norm=LogNorm())
            plt.xscale("log"), plt.yscale("log")
            plt.xlim(kper.min(), kper.max()), plt.ylim(kpar.min(), kpar.max())
            plt.colorbar(label="P($k_{\perp}$, $k_{\parallel}$) [mK$^2$]")
            plt.xlabel(r"$k_{\perp}$ [Mpc$^{-1}$]")
            plt.ylabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
            plt.savefig(fout.replace('txt', 'png'), bbox_inches="tight"), plt.clf()

        del Pk, kper, kpar;
        gc.collect()
