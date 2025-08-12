import numpy as np
import tools21cm as t2c
import matplotlib.pyplot as plt
import gc

from sklearn.metrics import r2_score
from matplotlib.colors import LogNorm
from tqdm import tqdm

path_pred = '/store/ska/sk014/serenet/outputs_recunet/serenet01-05T17-59-07_128slice_priorTS/prediction/'
path_input = '/store/ska/sk014/serenet/inputs/dataLC_128_pred_190922/'
path_out = '/scratch/snx3000/mibianco/output_sdc3/pred_pk/'

box_len = 256. # cMpc
depth_mhz = 20. # MHz
redshifts = np.loadtxt(path_input+'lc_redshifts.txt')

# limit redshift to avoid redshift limit error
mask1 = redshifts >= 7.48
mask2 = redshifts <= 10.08
redshift_indexes = np.arange(redshifts.size)[mask2*mask1] 

# define bins
kpar_bins = np.linspace(8.5e-2, 1.62, 60)
kper_bins = np.linspace(5e-2, 0.6, 20)
kpar = 0.5*(kpar_bins[1:]+kpar_bins[:-1])
kper = 0.5*(kper_bins[1:]+kper_bins[:-1])
np.savetxt('%skpar.txt' %path_out, kpar)
np.savetxt('%skper.txt' %path_out, kper)

for i in tqdm(range(0, 300)):
    # load input and reionization lightcone
    y_pred = t2c.read_cbin('%spreddT2_from_dT4pca4_21cm_i%d.bin' %(path_pred, i))
    y_true = t2c.read_cbin('%sdata/dT2_21cm_i%d.bin' %(path_input, i))
    xHI = t2c.read_cbin('%sdata/xHI_21cm_i%d.bin' %(path_input, i))

    # calculate reionization history
    mean_xHI = xHI.mean(axis=(0,1))

    # decide EoR milestones
    rnd_indexes = np.random.choice(a=redshift_indexes, size=100)
    stat_data = np.array([rnd_indexes, redshifts[rnd_indexes], mean_xHI[rnd_indexes]])
    r2pk_array = np.zeros(rnd_indexes.size)
    r2dT_array = np.zeros(rnd_indexes.size)

    for j, idx in enumerate(rnd_indexes):
        z = redshifts[idx]
        
        # calculate sub-volume and Power Spectra
        dT_sub, box_dims, cut_redshift = t2c.get_lightcone_subvolume(lightcone=y_true, redshifts=redshifts, central_z=z, depth_mhz=depth_mhz, odd_num_cells=False, subtract_mean=False, fov_Mpc=box_len, verbose=False)
        Pk, kper, kpar = t2c.power_spectrum_2d(dT_sub, kbins=[kper_bins, kpar_bins], box_dims=box_dims, nu_axis=2, window='blackmanharris')
        #np.savetxt('%sdT2_pk_i%d_z%d.txt' %(path_out, i, idx), Pk)
        #del Pk, kper, kpar;
        """
        if(i % 1000 == 0):
            plt.title('z = %.3f   x$_{HI}$ = %.2f' %(z, mean_xHI[idx]))
            plt.pcolormesh(kper, kpar, Pk.T, norm=LogNorm())
            plt.xscale("log"), plt.yscale("log")
            plt.xlim(kper.min(), kper.max()), plt.ylim(kpar.min(), kpar.max())
            plt.colorbar(label="P($k_{\perp}$, $k_{\parallel}$) [mK$^2$]")
            plt.xlabel(r"$k_{\perp}$ [Mpc$^{-1}$]")
            plt.ylabel(r"$k_{\parallel}$ [Mpc$^{-1}$]")
            plt.savefig('%sdT2_pk_i%d_z%d.png' %(path_out, i, idx), bbox_inches="tight"), plt.clf()
        """
        gc.collect()

        # calculate sub-volume and Power Spectra
        dTpred_sub, _, _ = t2c.get_lightcone_subvolume(lightcone=y_pred, redshifts=redshifts, central_z=z, depth_mhz=depth_mhz, odd_num_cells=False, subtract_mean=False, fov_Mpc=box_len, verbose=False)
        Pk_pred, kper, kpar = t2c.power_spectrum_2d(dTpred_sub, kbins=[kper_bins, kpar_bins], box_dims=box_dims, nu_axis=2, window='blackmanharris')
        #np.savetxt('%spreddT2_from_dT4pca4_pk_i%d_z%d.txt' %(path_out, i, idx), Pk_pred)

        r2pk_array[j] = r2_score(y_true=Pk.flatten(), y_pred=Pk_pred.flatten())
        r2dT_array[j] = r2_score(y_true=dT_sub.flatten(), y_pred=dTpred_sub.flatten())

        del Pk, Pk_pred, kper, kpar;
        gc.collect()

    stat_data = np.vstack((stat_data, r2pk_array[None,...]))
    stat_data = np.vstack((stat_data, r2dT_array[None,...])).T
    np.savetxt('%s_pk_i%d.txt' %(path_out+'ytrue', i), stat_data, fmt='%d\t%.3f\t%.5f\t\t%.4f\t%.4f', header='i\tz\tmean_xHI\tR2(Pk)\tR2(dT)')
