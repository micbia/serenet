import numpy as np, matplotlib.pyplot as plt, os, sys
import matplotlib
import matplotlib.ticker as plticker
import tensorflow as tf

from tqdm import tqdm
from glob import glob
import matplotlib.gridspec as gridspec

from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from utils.other_utils import read_cbin, save_cbin
from config.net_config import NetworkConfig
from utils_pred.prediction import Unet2Predict, LoadModel

def inverse_normalized_mean_absolute_error(y_true, y_pred):
    return 1 - np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(y_true - np.mean(y_true)))

config_file = sys.argv[1]
conf = NetworkConfig(config_file)

title_a = '\t\t _    _ _   _      _   \n\t\t| |  | | \ | |    | |  \n\t\t| |  | |  \| | ___| |_ \n\t\t| |  | | . ` |/ _ \ __|\n\t\t| |__| | |\  |  __/ |_ \n\t\t \____/|_| \_|\___|\__|\n'
title_b = ' _____              _ _      _         ___  __                \n|  __ \            | (_)    | |       |__ \/_ |               \n| |__) | __ ___  __| |_  ___| |_ ___     ) || | ___ _ __ ___  \n|  ___/ `__/ _ \/ _` | |/ __| __/ __|   / / | |/ __| `_ ` _ \ \n| |   | | |  __/ (_| | | (__| |_\__ \  / /_ | | (__| | | | | |\n|_|   |_|  \___|\__,_|_|\___|\__|___/ |____||_|\___|_| |_| |_|\n'
print(title_a+'\n'+title_b)

PLOT_STATS, PLOT_VISUAL, PLOT_ERROR, PLOT_SCORE = True, True, True, True

pred_idx = np.arange(300)
#pred_idx = np.array([232])

path_pred = '/store/ska/sk014/serenet/inputs/dataLC_128_pred_190922/'
path_model = conf.RESUME_PATH
#path_out = path_model+'prediction/'
path_out = '/scratch/snx3000/mibianco/output_serenet/'
try:
    os.makedirs(path_out)
except:
    pass

# load redshift
redshift = np.loadtxt('%slc_redshifts.txt' %path_pred)
    
# Cosmology and Astrophysical parameters
with open(path_pred+'parameters/user_params.txt', 'r') as file:
    params = eval(file.read())
    dr_xy = np.linspace(0, params['BOX_LEN'], params['HII_DIM'])
with open(path_pred+'parameters/cosm_params.txt', 'r') as file:
    c_params = eval(file.read())

astro_params = np.loadtxt('%sparameters/astro_params.txt' %path_pred)

# Load best model
model = LoadModel(config_file)
#model_segunet = LoadModel('/store/ska/sk014/serenet/outputs_segunet/all24-09T23-36-45_128slice/net_Unet_lc.ini')
nr = 4

# Prediction loop
for ii in tqdm(range(pred_idx.size)):
    i_pred = pred_idx[ii]
    idx, zeta, Rmfp, Tvir, rseed = astro_params[i_pred]
    a_params = {'HII_EFF_FACTOR':zeta, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

    # Load input and target
    x_input = read_cbin('%sdata/dT4pca%d_21cm_i%d.bin' %(path_pred, nr, i_pred))
    #x_prior = read_cbin('%sdata/xH_21cm_i%d.bin' %(path_pred, i_pred))
    x_prior = read_cbin('/store/ska/sk014/serenet/outputs_segunet/segunet24-09T23-36-45_128slice_paper2/prediction/pred_dT4pca4_21cm_i%d.bin' %i_pred)
    #x_prior = Unet2Predict(unet=model_segunet, lc=x_input, tta=False, seg=False)
    x_prior = np.clip(x_prior, 0, 1)

    x_input = (x_input, x_prior)
    y_true = read_cbin('%sdata/dT2_21cm_i%d.bin' %(path_pred, i_pred))
    xHI = read_cbin('%sdata/xHI_21cm_i%d.bin' %(path_pred, i_pred))

    # Prediction on dataset
    if not os.path.exists('%spreddT2_from_dT4pca4_21cm_i%d.bin' %(path_out, i_pred)):
        if(PLOT_ERROR):
            y_tta = Unet2Predict(unet=model, lc=x_input, tta=PLOT_ERROR, seg=False)
            y_pred = np.mean(y_tta, axis=0)
            y_error = np.std(y_tta, axis=0)
            save_cbin('%serrordT2_from_dT4pca4_21cm_i%d.bin' %(path_out, i_pred), y_error)
        else:
            y_tta = Unet2Predict(unet=model, lc=x_input, tta=PLOT_ERROR, seg=False)
            y_pred = y_tta.squeeze()
        assert y_true.shape == y_pred.shape
        save_cbin('%spreddT2_from_dT4pca4_21cm_i%d.bin' %(path_out, i_pred), y_pred)
    else:
        y_pred = read_cbin('%spreddT2_from_dT4pca4_21cm_i%d.bin' %(path_out, i_pred))
        if(PLOT_ERROR):
            y_error = read_cbin('%serrordT2_from_dT4pca4_21cm_i%d.bin' %(path_out, i_pred))

    # Statistical quantities
    r2 = np.array([r2_score(y_true[...,i].flatten(), y_pred[...,i].flatten()) for i in range(y_true.shape[-1])])
    if(PLOT_ERROR):
        err_r2 = np.array([np.mean(np.sqrt(((2*(y_pred[...,i] - y_pred[...,i].mean())*np.sum((y_pred[...,i] - y_true[...,i])**2)-2*(y_pred[...,i] - y_true[...,i])*np.sum((y_pred[...,i] - y_pred[...,i].mean())**2))/np.sum((y_pred[...,i] - y_pred[...,i].mean())**2)**2*y_error[...,i])**2 + (y_error[...,i]/y_error[...,i].size)**2)) for i in range(y_true.shape[-1])])
    inmae = np.array([inverse_normalized_mean_absolute_error(y_true[...,i], y_pred[...,i]) for i in range(y_true.shape[-1])])
    #rpearson = np.array([pearsonr(y_true[...,i].flatten(), y_pred[...,i].flatten()).statistic for i in range(y_true.shape[-1])])
    mse = np.array([mean_squared_error(y_true[...,i].flatten(), y_pred[...,i].flatten()) for i in range(y_true.shape[-1])])
    mae = np.array([mean_absolute_error(y_true[...,i].flatten(), y_pred[...,i].flatten()) for i in range(y_true.shape[-1])])
    mean_xHI = np.mean(xHI, axis=(0,1))

    #stat = np.nan_to_num(np.array([redshift, r2, rpearson, inmae, mse, mae]).T)
    #np.savetxt('%sstats_i%d.txt' %(path_out, i_pred), stat, fmt='%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f', header='eff_fact=%.4f\tRmfp=%.4f\tTvir=%.4e\t\nz\tR^2\tPearson\tINMAE\tMSE\tMAE' %(zeta, Rmfp, Tvir))
    stat = np.nan_to_num(np.array([redshift, r2, inmae, mse, mae]).T)
    np.savetxt('%sstats_i%d.txt' %(path_out, i_pred), stat, fmt='%.3f\t%.3f\t%.3f\t%.3f\t%.3f', header='eff_fact=%.4f\tRmfp=%.4f\tTvir=%.4e\t\nz\tR^2\tINMAE\tMSE\tMAE' %(zeta, Rmfp, Tvir))

    xHI_plot = np.arange(0.1, 1., 0.1)
    p = np.poly1d(np.polyfit(redshift, mean_xHI, 6))    # fit a polynomial to make the reionization history smoother
    redshift_plot = np.array([redshift[np.argmin(abs(p(redshift) - meanHI))] for meanHI in xHI_plot])

    if(PLOT_STATS):
        # PLOT STATS
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.size'] = 16

        fig, axs = plt.subplots(figsize=(15, 5), ncols=2, nrows=1)
        axs[0].plot(redshift, r2, label=r'R$^2$', lw=0.9)
        #axs[0].plot(redshift, inmae, label='INMAE', lw=0.9)
        #axs[0].plot(redshift, rpearson, label=r'R$^{\rm Pearson}$', lw=0.9)
        if(PLOT_ERROR):
            r2error_low = np.clip(r2-err_r2, 0, (r2-err_r2).max())
            r2error_up = np.clip(err_r2+r2, (err_r2+r2).min(), 1)
            axs[0].fill_between(redshift, r2error_low, r2error_up, color='lightblue', alpha=0.8)
        axs[0].legend()
        axs[0].vlines(redshift_plot, ymin=0, ymax=1, color='black', ls='--', lw=0.5)
        for iplot in range(redshift_plot.size):
            axs[0].text(redshift_plot[iplot]+0.03, 0.03, round(xHI_plot[iplot], 1), rotation=90, size=12, alpha=0.8)
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel('Score [%]'), axs[1].set_ylabel('Distance [mK]')

        axs[1].plot(redshift, mse, label='mse')
        axs[1].plot(redshift, mae, label='mae')        
        axs[1].legend()

        for ax in axs.flatten():
            ax.set_xlabel('z')

        plt.savefig('%sstats_i%d.png' %(path_out, i_pred), bbox_inches='tight'), plt.clf()
        
    if(PLOT_VISUAL):
        # Visual Plot
        i_slice = np.argmin(abs(mean_xHI  - 0.5))
        i_lc = params['HII_DIM']//2

        plt.rcParams['font.size'] = 20
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['axes.linewidth'] = 1.2

        if(PLOT_ERROR):
            fig = plt.figure(figsize=(27, 25), constrained_layout=True)
            gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[3, 1], height_ratios=[1, 1, 1, 1], hspace=0.1, wspace=0.05)
        else:
            fig = plt.figure(figsize=(27, 20), constrained_layout=True)
            gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[3, 1], height_ratios=[1, 1, 1], hspace=0.1, wspace=0.05)

        # FIRST LC PLOT
        ax0 = fig.add_subplot(gs[0,0])
        im = ax0.pcolormesh(redshift, dr_xy, x_input[0][:,i_lc,:], cmap='jet', vmin=x_input[0].min(), vmax=x_input[0].max())
        ax0.contour(redshift, dr_xy, x_input[1][:,i_lc,:])
        ax0.set_ylabel('y [Mpc]', size=20)

        # FIRST SLICE PLOT
        ax01 = fig.add_subplot(gs[0,1])
        ax01.pcolormesh(dr_xy, dr_xy, x_input[0][...,i_slice], cmap='jet', vmin=x_input[0].min(), vmax=x_input[0].max())
        ax01.contour(dr_xy, dr_xy, x_input[1][...,i_slice])
        fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax01, pad=0.01, fraction=0.048)
        ax0.label_outer(), ax01.label_outer()

        # SECOND LC PLOT
        ax1 = fig.add_subplot(gs[1,0])
        ax1.label_outer()
        ax1.pcolormesh(redshift, dr_xy, y_true[:,i_lc,:], cmap='jet', vmin=y_true.min(), vmax=y_true.max())
        #ax1.contour(redshift, dr_xy, x_input[1][:,i_lc,:])
        ax1.set_ylabel('y [Mpc]', size=20)

        # SECOND SLICE PLOT
        ax11 = fig.add_subplot(gs[1,1])
        ax11.set_title(r'$z$ = %.3f   $\bar{x}_{HI}=%.2f$' %(redshift[i_slice], mean_xHI[i_slice]), fontsize=20)
        im = ax11.pcolormesh(dr_xy, dr_xy, y_true[...,i_slice], cmap='jet', vmin=y_true.min(), vmax=y_true.max())
        #ax11.contour(dr_xy, dr_xy, x_input[1][...,i_slice])
        fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax11, pad=0.01, fraction=0.048)
        ax1.label_outer(), ax11.label_outer()

        # THIRD LC PLOT
        ax2 = fig.add_subplot(gs[2,0])
        ax2.pcolormesh(redshift, dr_xy, y_pred[:,i_lc,:], cmap='jet', vmin=y_true.min(), vmax=y_true.max())
        #ax1.contour(redshift, dr_xy, x_input[1][:,i_lc,:])
        ax2.set_ylabel('y [Mpc]', size=20)

        # THIRD SLICE PLOT
        ax21 = fig.add_subplot(gs[2,1])
        ax21.set_title(r'$R^2$ = %.1f %%' %(100*r2[i_slice]), fontsize=20)
        im = ax21.pcolormesh(dr_xy, dr_xy, y_pred[...,i_slice], cmap='jet', vmin=y_true.min(), vmax=y_true.max())
        #ax21.contour(dr_xy, dr_xy, x_input[1][...,i_slice])
        fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax21, pad=0.01, fraction=0.048)
        ax2.label_outer(), ax21.label_outer()

        if(PLOT_ERROR):
            # FOURTH LC PLOT
            ax3 = fig.add_subplot(gs[3,0])
            ax3.pcolormesh(redshift, dr_xy, y_error[:,i_lc,:], cmap='Oranges', vmin=0, vmax=y_error[:,i_lc,:].max())
            #ax3.contour(redshift, dr_xy, x_input[1][:,i_lc,:])
            ax3.set_ylabel('y [Mpc]', size=20)

            # FOURTH SLICE PLOT
            ax31 = fig.add_subplot(gs[3,1])
            im = ax31.pcolormesh(dr_xy, dr_xy, y_error[...,i_slice], cmap='Oranges', vmin=0, vmax=y_error[:,i_lc,:].max())
            #ax31.contour(dr_xy, dr_xy, x_input[1][...,i_slice])
            fig.colorbar(im, label=r'$\sigma_{std}$', ax=ax31, pad=0.01, fraction=0.048)
            ax3.label_outer(), ax31.label_outer()

            ax31.set_xlabel('x [Mpc]', size=20)
            ax3.set_xlabel('z', size=20)

            axes1 = [ax01, ax11, ax21, ax31]
            axes2 = [ax0, ax1, ax2, ax3]
        else:
            ax21.set_xlabel('x [Mpc]', size=20)
            ax2.set_xlabel('z', size=20)
            axes1 = [ax01, ax11, ax21]
            axes2 = [ax0, ax1, ax2]
        
        for axis1, axis2 in zip(axes1, axes2):
            axis1.tick_params(axis='both', which='major', length=10, width=2)
            axis1.tick_params(axis='both', which='minor', length=5, width=2)
            axis2.tick_params(axis='both', which='major', length=10, width=2)
            axis2.tick_params(axis='both', which='minor', length=5, width=2)
            axis1.xaxis.set_minor_locator(plticker.MultipleLocator(base=25))
            axis1.xaxis.set_major_locator(plticker.MultipleLocator(base=50))
            axis1.yaxis.set_minor_locator(plticker.MultipleLocator(base=25))
            axis2.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.25))
            axis2.yaxis.set_minor_locator(plticker.MultipleLocator(base=25))
            ax.label_outer()

        plt.savefig('%svisual_i%d.png' %(path_out, i_pred), bbox_inches='tight')
        plt.clf()

    if(PLOT_SCORE):
        if(ii == 0):
            fig1, ax_s = plt.subplots(figsize=(10,8), ncols=1)

        # get redshift color
        cm = matplotlib.cm.plasma
        sc = ax_s.scatter(mean_xHI, r2, c=redshift, vmin=redshift.min(), vmax=redshift.max(), s=25, cmap=cm, marker='.')
        norm = matplotlib.colors.Normalize(vmin=7, vmax=9, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
        redshift_color = np.array([(mapper.to_rgba(v)) for v in redshift])

        #if(PLOT_ERROR):
        #    for x, y, e, clr, red in zip(mean_true, mcc, mcc_err, redshift_color, redshift):
        #        ax_s.errorbar(x=x, y=y, yerr=e, lw=1, marker='o', capsize=1, color=clr)

        ax_s.set_xlim(mean_xHI.min()-0.02, mean_xHI.max()+0.02), ax_s.set_xlabel(r'$\rm x^v_{HI}$', size=20)
        ax_s.set_ylim(-0.02, 1.02), ax_s.set_ylabel(r'$\rm R^2$', size=20)
        ax_s.set_yticks(np.arange(0, 1.1, 0.1))
        ax_s.set_xticks(np.arange(0, 1.1, 0.2))
        if(ii == pred_idx.size-1):
            #ax_s.hlines(y=np.mean(r2), xmin=-0.02, xmax=1.1, ls='--', label=r'$R^2$ = %.1f' %(100*np.mean(r2)), alpha=0.8, color='tab:blue', zorder=3)
            fig1.colorbar(sc, ax=ax_s, pad=0.01, label=r'$\rm z$')
            fig1.savefig('%sr2score_dataset.png' %path_out, bbox_inches='tight')
            plt.clf()

print('... done.')

