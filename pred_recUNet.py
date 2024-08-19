import numpy as np, matplotlib.pyplot as plt, os, gc
import matplotlib
import tools21cm as t2c
import tensorflow as tf

from tqdm import tqdm
from glob import glob
import matplotlib.gridspec as gridspec

from sklearn.metrics import matthews_corrcoef, r2_score
from sklearn.metrics import confusion_matrix

from utils_pred.prediction import Unet2Predict, LoadModel


title_a = '\t\t _    _ _   _      _   \n\t\t| |  | | \ | |    | |  \n\t\t| |  | |  \| | ___| |_ \n\t\t| |  | | . ` |/ _ \ __|\n\t\t| |__| | |\  |  __/ |_ \n\t\t \____/|_| \_|\___|\__|\n'
title_b = ' _____              _ _      _         ___  __                \n|  __ \            | (_)    | |       |__ \/_ |               \n| |__) | __ ___  __| |_  ___| |_ ___     ) || | ___ _ __ ___  \n|  ___/ `__/ _ \/ _` | |/ __| __/ __|   / / | |/ __| `_ ` _ \ \n| |   | | |  __/ (_| | | (__| |_\__ \  / /_ | | (__| | | | | |\n|_|   |_|  \___|\__,_|_|\___|\__|___/ |____||_|\___|_| |_| |_|\n'
print(title_a+'\n'+title_b)

#PLOT_STATS, PLOT_MEAN, PLOT_VISUAL, PLOT_ERROR, PLOT_SCORE = True, True, True, True, True
PLOT_STATS, PLOT_MEAN, PLOT_VISUAL, PLOT_ERROR, PLOT_SCORE = True, True, True, False, True

#path_pred = '/store/ska/sk09/segunet/inputs/dataLC_128_pred_310822/'
#path_pred = '/store/ska/sk09/segunet/inputs/dataLC_128_pred_190922/'
#path_pred = '/store/ska/sk02/lightcones/EOS16/EOS16_dataset/'
path_pred = '/store/ska/sk014/segunet/inputs/dataLC_128_train_190922/'

#pred_idx = np.arange(1925, 10000)
pred_idx = np.arange(1)
#pred_idx = np.arange(64)

path_model = '/scratch/snx3000/mibianco/output_segunet/outputs/10-01T17-21-17_128slice/'
config_file = path_model+'net_RecUnet_lc.ini'
path_out = path_model+'prediction/'

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
nr = 4

# Prediction loop
for ii in tqdm(range(pred_idx.size)):
    i_pred = pred_idx[ii]
    idx, zeta, Rmfp, Tvir, rseed = astro_params[i_pred]
    a_params = {'HII_EFF_FACTOR':zeta, 'R_BUBBLE_MAX':Rmfp, 'ION_Tvir_MIN':Tvir}

    if(os.path.exists('%spred_dT4pca4_21cm_i%d.bin' %(path_out, i_pred))):
        continue

    # Load input and target
    #wanted_z = 7
    #redshift = np.array([wanted_z]*300)
    #x_input = np.load('/store/ska/sk09/mibianco/dataLC_128_pred_190922/data/dT4pca4_z%d.npy' %wanted_z)
    #x_input = np.moveaxis(x_input, 0, -1)
    #y_true = np.load('/store/ska/sk09/mibianco/dataLC_128_pred_190922/data/xH_z%d.npy' %wanted_z)
    #y_true = np.moveaxis(y_true, 0, -1)
    #xHI = np.load('/store/ska/sk09/mibianco/dataLC_128_pred_190922/data/xHI_z%d.npy' %wanted_z)
    #xHI = np.moveaxis(xHI, 0, -1)

    x_input = t2c.read_cbin('%sdata/dT4pca%d_21cm_i%d.bin' %(path_pred, nr, i_pred))
    y_true = t2c.read_cbin('%sdata/dT2_21cm_i%d.bin' %(path_pred, i_pred))
    xHI = t2c.read_cbin('%sdata/xHI_21cm_i%d.bin' %(path_pred, i_pred))

    # Prediction on dataset
    y_tta = Unet2Predict(unet=model, lc=x_input, tta=PLOT_ERROR)

    if(PLOT_ERROR):
        y_pred = np.round(np.mean(y_tta, axis=0))
        y_error = np.std(y_tta, axis=0)
        t2c.save_cbin('%spred_dT4pca4_21cm_i%d.bin' %(path_out, i_pred), y_pred)
        t2c.save_cbin('%serror_dT4pca4_21cm_i%d.bin' %(path_out, i_pred), y_error)
        #np.save('%sdata/ttaxH_21cm_i%d.npy' %(path_out, i_pred), y_tta)
        y_tta = np.round(y_tta)

        TP_tta = np.sum(y_tta * y_true[np.newaxis,...], axis=(1,2))
        TN_tta = np.sum((1-y_tta) * (1-y_true[np.newaxis,...]), axis=(1,2))
        FP_tta = np.sum(y_tta * (1-y_true[np.newaxis,...]), axis=(1,2))
        FN_tta = np.sum((1-y_tta) * y_true[np.newaxis,...], axis=(1,2))

        TP_err = np.std(TP_tta, axis=0)
        TN_err = np.std(TN_tta, axis=0)
        FP_err = np.std(FP_tta, axis=0)
        FN_err = np.std(FN_tta, axis=0)
        acc_err = np.std((TP_tta*TN_tta - FP_tta*FN_tta) / (np.sqrt((TP_tta+FP_tta)*(TP_tta+FN_tta)*(TN_tta+FP_tta)*(TN_tta+FN_tta)) + tf.keras.backend.epsilon()), axis=0)

        mean_error = np.std(np.mean(y_tta, axis=(1,2)), axis=0)
    else:
        #y_pred = np.round(np.clip(y_tta.squeeze(), 0, 1))
        y_pred = y_tta.squeeze()
        t2c.save_cbin('%spredxH_21cm_i%d.bin' %(path_out, i_pred), y_pred)
    
    assert x_input.shape == y_pred.shape
    del y_tta; gc.collect()

    # Statistical quantities
    TP = np.sum(y_pred * y_true, axis=(0,1))
    TN = np.sum((1-y_pred) * (1-y_true), axis=(0,1))
    FP = np.sum(y_pred * (1-y_true), axis=(0,1))
    FN = np.sum((1-y_pred) * y_true, axis=(0,1))
    #TN, FP, FN, TP = confusion_matrix(y_true[..., 0], y_pred[..., 0]).ravel()

    FNR = FN/(FN+TP)
    FPR = FP/(FP+TN)
    TNR = TN/(FP+TN)   # a.k.a specificy
    TPR = TP/(TP+FP)   # a.k.a precision
    acc = np.array([r2_score(y_true=y_true[...,k].flatten(), y_pred=y_pred[...,k].flatten()) for k in range(x_input.shape[-1])])

    #rec = TP/(TP+FN)    # very similar to precision
    iou = TP/(TP+FP+FN)
    mean_pred = np.mean(y_pred, axis=(0,1))
    mean_true = np.mean(y_true, axis=(0,1))
    mean_xHI = np.mean(xHI, axis=(0,1))
    if(PLOT_ERROR):
        np.savetxt('%sstats_i%d.txt' %(path_out, i_pred), np.array([redshift, acc, TPR, TNR, iou, acc, acc_err, mean_pred, mean_error, mean_true]).T, fmt='%.3f\t'+('%.3e\t'*9)[:-1], header='eff_fact=%.4f\tRmfp=%.4f\tTvir=%.4e\t\nz\tacc\t\tprec\t\tspec\t\tiou\t\tacc\t\terr_acc\t\tx_pred\t\terr_x_pred\t\ty_true' %(zeta, Rmfp, Tvir))
    else:
        np.savetxt('%sstats_i%d.txt' %(path_out, i_pred), np.array([redshift, acc, TPR, TNR, iou, acc, mean_pred, mean_true]).T, fmt='%.3f\t'+('%.3e\t'*7)[:-1], header='eff_fact=%.4f\tRmfp=%.4f\tTvir=%.4e\t\nz\tacc\t\tprec\t\tspec\t\tiou\t\tacc\t\tx_pred\t\ty_true' %(zeta, Rmfp, Tvir))

    xHI_plot = np.arange(0.1, 1., 0.1)
    redshift_plot = np.array([redshift[np.argmin(abs(mean_true - meanHI))] for meanHI in xHI_plot])

    if(PLOT_STATS):
        # PLOT MATTHEWS CORRELATION COEF
        fig = plt.figure(figsize=(10, 8))
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['font.size'] = 16
        plt.plot(redshift, acc, '-', label=r'$R^2$ score')
        if(PLOT_ERROR):
            acc_error_low = np.clip(acc-acc_err, 0, (acc-acc_err).max())
            acc_error_up = np.clip(acc_err+acc, (acc_err+acc).min(), 1)
            plt.fill_between(redshift, acc_error_low, acc_error_up, color='lightblue', alpha=0.8)
        #plt.vlines(redshift_plot, ymin=0, ymax=1, color='black', ls='--')
        plt.xlabel('z'), plt.ylabel(r'$r_{\phi}$')
        plt.ylim(0, 1)
        plt.legend()
        #for iplot in range(redshift_plot.size):
        #    plt.text(redshift_plot[iplot]+0.03, 0.95, round(xHI_plot[iplot],1), rotation=90)
        plt.savefig('%sr2_i%d.png' %(path_out, i_pred), bbox_inches='tight'), plt.clf()
        plt.clf()

    if(PLOT_MEAN):
        # PLOTS AVERGE MASK HI
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.8])

        # Main plot
        ax0 = plt.subplot(gs[0])
        ax0.plot(redshift, mean_pred, ls='-', color='tab:orange', label='Prediction', lw=1.5)
        ax0.plot(redshift, mean_true, ls='-', color='tab:blue', label='True', lw=1.5)
        if(PLOT_ERROR):
            mean_error_low = np.clip(mean_pred-mean_error, 0, (mean_pred-mean_error).max())
            mean_error_up = np.clip(mean_error+mean_pred, (mean_error+mean_pred).min(), 1)
            ax0.fill_between(redshift, mean_error_low, mean_error_up, color='lightcoral', alpha=0.2)
        ax0.set_ylim(-0.01, 1.01)
        ax0.legend(loc=4)
        ax0.set_ylabel(r'$x_{HI}$')

        # plot relative difference
        ax1 = plt.subplot(gs[1], sharex = ax0)
        perc_diff = 100*(1-(mean_true + 1)/(mean_pred + 1))   # here is basically rescaling from 1 to 2 to avoid explosion of the numertor
        ax1.plot(redshift, perc_diff, 'k-', lw=1.5)
        ax1.set_ylabel('difference (%)')
        ax1.set_xlabel('$z$')
        #ax1.fill_between(z_quad, diff_s_avrgR_under, diff_s_avrgR_over, color='lightgreen', alpha=0.1)
        ax1.axhline(y=0,  color='black', ls='dashed')
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        plt.savefig('%smean_i%d.png' %(path_out, i_pred), bbox_inches='tight'), plt.clf()
        plt.clf()

    if(PLOT_VISUAL):
        # Visual Plot
        i_slice = np.argmin(abs(mean_xHI - 0.5))
        i_lc = params['HII_DIM']//2

        plt.rcParams['font.size'] = 20
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['axes.linewidth'] = 1.2

        fig = plt.figure(figsize=(35, 15))
        gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[3, 1], height_ratios=[1,1])

        # FIRST LC PLOT
        ax0 = fig.add_subplot(gs[0,0])
        im = ax0.pcolormesh(redshift, dr_xy, y_true[:,i_lc,:], cmap='jet')
        #ax0.contour(redshift, dr_xy, y_true[:,i_lc,:])
        ax0.set_ylabel('y [Mpc]', size=20)
        ax0.set_xlabel('z', size=20)

        # FIRST SLICE PLOT
        ax01 = fig.add_subplot(gs[0,1])
        ax01.set_title(r'$z$ = %.3f   $x_{HI}=%.2f$' %(redshift[i_slice], mean_xHI[i_slice]), fontsize=20)
        ax01.pcolormesh(dr_xy, dr_xy, y_true[...,i_slice], cmap='jet')
        #ax01.contour(dr_xy, dr_xy, y_true[...,i_slice])
        fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax01, pad=0.01, fraction=0.048)

        # SECOND LC PLOT
        ax1 = fig.add_subplot(gs[1,0])
        im = ax1.pcolormesh(redshift, dr_xy, y_pred[:,i_lc,:], cmap='jet', vmin=y_pred.min(), vmax=y_pred.max())
        #ax1.contour(redshift, dr_xy, y_true[:,i_lc,:])
        ax1.set_ylabel('y [Mpc]', size=20)
        ax1.set_xlabel('z', size=20)

        # SECOND SLICE PLOT
        ax11 = fig.add_subplot(gs[1,1])
        ax11.set_title(r'$r_{\phi}$ = %.3f' %(acc[i_slice]), fontsize=20)
        ax11.pcolormesh(dr_xy, dr_xy, y_pred[...,i_slice], cmap='jet', vmin=y_pred.min(), vmax=y_pred.max())
        fig.colorbar(im, label=r'$\delta T_b$ [mK]', ax=ax11, pad=0.01, fraction=0.048)
            #ax11.contour(dr_xy, dr_xy, y_true[...,i_slice])

        for ax in [ax01, ax11]:
            ax.set_ylabel('y [Mpc]', size=20)
            ax.set_xlabel('x [Mpc]', size=20)

        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        plt.savefig('%svisual_i%d.png' %(path_out, i_pred), bbox_inches='tight')
        plt.clf()

    if(PLOT_ERROR):
        # Visual Plot
        i_slice = np.argmin(abs(mean_true - 0.5))
        i_lc = params['HII_DIM']//2

        plt.rcParams['font.size'] = 20
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['xtick.top'] = True
        plt.rcParams['ytick.right'] = True
        plt.rcParams['axes.linewidth'] = 1.2

        fig = plt.figure(figsize=(35, 15))
        gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[3,1], height_ratios=[1,1])

        # FIRST LC PLOT
        ax0 = fig.add_subplot(gs[0,0])
        #ax0.set_title('$r_{\phi}=%.3f$ $t_{obs}=%d\,h$' %(acc[i_slice], 1000), fontsize=20)
        ax0.pcolormesh(redshift, dr_xy, y_pred[:,i_lc,:], cmap='jet')
        ax0.contour(redshift, dr_xy, y_true[:,i_lc,:])
        ax0.set_ylabel('y [Mpc]', size=20)
        ax0.set_xlabel('z', size=20)

        # FIRST SLICE PLOT
        ax01 = fig.add_subplot(gs[0,1])
        ax01.set_title(r'$z$ = %.3f   $x_{HI}=%.2f$' %(redshift[i_slice], mean_true[i_slice]), fontsize=20)
        im = ax01.pcolormesh(dr_xy, dr_xy, y_pred[...,i_slice], cmap='jet')
        ax01.contour(dr_xy, dr_xy, y_true[...,i_slice])

        # SECOND LC PLOT
        ax1 = fig.add_subplot(gs[1,0])
        im = ax1.pcolormesh(redshift, dr_xy, y_error[:,i_lc,:]  , cmap='jet', vmin=y_error.min(), vmax=y_error.max())
        ax1.contour(redshift, dr_xy, y_true[:,i_lc,:])
        fig.colorbar(im, label=r'$\sigma_{std}$', ax=ax1, pad=0.01, fraction=0.048)
        ax1.set_ylabel('y [Mpc]', size=20)
        ax1.set_xlabel('z', size=20)

        # SECOND SLICE PLOT
        ax11 = fig.add_subplot(gs[1,1])
        ax11.set_title(r'$r_{\phi}$ = %.3f' %(acc[i_slice]), fontsize=20)
        im = ax11.pcolormesh(dr_xy, dr_xy, y_error[...,i_slice], cmap='jet', vmin=y_error[...,i_slice].min(), vmax=y_error[...,i_slice].max())
        ax11.contour(dr_xy, dr_xy, y_true[...,i_slice])
        fig.colorbar(im, label=r'$\sigma_{std}$', ax=ax11, pad=0.01, fraction=0.048)

        for ax in [ax01, ax11]:
            ax.set_ylabel('y [Mpc]', size=20)
            ax.set_xlabel('x [Mpc]', size=20)

        plt.subplots_adjust(hspace=0.3, wspace=0.15)
        plt.savefig('%serror_i%d.png' %(path_out, i_pred), bbox_inches='tight')
        plt.clf()

    if(PLOT_SCORE):
        if(ii % 100 == 0):
            if(ii == 0):
                fig1, ax_s = plt.subplots(figsize=(10,8), ncols=1)

            # get redshift color
            cm = matplotlib.cm.plasma
            sc = ax_s.scatter(mean_xHI, acc, c=redshift, vmin=redshift.min(), vmax=redshift.max(), s=25, cmap=cm, marker='.')
            norm = matplotlib.colors.Normalize(vmin=7, vmax=9, clip=True)
            mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
            redshift_color = np.array([(mapper.to_rgba(v)) for v in redshift])

            #if(PLOT_ERROR):
            #    for x, y, e, clr, red in zip(mean_xHI, acc, acc_err, redshift_color, redshift):
            #        ax_s.errorbar(x=x, y=y, yerr=e, lw=1, marker='o', capsize=1, color=clr)

            ax_s.set_xlim(mean_xHI.min()-0.02, mean_xHI.max()+0.02), ax_s.set_xlabel(r'$\rm x^v_{HI}$', size=20)
            ax_s.set_ylim(-0.02, 1.02), ax_s.set_ylabel(r'$\rm r_{\phi}$', size=20)
            ax_s.set_yticks(np.arange(0, 1.1, 0.1))
            ax_s.set_xticks(np.arange(0, 1.1, 0.2))
            ax_s.hlines(y=np.mean(acc), xmin=-0.02, xmax=1.1, ls='--', label=r'$r_{\phi}$ = %.3f' %(np.mean(acc)), alpha=0.8, color='tab:blue', zorder=3)
            plt.legend(loc=1)
            if(ii == pred_idx.size-1):
                fig1.colorbar(sc, ax=ax_s, pad=0.01, label=r'$\rm z$')
                fig1.savefig('%sr2_dataset.png' %path_out, bbox_inches='tight')
                plt.clf()

print('... done.')

