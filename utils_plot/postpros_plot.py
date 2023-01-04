import numpy as np, matplotlib.pyplot as plt, os
import matplotlib.ticker as ticker
from glob import glob

import sys
sys.path.insert(0,'../')
from config.net_config import NetworkConfig

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = True

script, path = sys.argv
os.chdir(path)

config_file = glob('*.ini')[0]
conf = NetworkConfig(config_file)
metrics_wanted = conf.METRICS
metrics_wanted.remove('precision')
metrics_wanted.remove('balanced_accuracy')

# Load Data
epoch = conf.RESUME_EPOCH
name_val_metric = np.array(['outputs/val_%s_ep-%d.txt' %(nm, epoch) for nm in metrics_wanted])
name_metric = np.array(['outputs/%s_ep-%d.txt' %(nm, epoch) for nm in metrics_wanted])
loss, val_loss = np.loadtxt('outputs/loss_ep-%d.txt' %epoch), np.loadtxt('outputs/val_loss_ep-%d.txt' %epoch)
lr = np.loadtxt('outputs/lr_ep-%d.txt' %epoch)

idx_best_mode = conf.BEST_EPOCH-1

print(' best loss (epoch = %d / %d):\t%.3e' %(idx_best_mode+1, val_loss.size, val_loss[idx_best_mode]))

# Plot
fig1 = plt.figure(figsize=(16, 6)) 
fig1.subplots_adjust(hspace=0.3, wspace=0.25, top=0.99, bottom=0.08, right=0.955, left=0.045) 

ax1 = plt.subplot(1,2,1)
ax1.set_ylabel('Loss functions'), ax1.set_xlabel('Epoch') 
ax1.semilogy(val_loss, color='cornflowerblue', label='Validation Loss', ls='--')  
ax1.semilogy(loss, color='navy', label='Training Loss') 
ax1.scatter(idx_best_mode, val_loss[idx_best_mode], marker="x", color="r", label="Best Model: %.3e" %(np.min(val_loss[idx_best_mode])))
plot_min, plot_max = np.min([loss.min(), val_loss.min()])*0.9, np.min([loss.max(), val_loss.max()])
ax1.set_xlim(-1, loss.size), ax1.set_ylim(plot_min, plot_max)

ax3 = ax1.twinx() 
ax3.semilogy(lr, color='k', alpha=0.4, label='Learning Rate') 
ax3.set_ylabel('Learning Rate') 
lns, labs   = ax1.get_legend_handles_labels() 
lns2, labs2 = ax3.get_legend_handles_labels() 
ax1.legend(lns+lns2, labs+labs2, loc=1) 

colours = ['blue', 'orange', 'green', 'red']
i_cl = 0

ax2 = plt.subplot(1,2,2) 
ax2.set_ylabel('Accuracy'), ax2.set_xlabel('Epoch')
for i_nm, (nm, vnm) in enumerate(zip(name_metric, name_val_metric)): 
    if not('loss' in nm):
        metric, val_metric = np.loadtxt(nm), np.loadtxt(vnm)
        print(' %s :\t%.2f%%' %(metrics_wanted[i_nm], 100*val_metric[idx_best_mode]))
        ax2.plot(val_metric, color='tab:'+colours[i_cl], ls='--')
        ax2.plot(metric, color='dark'+colours[i_cl], label=metrics_wanted[i_nm])
        ax2.scatter(idx_best_mode, val_metric[idx_best_mode], marker="x", color="r")
        i_cl += 1

ax2.set_xlim(-1,loss.size)#, ax2.set_ylim(0.6, 0.9)
ax2.set_ylim(0., 1.0)
ax4 = ax2.twinx() 
ax4.semilogy(lr, color='k', alpha=0.4, label='Learning Rate') 
ax4.set_ylabel('Learning Rate') 
#lns2, labs2 = ax4.get_legend_handles_labels() 
#lns, labs = ax2.get_legend_handles_labels()
#ax2.legend(lns+lns2, labs+labs2, loc='best') 
ax2.legend(loc='best')

ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(10))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

ax1.tick_params(axis='both', length=7, width=1.1)
ax1.tick_params(which='minor', axis='both', length=3, width=1.1)
ax2.tick_params(axis='both', length=7, width=1.1)
ax2.tick_params(which='minor', axis='both', length=3, width=1.1)
ax3.tick_params(axis='both', length=7, width=1.1)
ax3.tick_params(which='minor', axis='both', length=3, width=1.1)
"""
ax4.tick_params(axis='both', length=7, width=1.1)
ax4.tick_params(which='minor', axis='both', length=3, width=1.1)
"""

plt.savefig('outputs/loss.png', bbox_inches='tight')
#plt.show()
