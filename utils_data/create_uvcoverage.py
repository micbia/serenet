import numpy as np, os, pickle, time
import tools21cm as t2c
from tqdm import tqdm
from glob import glob

path_out = '/scratch/snx3000/mibianco/test_uv/'
#antconf = '/store/ska/sk09/segunet/SKA1_LowConfig_Sept2016_XYZ.txt'
antconf = None

#params = {'HII_DIM':128, 'DIM':384, 'BOX_LEN':256}
#params = {'HII_DIM':256, 'DIM':756, 'BOX_LEN':512}
params = {'HII_DIM':166, 'BOX_LEN':244./0.7}
#redshift = np.arange(7.001, 12.001, 0.001)
#redshift = np.arange(7.000, 30.005, 0.001)
#redshift = np.arange(30.006, 35.100, 0.001)
#redshift = np.arange(5., 7.000, 0.001)
redshift = np.loadtxt('/store/ska/sk02/lightcones/244Mpc_f2_8.2pS_500to166/redshift_244Mpc166_EoR.txt')

# multiprocessor variables
rank = int(os.environ['SLURM_ARRAY_TASK_ID'])
nprocs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
loop_start, loop_end = 0, redshift.size
perrank = (loop_end-loop_start)//nprocs

# terminate job control array
#endmessage_array = np.zeros(nprocs)

# loop index
resume_step = 0
i_start = int(loop_start+resume_step+rank*perrank)
if(rank != nprocs-1):
    i_end = int(loop_start+(rank+1)*perrank)
else:
    i_end = loop_end

for i in tqdm(range(i_start, i_end)):
    z = redshift[i]
    file_uv, file_Nant = '%s/uvmap_z%.3f.npy' %(path_out, z), '%s/Nantmap.npy' %(path_out)

    if not (os.path.exists(file_uv)):
        uv, Nant = t2c.get_uv_daily_observation(params['HII_DIM'], z, filename=antconf, total_int_time=6.0, int_time=10.0, boxsize=params['BOX_LEN'], declination=-30.0, verbose=False)
        np.save(file_uv, uv)

    if not (os.path.exists(file_Nant)):
        np.save(file_Nant, Nant)

# mark job finished
#endmessage_array[rank] = 1
jobstatus = False

while jobstatus:
    nr_files = len(glob('%s*.npy' %path_out)) - len(glob(file_Nant))

    if(rank == 0 and nr_files == redshift.size):
        uvs = {}
        for z in tqdm(redshift):
            uv_map = np.load('%suvmap_z%.3f.npy' %(path_out, z))
            uvs['{:.3f}'.format(z)] = uv_map

        uvs['Nant'] = int(np.load(file_Nant))
        pickle.dump(uvs, open('%suvmap_%d_z%d-%d.pkl' %(path_out, params['HII_DIM'], redshift.min(), redshift.max()), 'wb'))

        os.system('rm %s*.npy' %path_out)
        jobstatus = False
    else:
        time.sleep(300)

print("...done")