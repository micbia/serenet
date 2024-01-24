import numpy as np, sys, matplotlib.pyplot as plt
import tools21cm as t2c

from ps_eor import datacube, fgfit, fitutil, pspec, psutil

# Reading the data
path_out = '/scratch/snx3000/mibianco/output_sdc3/dataLC_256_train_090523/test/'
path_input = '/store/ska/sk014/mibianco/dataLC_128_pred_190922/'

astro_par = np.loadtxt(path_input+'parameters/astro_params.txt')

with open(path_input+'parameters/user_params.txt', 'r') as file:
    user_par = eval(file.read())
    box_len = user_par['BOX_LEN']
    n_pix = user_par['HII_DIM']
    dx = np.linspace(0, box_len, n_pix)

with open(path_input+'parameters/cosm_params.txt', 'r') as file:
    cosm_par = eval(file.read())
    t2c.set_hubble_h(1)
    t2c.set_omega_lambda(1-cosm_par['OMm'])
    t2c.set_omega_matter(cosm_par['OMm'])
    t2c.set_omega_baryon(0.045)
    t2c.set_sigma_8(0.82)
    t2c.set_ns(0.97)
    psutil.set_cosmology(LambdaCDM(H0=100, Om0=cosm_par['OMm'], Ode0=1-cosm_par['OMm'], Ob0=0.045))


redshift = np.loadtxt(path_input+'lc_redshifts.txt')

for i in range(300):
    data_dT2 = t2c.read_cbin('%sdata/dT2_21cm_i%d.bin' %(path_input, i))
    data_dT4 = t2c.read_cbin('%sdata/dT4_21cm_i%d.bin' %(path_input, i))
    data_xH = t2c.read_cbin('%sdata/xH_21cm_i%d.bin' %(path_input, i))

    # Get a subvolume
    z, depth_mhz = 9., 25
    dT2_sub, _, _ = t2c.get_lightcone_subvolume(lightcone=data_dT2, redshifts=redshift, central_z=z, depth_mhz=depth_mhz, odd_num_cells=False, subtract_mean=False, fov_Mpc=box_len, verbose=False)
    dT4_sub, box_dims, cut_redshift = t2c.get_lightcone_subvolume(lightcone=data_dT4, redshifts=redshift, central_z=z, depth_mhz=depth_mhz, odd_num_cells=False, subtract_mean=False, fov_Mpc=box_len)

    i_min, i_max = np.argmin(np.abs(redshift - cut_redshift.min())), np.argmin(np.abs(redshift - cut_redshift.max()))
    xH_sub = data_xH[..., i_min:i_max+1]

    # Modify the data for ps_eor
    redshift_ps = cut_redshift[::-1]
    freqs_ps = t2c.z_to_nu(redshift_ps)
    dT2_ps = np.moveaxis(dT2_sub, -1, 0)[::-1]
    dT4_ps = np.moveaxis(dT4_sub, -1, 0)[::-1]
    xH_ps = np.moveaxis(xH_sub, -1, 0)[::-1]
    box_dims_ps = box_dims[::-1]

    min_freqs, max_freqs = freqs_ps.min(), freqs_ps.max()

    FoV = t2c.angular_size_comoving(cMpc=box_len, z=redshift_ps).mean() # in [deg]
    res = np.radians(FoV) / n_pix # in [rad]
    df = abs(np.diff(freqs_ps).mean()) * 1e6 # in [Hz]

    meta = datacube.ImageMetaData.from_res(res, (n_pix, n_pix))
    meta.wcs.wcs.cdelt[2] = df

    # image cube in K, freqs in Hz
    image_cube2 = datacube.CartImageCube(dT2_ps * 1e-3, freqs_ps * 1e6, meta)
    image_cube4 = datacube.CartImageCube(dT4_ps * 1e-3, freqs_ps * 1e6, meta)

    # image to visibility, keeping modes between ... to ... wavelengths
    vis_cube4 = image_cube4.ft(1, 1e6)
    vis_cube2 = image_cube2.ft(1, 1e6)

    # Fake noise cube which is required for the FG fitting
    noise_cube = vis_cube2.new_with_data(np.random.randn(*vis_cube4.data.shape) * 1e-6)

    # --- Polynomial Fitting ---
    poly_fitter = fgfit.PolyForegroundFit(4, 'power_poly')
    poly_fit = poly_fitter.run(vis_cube4, noise_cube)
    img_cube_poly = poly_fit.sub.image()

    dT4poly = np.moveaxis(img_cube_poly.data[::-1], 0, 2).astype(np.float32) * 1e3
    if not os.path.exists('%sdT4poly_21cm_i%d_z%.3f_%dMHz.bin' %(path_out, i, z, depth_mhz)):
        t2c.save_cbin('%sdT4poly_21cm_i%d_z%.3f_%dMHz.bin' %(path_out, i, z, depth_mhz), dT4poly)
    else:
        print('%sdT4poly_21cm_i%d_z%.3f_%dMHz.bin already exist' %(path_out, i, z, depth_mhz))

    # --- Gaussain Processed Reduction ---
    vis_cube_s = vis_cube4.get_slice(min_freqs*1e6, max_freqs*1e6)
    noise_cube_s = noise_cube.get_slice(min_freqs*1e6, max_freqs*1e6)

    gpr_config = fitutil.GprConfig()

    gpr_config.fg_kern = fitutil.GPy.kern.RBF(1, name='fg')
    gpr_config.fg_kern.lengthscale.constrain_bounded(20, 60)

    gpr_config.eor_kern = fitutil.GPy.kern.Matern32(1, name='eor')
    gpr_config.eor_kern.lengthscale.constrain_bounded(0.2, 1.2)

    gpr_fitter = fgfit.GprForegroundFit(gpr_config)
    gpr_res_s = gpr_fitter.run(vis_cube_s, noise_cube_s)

    img_cube_gpr = gpr_res_s.sub.image()

    dT4gpr = np.moveaxis(img_cube_gpr.data[::-1], 0, 2).astype(np.float32) * 1e3
    if not os.path.exists('%sdT4gpr_21cm_z%.3f_%dMHz.bin' %(path_out, z, depth_mhz)):
        t2c.save_cbin('%sdT4gpr_21cm_z%.3f_%dMHz.bin' %(path_out, z, depth_mhz), dT4gpr)
    else:
        print('%sdT4gpr_21cm_z%.3f_%dMHz.bin already exist' %(path_out, z, depth_mhz))