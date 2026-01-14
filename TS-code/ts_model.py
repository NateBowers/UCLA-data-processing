import numpy as np
import plasmapy
import matplotlib.pyplot as plt
import numpy as np
import lmfit
import matplotlib.pyplot as plt
from plasmapy.diagnostics import thomson
from astropy import units as u
from scipy.stats import norm
from scipy.optimize import minimize 
from PIL import Image
import re
import matplotlib.gridspec as gridspec
import matplotlib.widgets as mw
import matplotlib.ticker as tick

plt.rcParams.update({ "text.usetex": True, "font.family": "serif",})
plt.rcParams['text.latex.preamble'] = (r'\usepackage{amsmath} \usepackage{amssymb}')


# Physical quantities/system parameters
PROBE_VEC = np.array([1, 0, 0])
SCATTER_ANG = np.deg2rad(90)
SCATTER_VEC = np.array([np.cos(SCATTER_ANG), np.sin(SCATTER_ANG), 0])
PROBE_LAMBDA = 532 * u.nm
NOTCH = [531 , 533] * u.nm
INSTR_FWHM = 0.1775 # nm
EV_TO_K = 11604.518 # NIST 2022 CODATA
K_TO_EV = 1 / EV_TO_K
SYS_SIGMA = 1.9873 # calculated via ts_system_sigma.py


def load_data(paths):
    """Read .tiff files included in paths array."""
    if type(paths) == str:
        return np.array(Image.open(paths))
    else:
        data = np.array([Image.open(path) for path in paths])
        avg_image = np.average(data, axis=0)
        return avg_image

def px_to_nm(px, absolute=True):
    """Convert pixels to nanometers using Ne calibration values."""
    if absolute:
        return  0.0386 * px + 522.9215
    else:
        return 0.0386 * px

def nm_to_px(nm, absolute=True):
    """Convert nanometers to pixels using Ne calibration values."""
    if absolute: 
        return (nm - 522.9215) / 0.0386
    else:
        return nm / 0.0386

def instr_func(arr: np.array) -> np.array:
    """Calculates the instrument function over array. It is assumed that (a)
    the array is centered on zero and (b) it corresponds to units of 
    nanometers (NOT pixels)."""

    scale_nm = INSTR_FWHM / (2 * np.sqrt(2 * np.log(2)))
    spread = norm.pdf(arr, loc=0, scale=scale_nm)
    return spread

def spectral_density_wrapper(wavelengths, n, temp, scale, zero, return_alpha=False):
    """Wrapper for plasmapy's thomson.spectral_density to make it into the
    proper format for the fitting routine."""

    T_e_K = temp * EV_TO_K * u.K
    T_i_K = T_e_K 

    alpha, skw = thomson.spectral_density(
        wavelengths = wavelengths * u.nm,
        n = n * 1e6 * u.m ** -3,
        T_e = T_e_K,
        T_i = T_i_K,
        probe_wavelength=PROBE_LAMBDA,
        probe_vec=PROBE_VEC,
        notch=NOTCH,
        instr_func=instr_func,
    )
    skw /= np.max(skw)
    skw = (skw - zero) * scale

    if isinstance(wavelengths, u.Quantity):
        wavelengths = wavelengths.value

    l_notch_idx = np.argmin(np.abs(wavelengths - NOTCH[0].value))
    h_notch_idx = np.argmin(np.abs(wavelengths - NOTCH[1].value))
   
    skw[l_notch_idx:h_notch_idx] = np.nan
    
    if return_alpha:
        return skw, alpha
    else:
        return skw
    

def fit_model(
        spectrum: np.ndarray, 
        wavelengths: np.ndarray, 
        p_0: str = None, 
        params_to_vary: str = None,
        sigma: float | np.ndarray = None,
        **kwargs
    ) -> lmfit.model.ModelResult:
    """ For the passed spectrum, uses a least squares approach to find the
    parameters of best fit. The fitted parameters are:
        n: (electron) density.
        temp: (electron) temperature. It is assumed the plasma is in thermal
            equilibrium.
        scale: a normalization constant to ensure the spectrum ranges from
            zero to one.
        zero: a baseline constant to ensure the spectrum's minimum is zero.
    Of the parameters, scale and zero are nuisance parameters.

    Args:
        spectrum (np.ndarray): Experimental spectrum to fit.
        wavelengths (np.ndarray): Wavelengths for the spectrum. Must be same
            size as spectrum.
        p_0 (str, optional): Initial guesses for density, temperature, scale, 
            and zero, in that order. If not none, must have 4 elements. 
            Defaults to None.
        params_to_vary (str, optional): Which parameters to fit, all other 
            parameters are held constant when fitting. Defaults to None.
        sigma (float | np.ndarray, optional): Error in each value. Must be
            able to be cast to same size as spectrum.

    Returns:
        lmfit.model.ModelResult: Fitted model with relevant statistical information.
    """
    
    param_names = ['n', 'temp', 'scale', 'zero']
    if params_to_vary is None:
        params_to_vary = param_names
    if p_0 is None:
        p_0 = [1e14, 10, 10, 0]
    mins = [0, 0, 0, -np.inf]

    params = lmfit.Parameters()

    if sigma == None:
        sigma = np.ones(spectrum.shape)
    else:
        try:
            sigma = np.full(spectrum.shape, fill_value = sigma)
        except:
            raise ValueError('Sigma must be either a float or the' \
            'same size as spectrum')

    for param, val, min in zip(param_names, p_0, mins):
        if param in params_to_vary:
            params.add(param, value=val, vary=True, min=min)
        else:
            params.add(param, value=val, vary=False, min=min)

    model = lmfit.Model(
        spectral_density_wrapper, 
        independent_vars=['wavelengths'],
        nan_policy='omit',
    )

    # defaults to least squares!!!
    results = model.fit(
        data=spectrum,
        params=params,
        wavelengths=wavelengths,
        weights = 1/sigma,
        scale_covar=False,
    )

    return results



def main(
        fg_paths: list[str] | str, 
        bg_paths: list[str] | str
    ):
    """

    For calculating the errors, the model assumes 

    Args:
        fg_paths (list[str] | str): Path to each foreground image
        bg_paths (list[str] | str): Path to each background image
    """

    def clip_raw_for_visualize(arr: np.ndarray) -> np.ndarray:
        # Helper function for visualizing raw data
        mid = np.average(arr)
        std = np.std(arr)
        tmp = np.clip(arr, a_min = mid - 3 * std, a_max = mid + 3 * std)
        return tmp

    for paths in fg_paths, bg_paths:
        if type(paths) != list:
            paths = [paths]

    fg = load_data(fg_paths)
    bg = load_data(bg_paths)
    im = np.array(fg - bg)
    top = 380
    bot = 220
    wavelengths = px_to_nm(np.arange(512))

    # 1/n_ef = 1/n_fg + 1/n_bg 
    # (assuming everything is distributed as a Gaussian)
    n_ef = 1 / ((1/len(fg_paths)) + (1/len(bg_paths)))
    sig_ef = SYS_SIGMA / np.sqrt(n_ef)
    

    spectrum = np.average(im[bot:top], axis=0)    
    res = fit_model(
        spectrum, 
        wavelengths, 
        sigma = sig_ef, 
        params_to_vary=['n', 'temp', 'scale'] # Add 'zero' to adjust baseline
    )
    fit = res.best_fit
    vals = res.summary()['params']
    fit_n = vals[0][1]
    fit_T = vals[1][1]
   
    theta_hat = [
            vals[0][1],
            vals[1][1],
            vals[2][1],
            vals[3][1],
        ]


    print(res.fit_report())
    dely = np.sqrt(res.eval_uncertainty()**2 + sig_ef**2)

    

    print('Fit results')
    print(f'T={fit_n:.4f} eV, n={fit_T:.3e} cc')

    fig = plt.figure(figsize=(6,5), dpi=200)
    ax2, ax = fig.subplots(2, sharex=True, height_ratios=(1.5, 5))

    ext = [wavelengths[0], wavelengths[-1], top, bot]
    ax2.imshow(clip_raw_for_visualize(im)[bot:top], aspect='auto', extent=ext)
    ax2.set_title('Raw spectrum')
    ax2.set_yticklabels('')

    ax.grid(zorder=-10, c='k', linewidth=0.5)
    arr_tmp = np.linspace(-10, 10, 21)
    mask = norm.pdf(arr_tmp, loc=0, scale = (4.5992 / (2 * np.sqrt(2 * np.log(2)))) )
    spectrum_smooth = np.convolve(spectrum, mask, mode='same')
    l_notch_idx = np.argmin(np.abs(wavelengths - NOTCH[0].value))
    h_notch_idx = np.argmin(np.abs(wavelengths - NOTCH[1].value))
    spectrum_smooth[l_notch_idx:h_notch_idx] = np.nan

    n_exp = np.floor(np.log10(theta_hat[0]))
    s_n = f'$n = {theta_hat[0]/(10**n_exp):.3f}\\times 10^{{{int(n_exp)}}}$cm$^{{-3}}$'
    s_t = f'$T = {theta_hat[1]:.3f}$ eV'
        
    ax.plot(wavelengths, (spectrum_smooth / theta_hat[2]) + theta_hat[3], c='b', zorder=-7)
    ax.plot(wavelengths, (fit / theta_hat[2]) + theta_hat[3], c='r', label=f'{s_n}\n{s_t}')

    err_bar_top = ((fit + dely) / theta_hat[2]) + theta_hat[3]
    err_bar_bot = ((fit - dely) / theta_hat[2]) + theta_hat[3]

    err_bar_top = (fit + dely) / theta_hat[2] + theta_hat[3]
    err_bar_bot = (fit - dely) / theta_hat[2] + theta_hat[3]
    ax.fill_between(wavelengths, err_bar_bot, err_bar_top, color='r', alpha=0.3)

    ax.set_xticks([523, 526, 529, 532, 535, 538, 541])
    ax.set_xlabel(r"Wavelength [nm]")
    ax.set_ylabel('Normalized Intensity [A.U.]')
    ax.set_title('Fit Thomson Scattering Curve')
    ax.set_ylim(-0.2, 1.3)
    plt.legend()

    #plt.savefig('save_path')
    plt.show()





    
    # if verbose:
    #     fig = plt.figure(constrained_layout=True, figsize=(6, 4.5), dpi=200)
    #     fig1, fig2 = fig.subfigures(nrows=1, ncols=2, width_ratios=(1,2))

    #     ext = [wavelengths[0], wavelengths[-1], 511, 0]
    #     ax_diff = fig2.subplots()
    #     ax_diff.imshow(clip_raw_for_visualize(im), aspect='auto', extent=ext)
    #     ax_diff.set_title('Foreground - Background')
    #     ax_diff.axhline(bot, c='k', linestyle='--')
    #     ax_diff.axhline(top, c='k', linestyle='--')


    #     ax_fg, ax_bg = fig1.subplots(nrows=2)
    #     ax_fg.imshow(clip_raw_for_visualize(fg), aspect='auto', extent=ext)
    #     ax_bg.imshow(clip_raw_for_visualize(bg), aspect='auto', extent=ext)
    #     ax_fg.tick_params(axis='both', labelsize=8)
    #     ax_bg.tick_params(axis='both', labelsize=8)
    #     ax_fg.set_title('Avg Foreground')
    #     ax_bg.set_title('Avg Background')

    #     for ax in (ax_diff, ax_fg, ax_bg):
    #         ax.set_yticks([])
    #         ax.set_xlabel(r'$\lambda$ [nm]')
    #         ax.set_xticks([524, 528, 532, 536, 540])

    #     # plt.savefig('ts_data_CH_loc1_100V.png')
    #     plt.show()
    #     plt.close()


    # def nll_thomson(theta):
    #     wavelengths = px_to_nm(np.arange(512))
    #     n, temp, scale, zero = theta
    #     fit_tmp = spectral_density_wrapper(wavelengths, n, temp, scale, zero)
    #     l_notch_idx = np.argmin(np.abs(wavelengths - NOTCH[0].value))
    #     h_notch_idx = np.argmin(np.abs(wavelengths - NOTCH[1].value))
    #     spectrum_tmp = np.copy(spectrum)
    #     spectrum_tmp[l_notch_idx:h_notch_idx] = np.nan
    #     wavelength_tmp = np.copy(wavelengths)
    #     wavelength_tmp[l_notch_idx:h_notch_idx] = np.nan

    #     fit_tmp = fit_tmp[~np.isnan(fit_tmp)]
    #     spectrum_tmp = spectrum_tmp[~np.isnan(spectrum_tmp)]
    #     wavelength_tmp = wavelength_tmp[~np.isnan(wavelength_tmp)]

    #     q = (len(wavelength_tmp) * np.log(sig_ef * np.sqrt(2 * np.pi)) 
    #          + np.sum(np.square(spectrum_tmp - fit_tmp)) / (2 * sig_ef**2)
    #          )

    #     return q



    

    # if errors:
    #     print()
    #     print('Running fit statistics')
        
    #     q_0 = nll_thomson(theta_hat)

    #     n_ci_68 = []
    #     n_ci_95 = []
    #     t_ci_68 = []
    #     t_ci_95 = []

    #     for i in (-1, 1):
    #         for Q, ci in zip([1, 4], (t_ci_68, t_ci_95)):
    #             t_val = theta_hat[1] + i * Q * vals[1][7]
    #             def profile_wrt_temp(t_val):
    #                 if type(t_val) == np.ndarray:
    #                     t_val = t_val[0]
    #                 theta0 = np.copy(theta_hat)
    #                 theta0[1] = t_val
    #                 res_tmp = fit_model(spectrum, wavelengths, params_to_vary = ['n', 'scale', 'zero'], p_0=theta0)
    #                 theta_eta = res_tmp.best_values.values()
    #                 q = nll_thomson(theta_eta)
    #                 zeta = 2 * (q - q_0)
    #                 return np.abs(zeta - Q)
    #             out = minimize(profile_wrt_temp, x0=t_val)
    #             ci.append(out.x[0])
    #     print('Confidence Intervals for Temperature')
    #     print(f'   68% -> {list(t_ci_68)}')
    #     print(f'   95% -> {list(t_ci_95)}')

    #     for i in (-1, 1):
    #         for Q, ci in zip([1, 4], (n_ci_68, n_ci_95)):
    #             n_val = theta_hat[0] + i * Q * vals[0][7]
    #             def profile_wrt_temp(n_val: float):
    #                 if type(n_val) == np.ndarray:
    #                     n_val = n_val[0]
    #                 theta0 = np.copy(theta_hat)
    #                 theta0[1] = n_val
    #                 res_tmp = fit_model(spectrum, wavelengths, params_to_vary = ['temp', 'scale', 'zero'], p_0=theta0)
    #                 theta_eta = res_tmp.best_values.values()
    #                 q = nll_thomson(theta_eta)
    #                 zeta = 2 * (q - q_0)
    #                 return np.abs(zeta - Q)
    #             out = minimize(profile_wrt_temp, x0=n_val)
    #             ci.append(out.x[0])

    #     print('Confidence Intervals for Density')
    #     print(f'   68% -> {list(n_ci_68)}')
    #     print(f'   95% -> {list(n_ci_95)}')
    #     n_exp = np.floor(np.log10(theta_hat[0]))
    #     s_n = f'$n = {theta_hat[0]/(10**n_exp):.3f}^{{+{(n_ci_68[1]-theta_hat[0])/(10**n_exp):.3f}}}_{{{(n_ci_68[0]-theta_hat[0])/(10**n_exp):.3f}}}$ $\\times 10^{{{int(n_exp)}}}$cm$^{{-3}}$'
    #     s_t = f'$T = {theta_hat[1]:.3f}^{{+{t_ci_68[1]-theta_hat[1]:.3f}}}_{{{t_ci_68[0]-theta_hat[1]:.3f}}}$ eV'

    #     err_bar_top = fit
    #     err_bar_bot = fit
    #     n_range = np.linspace(n_ci_68[0], n_ci_68[1], 5)
    #     t_range = np.linspace(t_ci_68[0], t_ci_68[1], 5)
    #     for n in n_range:
    #         for t in t_range:
    #             tmp = spectral_density_wrapper(wavelengths, n, t, scale=theta_hat[2], zero=theta_hat[3]).value
    #             err_bar_top = np.maximum(err_bar_top, tmp)
    #             err_bar_bot = np.minimum(err_bar_bot, tmp)

    
    # else:
    #     n_exp = np.floor(np.log10(theta_hat[0]))
    #     s_n = f'$n = {theta_hat[0]/(10**n_exp):.3f}\\times 10^{{{int(n_exp)}}}$cm$^{{-3}}$'
    #     s_t = f'$T = {theta_hat[1]:.3f}$ eV'

    # # to calc error bands, generate several spectra and find the min, max of all of them


        


    # fig = plt.figure(figsize=(6,5), dpi=200)
    # ax2, ax = fig.subplots(2, sharex=True, height_ratios=(1.5, 5))

    # ext = [wavelengths[0], wavelengths[-1], top, bot]
    # ax2.imshow(clip_raw_for_visualize(im)[bot:top], aspect='auto', extent=ext)
    # ax2.set_title('Raw spectrum')
    # ax2.set_yticklabels('')
    # # ax2.axhline(bot, c='k', linestyle='--')
    # # ax2.axhline(top, c='k', linestyle='--')

    # # ax = fig.subplots()
    # ax.grid(zorder=-10, c='k', linewidth=0.5)
    # arr_tmp = np.linspace(-10, 10, 21)
    # mask = norm.pdf(arr_tmp, loc=0, scale = (4.5992 / (2 * np.sqrt(2 * np.log(2)))) )

    # spectrum_smooth = np.convolve(spectrum, mask, mode='same')
    # l_notch_idx = np.argmin(np.abs(wavelengths - NOTCH[0].value))
    # h_notch_idx = np.argmin(np.abs(wavelengths - NOTCH[1].value))
    # spectrum_smooth[l_notch_idx:h_notch_idx] = np.nan
        
    # # ax.plot(wavelengths, spectrum / norm_factor, linewidth=0.7, c='b', zorder=-8, alpha=0.5, label='Raw Spectrum')
    # ax.plot(wavelengths, (spectrum_smooth / theta_hat[2]) + theta_hat[3], c='b', zorder=-7)
    # ax.plot(wavelengths, (fit / theta_hat[2]) + theta_hat[3], c='r', label=f'{s_n}\n{s_t}')

    # err_bar_top = ((fit + dely) / theta_hat[2]) + theta_hat[3]
    # err_bar_bot = ((fit - dely) / theta_hat[2]) + theta_hat[3]

    # err_bar_top = (fit + dely) / theta_hat[2] + theta_hat[3]
    # err_bar_bot = (fit - dely) / theta_hat[2] + theta_hat[3]
    # ax.fill_between(wavelengths, err_bar_bot, err_bar_top, color='r', alpha=0.3)

    # # if errors:
    #     # ax.fill_between(wavelengths, (err_bar_bot / theta_hat[2]) + theta_hat[2], (err_bar_top / theta_hat[2]) + theta_hat[2], color='r', alpha=0.3)

    # ax.set_xticks([523, 526, 529, 532, 535, 538, 541])
    # ax.set_xlabel(r"Wavelength [nm]")
    # ax.set_ylabel('Normalized Intensity [A.U.]')
    # ax.set_title('Fit Thomson Scattering Curve')
    # # ax.set_ylim(-0.2, 1.3)
    # plt.legend()

    # # plt.savefig('ts_fit_loc1_100V.png')
    # plt.show()
    # plt.close()



    # print()
    # print('Generating confidence regions')

    # if errors:
    #     n_range = np.linspace(vals[0][1] - 3*vals[0][7] / np.sqrt(n_ef), vals[0][1] + 3*vals[0][7] / np.sqrt(n_ef))
    #     t_range = np.linspace(vals[1][1] - 3*vals[1][7] / np.sqrt(n_ef), vals[1][1] + 3*vals[1][7] / np.sqrt(n_ef))
    #     nll_img = np.empty((50, 50))
    #     for i, tval in enumerate(t_range):
    #         for j, nval in enumerate(n_range):
    #             nll_img[i, j] = 2 * (nll_thomson([nval, tval, theta_hat[2], theta_hat[3]]) - q_0)
        

    #     plt.figure(figsize=(4, 4), dpi=200)
    #     # foo = plt.imshow(nll_img, origin='lower', aspect='auto',
    #             #    extent=(n_range[0], n_range[-1], t_range[0], t_range[-1]))
    #     X, Y = np.meshgrid(n_range, t_range)
    #     CS = plt.contour(X, Y, nll_img, levels=[2.30,5.99], colors='k', linestyles=['-','--'], linewidths=0.8)
    #     plt.clabel(CS, CS.levels, fontsize=10, fmt={2.3: f'68\\%', 5.99: f'95\\%'})
    #     plt.scatter(theta_hat[0], theta_hat[1], c='r', marker='x', label=f'$n={theta_hat[0]:.3e}$ cm$^{{-3}}$\n$T={theta_hat[1]:.4f}$ eV')
    #     plt.xlabel(r'n [cm$^{-3}$]')
    #     plt.ylabel('T [eV]')
    #     plt.title('Confidence Regions for T and n')
    #     plt.legend()
    #     # plt.savefig('ts_ci_region_Loc1_100V.png')
    #     plt.show()
    #     plt.close()

    

if __name__ == '__main__':
    test_fg_path = ['ts_tiff_images_06-04/foreground/CH/yhyt_CH_Loc1_25V_#35.tiff']
    test_bg_path = ['ts_tiff_images/yes_heater_no_ts/avg_CH.tiff']


    test_fg_arr = [
        'ts_tiff_images_06-04/foreground/CH/yhyt_CH_Loc1_25V_#35.tiff',
        'ts_tiff_images_06-04/foreground/CH/yhyt_CH_Loc1_25V_#36.tiff',
        'ts_tiff_images_06-04/foreground/CH/yhyt_CH_Loc1_25V_#38.tiff',
        'ts_tiff_images_06-04/foreground/CH/yhyt_CH_Loc1_25V_#13.tiff',
    ]

    fg_data = [
        'ts_tiff_img_CH/fg/CH_fg_loc1_400V_#44.tiff',
        'ts_tiff_img_CH/fg/CH_fg_loc1_400V_#40.tiff',
        'ts_tiff_img_CH/fg/CH_fg_loc1_400V_#39.tiff',
    ]

    bg_data = [
        'ts_tiff_img_CH/bg/CH_bg_loc1_0V_#44.tiff',
    ]

    # 143.90ns
    fg_data_l1_0V = [
        'ts_tiff_img_CH/fg/CH_fg_loc1_100V_#39.tiff', # 133.5
        'ts_tiff_img_CH/fg/CH_fg_loc1_100V_#40.tiff', # 129.1
    ]

    bg_data_l1_0V = [
        # 160 ns
        'ts_tiff_img_CH/bg/CH_bg_loc1_100V_#37.tiff',
        'ts_tiff_img_CH/bg/CH_bg_loc1_100V_#38.tiff',
        'ts_tiff_img_CH/bg/CH_bg_loc1_100V_#39.tiff',
        'ts_tiff_img_CH/bg/CH_bg_loc1_100V_#40.tiff',
        'ts_tiff_img_CH/bg/CH_bg_loc1_100V_#4.tiff',
    ]


    main(fg_data_l1_0V, bg_data_l1_0V)