import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import lmfit
import matplotlib.pyplot as plt
from plasmapy.diagnostics import thomson
from astropy import units as u
from scipy.stats import norm
from PIL import Image

plt.rcParams.update({ "text.usetex": True, "font.family": "serif",})
plt.rcParams['text.latex.preamble'] = (r'\usepackage{amsmath}')
plt.rcParams['text.latex.preamble'] = (r'\usepackage{amssymb}')


# Physical quantities/system parameters
PROBE_VEC = np.array([1, 0, 0])
SCATTER_ANG = np.deg2rad(90)
SCATTER_VEC = np.array([np.cos(SCATTER_ANG), np.sin(SCATTER_ANG), 0])
PROBE_LAMBDA = 532 * u.nm
NOTCH = [531 , 533] * u.nm
INSTR_FWHM = 0.1775 # nm
EV_TO_K = 11604.518 # via NIST 2022 CODATA
K_TO_EV = 1 / EV_TO_K
SYS_SIGMA = 1.9873 # calculated w/ ts_system_sigma.py


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

def instr_func(arr: np.ndarray) -> np.ndarray:
    """Calculates the instrument function over array. It is assumed that (a)
    the array is centered on zero and (b) it corresponds to units of 
    nanometers (NOT pixels)."""

    scale_nm = INSTR_FWHM / (2 * np.sqrt(2 * np.log(2)))
    spread = norm.pdf(arr, loc=0, scale=scale_nm)
    return spread

def notch(arr: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Replace elements in arr corresponding to notched wavelengths with
    np.nan"""
    l_notch_idx = np.argmin(np.abs(wavelengths - NOTCH[0].value))
    h_notch_idx = np.argmin(np.abs(wavelengths - NOTCH[1].value))
    arr[l_notch_idx:h_notch_idx] = np.nan
    return arr

def spectral_density_wrapper(wavelengths, n, temp, scale, zero):
    """Wrapper for plasmapy's thomson.spectral_density to make it into the
    proper format for the fitting routine."""

    T_e_K = temp * EV_TO_K * u.K
    T_i_K = T_e_K 

    _, skw = thomson.spectral_density(
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

    skw = notch(skw, wavelengths)
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
        lmfit.model.ModelResult: Fitted model with relevant statistical 
        information. See https://lmfit.github.io/lmfit-py/model.html
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
    """Main function for thomson spectrum analysis. Takes foreground
    and background .tiff files and finds the best density and temperature
    with a least squares method. Assumes the plasma is in thermal equilibrium
    i.e., T_e = T_i = T. PlasmaPy is used to generate the spectral density
    function. For details, see
    https://docs.plasmapy.org/en/stable/ad/diagnostics/thomson.html.

    Notes on the program:

    To generate the raw spectrum:
    1) The foreground and background images are averaged, subtracted
    from each other, and clipped vertically from row 220 to 380.
    2) Each row is then averaged to make a spectrum.

    
    When fitting the model:
    1) Three variables are fitted: density, temperature, and scale where
    scale is a normalization constant.
    2) The baseline value (zero) can also be fit if the baseline
    looks off. 
    3) The objective function that is minimized is `obj(n,T,scale,zero) = 
    scale(Skw(n,T) - zero) - data` where Skw(n,T) is the spectral density
    function given by plasmapy
    4) lmfit is used for the fitting, by default uses a least squares
    method. Other methods can be used, for further information see 
    https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.fit

    
    For calculating the errors, the model uses the following pipeline:
    1) Assumes per pixel noise is the same across the detector (sigma_sys)
    2) Finds effective sample size (n_eff) based on number of foreground
    and background images. Calculated by 1/n_eff = 1/n_fg + 1/n_bg.
    3) Assumes each data point is distributed as a gaussian with mean
    given by Thomson spectrum and std given by sigma_sys / sqrt(n_eff).
    4) Under those assumptions, `lmfit.ModelResult.uvars` and
    `lmfit.ModelResult.dely_pred` give a parameter uncertainty and
    the prediction interval

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

    # Initialization
    fg = load_data(fg_paths)
    bg = load_data(bg_paths)
    im = np.array(fg - bg)
    top = 380
    bot = 220
    wavelengths = px_to_nm(np.arange(512))
    n_ef = 1 / ((1/len(fg_paths)) + (1/len(bg_paths)))
    sig_ef = SYS_SIGMA / np.sqrt(n_ef)
    spectrum = np.average(im[bot:top], axis=0)    

    res = fit_model(
        spectrum, 
        wavelengths, 
        sigma = sig_ef, 
        params_to_vary=['n', 'temp', 'scale'] # Add 'zero' to adjust baseline
    )

    # Get parameters from model
    fit = res.best_fit
    vals = res.summary()['params']
    fit_n = vals[0][1]
    fit_T = vals[1][1]
    fit_scale = vals[2][1]
    fit_zero = vals[3][1]

    # Uncertainty stuff
    # Prediction interval is where we expect to see data
    # Confidence interval is where we expect to see the mean
    res.eval_uncertainty()
    dely_pred = res.dely_predicted
    dely = res.dely

    # Graphing/reporting!
    print('Fit results')
    print(f'Reduced Chi Squared: {res.redchi} (should be close to one)')
    print(f'T = {res.uvars['n']:.3e} cc, n = {res.uvars['temp']:.3f} eV')

    fig = plt.figure(figsize=(6,5), dpi=200)
    ax_raw, ax_fit = fig.subplots(2, sharex=True, height_ratios=(1.5, 5))

    # graph the raw spectrum on top
    ext = [wavelengths[0], wavelengths[-1], top, bot]
    ax_raw.imshow(
        clip_raw_for_visualize(im)[bot:top], 
        aspect='auto', 
        extent=ext
    )
    ax_raw.set_title('Raw spectrum')
    ax_raw.set_yticklabels('')
   
    # graph the fit
    arr_tmp = np.linspace(-10, 10, 21)
    mask = norm.pdf(arr_tmp, loc=0, scale = (4.5992/(2*np.sqrt(2*np.log(2)))))
    spectrum_smooth = np.convolve(spectrum, mask, mode='same')
    spectrum_smooth = notch(spectrum_smooth, wavelengths)

    ax_fit.plot(
        wavelengths, 
        (spectrum_smooth / fit_scale) + fit_zero, 
        c='b', 
        zorder=-7
    )
    ax_fit.plot(
        wavelengths, 
        (fit / fit_scale) + fit_zero, 
        c='k', 
        label=f'n = ${res.uvars['n']:.2eL}$ cm$^{{-3}}$'\
              f'\nT = ${res.uvars['temp']:.2fL}$ eV'
    )

    pred_bar_top = ((fit + dely_pred) / fit_scale) + fit_zero
    pred_bar_bot = ((fit - dely_pred) / fit_scale) + fit_zero
    conf_bar_top = ((fit + dely) / fit_scale) + fit_zero
    conf_bar_bot = ((fit - dely) / fit_scale) + fit_zero

    ax_fit.fill_between(
        wavelengths, 
        pred_bar_bot, 
        pred_bar_top, 
        color='r', 
        alpha=0.2, 
        label='Prediction interval'
    )
    ax_fit.fill_between(
        wavelengths, 
        conf_bar_bot, 
        conf_bar_top, 
        color='r', 
        alpha=1, 
        label='Confidence interval'
    )
    
    ax_fit.grid(zorder=-10, c='k', linewidth=0.5)
    ax_fit.set_xticks([523, 526, 529, 532, 535, 538, 541])
    ax_fit.set_xlabel(r"Wavelength [nm]")
    ax_fit.set_ylabel('Normalized Intensity [A.U.]')
    ax_fit.set_title('Fit Thomson Scattering Curve')
    ax_fit.set_ylim(-0.2, 1.3)
    plt.legend()

    #plt.savefig('save_path')
    plt.show()


if __name__ == '__main__':
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