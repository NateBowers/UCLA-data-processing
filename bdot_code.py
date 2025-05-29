import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import json
import lmfit
import time
from scipy.integrate import cumulative_trapezoid
import os


# CONSTANTS
mu_0 = 4 * np.pi * 10e-7




class _Probe():
    """Container class for one, three axis probes. Should not be called"""
    def __init__(self, 
                 number: int | None = None, 
                 name: str | None = None):
        """
        Initialize name, number, and a few flags. Either a name or number
        must be passed

        Args:
            number (int | None, optional): The number of the probe. 
                Defaults to None.
            name (str | None, optional): The name of the probe. 
                Defaults to None.

        Raises:
            TypeError: Checks the probe number is an int
            TypeError: Checks the probe name is a string
            ValueError: Checks that either probe name or number is passed
        """
        
        if number is not None and type(number) is not int:
            raise TypeError('Probe number must be an integer')
        if type(name) is not str and name is not None:
            raise TypeError('Probe name must be a string')
        if number == -1 and name == 'unnamed':
            raise ValueError('Probe must have a name or number')
        self.name = name
        self.num = number

        self.calibrated = False
        self.loaded_params = False
        
    def _load(self, 
              path: str
            ) -> tuple:
        """Load data

        Args:
            path (str): _description_

        Returns:
            tuple: (frequency (Ang), real component, imaginary component)
        """
        freq, mag, phase = np.genfromtxt(path, skip_header=15).T
        freq = freq * 2 * np.pi
        phase = phase * np.pi / 180
        re_PjBi = mag * np.cos(phase)
        im_PjBi = mag * np.sin(phase)
        return freq, re_PjBi, im_PjBi

    def _re_curve_meinecke(self, 
                           w: float, 
                           a: float, 
                           tau: float, 
                           tau_s: float
                        ) -> float:
        """ 
        Real component of Vmeas/Vref as defined in equation (94) of Prof
        Meinecke's thesis (p. 100). 
        
        Args:
            w (float): angular frequency
            a (float): cross sectional area of probe tip
            tau (float): time delay from the wires
            tau_s (float): relaxation 
        
        Returns: 
            y (float): Real component of Vmeas/Vref
        """
        return self.factor * ((a * (w ** 2) * (tau_s - tau)) /
                            (1 + (tau_s * w) ** 2))

    def _im_curve_meinecke(self,
                           w: float,
                           a: float,
                           tau: float,
                           tau_s: float
                        ) -> float:
        """ 
        Imaginary component of Vmeas/Vref as defined in equation (94) of Prof
        Meinecke's thesis (p. 100). 
        
        Args:
            w (float): angular frequency
            a (float): cross sectional area of probe tip
            tau (float): time delay from the wires
            tau_s (float): relaxation 
        
        Returns: 
            y (float): Real component of Vmeas/Vref
        """
        return self.factor * ((a * tau * tau_s * (w**3) + a * w) / 
                            (1 + (tau_s * w) ** 2))
    







class OneAxisProbe(_Probe):
    """Contains functionality for probe calibration, report generation,
    and magnetic field reconstruction. See the docstrings for the 
    methods below for details on all the features implimented.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self, folder: str):
        """load bdot calibration data, converts readout from mag, phase to 
           re, im, and transforms freq to angular freq. This method assumes
           data is formatted how we were collecting data from the UCLA 
           calibration setup

        Args:
            folder (str): folder holds 1IN.TXT where the first 15 rows are 
            header, the first column is frequency (Hz) the 2nd magnitude 
            (linear unitless), and 3rd phase (deg). folder also includes 
            setup.json which includes the gain (g), Helmholtz radius (r), 
            resister measured across (R_p) and number of loops in the probe 
            (N).
        """
        with open(f'{folder}/setup.json', 'r') as file:
            d = json.load(file)
            self.factor = ((d['g'] * d['N'] * mu_0 * 16) /
                           (d['R_p'] * d['r'] * (5**1.5)))
            self.N = d['N']
        file.close()

        self.freq, self.re, self.im = super()._load(f'{folder}/1IN.TXT')

    def clip(self, low: int=10, high: int=-1):
        """Clips freq, re, im data to remove noisy data at the high or low 
           ends to help clean up data before calibrating.

        Args:
            low (int, optional): Lower bound for clipping (inclusive). 
            Defaults to 10.
            high (int, optional): Upper bound for clipping (exclusive). 
            Defaults to -1.
        """
        self.freq = self.freq[low:high]
        self.re = self.re[low:high]
        self.im = self.im[low:high]

    def graph_raw_data(self, 
                       axs=False, 
                       show=False
                    ) -> matplotlib.axes.Axes:
        """Graph the the real and imaginary parts of the calibration data on
           the same plot but with seperate y axes to properly scale

        Args:
            axs (bool, optional): If passed, the plot is drawn on the given 
            axis. Defaults to False.
            show (bool, optional): Whether or not to show the graph.
            Defaults to False.

        Returns:
            matplotlib.axes.Axes: an axis with the raw data graphed on it.
        """
        if axs:
            axs.set_ylabel('Linear units', color='red')
            axs.plot(self.freq*1e-6, self.re, 
                     label='Real part of V_meas/V_ref', color='red')
            axs.set_xlabel('Angular frequency (Mrad/s)')
            ax = axs.twinx()
            ax.set_ylabel('Linear units', color='blue')
            ax.plot(self.freq*1e-6, self.im, 
                    label='Imaginary part of V_meas/V_ref', color='blue')
            return axs, ax
        else:
            fig, ax1 = plt.subplots()
            ax1.set_ylabel('Linear units', color='red')
            ax1.plot(self.freq*1e-6, self.re, 
                     label='Real part of V_meas/V_ref', color='red')
            ax1.set_xlabel('Angular frequency (Mrad/s)')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Linear units', color='blue')
            ax2.plot(self.freq*1e-6, self.im, 
                     label='Imaginary part of V_meas/V_ref', color='blue')
            fig.legend()
            fig.tight_layout
        if show:
            plt.show()

    def load_params(self, path: str):
        """Load calibrated parameters into the probe

        Args:
            path (str): Path to the json file contianing the probe 
	    parameters

        Raises:
            ValueError: Ensures that only 1in probes can be loaded into the 
            OneAxisProbe object
        """
        with open (path) as f:
            data = json.load(f)
            if data['type'] != '1in':
                raise ValueError('Can only load 1in probes into ' \
                                    'OneAxisProbe')
            self.a = data['a']
            self.tau = data['tau']
            self.tau_s = data['tau_s']
            self.N = data['N']
            if self.num is None:
                self.num = data['num']
            elif self.name is None:
                self.name = data['name']
        self.loaded_params = True

    def _objective(self,
                    params: lmfit.Parameters,
                    freq: np.array,
                    re_true: np.array,
                    im_true: np.array
                ) -> np.array:
        
        """
        Objective function to be minimized when fitting the theoretical
        curves to calibration data. The quantity minimized is a list of 
        the signed differences between the real and predicted real and
        imaginary components of the probe voltage responce.

        Arguments:
            params (lmfit.Parameters): param dict containing a, tau, and tau_s
            freq (np.array): frequency array in rad/s
            re_true (np.array): real component of measured voltage responce in
                unitless
            im_true (np.array): imaginary component of measured voltage
                responce in unitless

        Returns:
            np.array: array of signed differences
        """
        
        a = params['a']
        tau = params['tau']
        tau_s = params['tau_s']

        re_pred = np.array(super()._re_curve_meinecke(freq, a, tau, tau_s))
        im_pred = np.array(super()._im_curve_meinecke(freq, a, tau, tau_s))

        resid_re = re_true - re_pred
        resid_im = im_true - im_pred
        return np.concat((resid_re, resid_im))

    def calibrate(self, 
                  save: bool=True, 
                  verbose: bool=True, 
                  overwrite: bool=False, 
                  notes: str=''
                ) -> tuple:
        
        """Calibration routine for bdot probe. See _objective() for details
        on the minimized cost function.

        Args:
            saves (bool): Whether to save the calibration results. If true, 
                results are saved to 'bdot_data/params/probe_{self.num}.json'
                Defaults to True
            verbose (bool): Whether to print calibration results to the
                command line. Defaults to True
            overwrite (bool): If a probe with the same number has previously
                been calibrated, whether to overwrite existing calibration 
                results when saved.

        Raises:
            AttributeError: raised if results are attempted to be overwriten
                without overwrite=True

        Returns:
            tuple: calibrated values for (a, tau, tau_s).
        """


        self.calibrated = True

        params = lmfit.Parameters()
        params.add_many(('a',  0.00064516), ('tau', 3e-8), ('tau_s', 3e-8))
        
        result = lmfit.minimize(self._objective, 
                                params, 
                                args=(self.freq, self.re, self.im))
        a, tau, tau_s = result.params.valuesdict().values()
        self.a = a
        self.tau = tau
        self.tau_s = tau_s
        self.result = lmfit.fit_report(result)
        if save:
            save_data = {
                'num': self.num,
                "name": self.name,
                "type": '1in',
                "N": self.N,
                "a": a,
                "tau": tau,
                "tau_s": tau_s,
                "calibration_info" :{
                    "calibration_time": time.strftime('%X %x %Z', 
                                                      time.localtime()),
                    "calibration_notes": notes,
                    "calibration_results": lmfit.fit_report(result),
                },
            }
        
            save_path = f'bdot_data/params/probe_{self.num}.json'
            if not overwrite and os.path.exists(save_path):
                raise AttributeError(f'A file already exists at \
                                     {save_path} and overwrite = false')
            else:
                with open(save_path, 'w') as save_file:
                    json.dump(save_data, save_file, indent=4)

        if verbose:
            print(f'{'-'*80}{'\n'}FIT REPORT FOR PROBE NUMBER {self.num}')
            print(self.result)
        else:
            print(f'Probe number {self.num} is calibrated')

        return a, tau, tau_s
    
    def gen_probe_report(self):
        """Generate PDF report on the calibration outcomes"""

        if not (self.calibrated or self.loaded_params):
            raise ValueError('Probe must have loaded parameters before ' \
                             'generating report')
        
        with PdfPages(f'bdot_data/reports/probe_{self.num}_report.pdf') as pdf:
            fig1 = plt.figure(figsize=(8.5, 11))
            header, plot_fig = fig1.subfigures(nrows=2, 
                                               ncols=1, 
                                               height_ratios=[3, 8])
            header.text(0.5, 0.5, f'Calibration data for probe number \
                        {self.num} ({self.name})', wrap=True, ha='center', 
                        fontvariant='small-caps', fontsize='x-large')
            header.text(0.5, 0.4, f'Calibrated on \
                        {time.strftime('%X %x %Z', time.localtime())}', 
                        ha='center')
            plot_fig.text(0.5,0.05, s='1', ha='center')
            ax = plot_fig.add_subplot()
            self.graph_raw_data(ax, show=False)
            plt.title('Raw scope data')
            plt.legend()
            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.9)
            pdf.savefig()
            plt.close()
            plt.clf()


            page2 = plt.figure(figsize=(8.5, 11))
            header, plot_fig = page2.subfigures(nrows=2, ncols=1, 
                                               height_ratios=[3, 8])
            header.text(0.5, 0.8, 'Fit results', ha='center', 
                        fontsize='large')
            header.text(0.5,0, self.result, ha='center', 
                        ma='left', fontsize='small')
            axs = plot_fig.subplots(2, 1, sharex=True)            
            title = ['Re', 'Im']
            data_true = [self.re, self.im]
            data_pred = [self._re_curve_meinecke(self.freq, self.a, 
                                                 self.tau, self.tau_s), 
                         self._im_curve_meinecke(self.freq, self.a, 
                                                 self.tau, self.tau_s)]
            for i, ax in enumerate(axs):
                ax.set_title(f'Data v. Predicted Fit for {title[i]}'
			'f Component')
                ax.set_xlabel('Angular frequency (Mrad/s)')
                ax.set_ylabel('Linear unitless')
                ax.plot(self.freq*1e-6, data_true[i], color='blue', 
                        label='Data')
                ax.plot(self.freq*1e-6, data_pred[i], color='red', 
                        linestyle='--', label='Predicted fit')
                ax.grid(color='darkgray')
                ax.minorticks_on()
                ax.grid(which='minor', linestyle='--', color='lightgray')
                ax.legend()
            plot_fig.text(0.5,0.05, 2, ha='center')
            plt.subplots_adjust(0.15, 0.15, 0.85, 0.9, hspace=0.25)
            pdf.savefig()
            plt.close()

    def reconstruct(self, 
                    voltages: np.array, 
                    times: np.array, 
                    g: float, 
                    b_0: float=0,
                    correct_drift: bool = False) -> np.array:
        """Reconstructs magnetic field from voltage reading by numerically
        integrating equation (10) in E. Everson (2009).


        Args:
            voltages (np.array): measured probe voltages
            times (np.array): time for each voltage measurement
            g (float): gain on scope
            b_0 (float, optional): initial magnetic field. Defaults to 0.

        Raises:
            Exception: Raised if the probe is not calibrated or has not had
                parameters loaded
            Exception: Ensures voltages is a 1 dimensional array
            Exception: Raised if voltages and times have different shapes

        Returns:
            np.array: reconstructed field.
        """
        if not (self.loaded_params or self.calibrated):
            raise Exception('Probe must be calibrated or have parameters' \
            ' loaded before reconstructing.')
        if np.ndim(voltages) != 1:
            raise Exception(f'voltages must be a 1 dimensional array, but \
                            voltages has shape {voltages.shape}.')
        if len(voltages) != len(times):
            raise Exception(f'voltages and times must have the same shape,\
                             but voltages is {voltages.shape} and times is \
                             {times.shape}.')
        
        if correct_drift:
            num_timesteps = times.shape[0]
            voltages -= np.average(voltages[:int(num_timesteps*0.04)])


        field = np.zeros_like(voltages)
        field[0] = b_0

        const1 = 1 / (self.a * self.N * g)
        const2 = field[0] - self.tau_s * const1 * voltages[0]

        voltages_integrated = cumulative_trapezoid(voltages, x=times)
        for i in range(len(voltages_integrated)):
            field[i+1] = const1 * (voltages_integrated[i] + 
                                   self.tau_s*voltages[i]) + const2
        return field
    
    def reconstruct_array(self, 
                          volts_arr: np.array, 
                          times_arr: np.array, 
                          g: float, 
                          b_0: float=0,
                          **kwargs) -> np.array:
        """Applies reconstruct() to each row in an m x n array of voltages,
           where the zeroth axis represents different shots and the first
           axis represents the measurements along time of an individual
           shot.

        Args:
            volts_arr (np.array): voltage values, where axis 1 is the axis to
                 be integrated along
            times_arr (np.array): time values, where axis 1 is the axis to be 
                integrated along
            g (float): scope gain
            b_0 (float, optional): initial field. Defaults to 0.

        Raises:
            ValueError: _description_

        Returns:
            np.array: m x n array where each row is the reconstructed field
            corresponding to that row in the input voltages array.
        """

        
        if volts_arr.shape != times_arr.shape:
            raise ValueError('volts_arr and times_arr must has the \
	    	same shape')
        
        num_rows = volts_arr.shape[0]
        if type(b_0) is int:
            b_0 = np.full(num_rows, b_0)
        field_arr = []
        for i, row in enumerate(volts_arr):
            field_row = self.reconstruct(row, times_arr[i], g, b_0[i], kwargs)
            field_arr.append(field_row)
        return np.array(field_arr)
    




### Note: j indexes over P (which is probe axis), i indexes over B 
# (which is field axis)

class ThreeAxisProbe(_Probe):
    def __init__(self, 
                 number: int | None=None,
                 name: str | None=None,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

    
    def load_data(self, folder: str):
        """Load bdot calibration data from each nine runs. Assumes each
        file is named PJBI.TXT where J in {x,y,z} is the probe axis and 
        I is the applied field axis in the calibration. Also assumes 
        the data is formatted in the same fashion as we collected at UCLA.
        Also checks to make sure that the samples frequencies are the
        same for run.

        Args:
            folder (str): folder tht holds PJBI.TXT for I,J in (X,Y,Z) where
            the first 15 rows are header, the first column is frequency (Hz)
            the 2nd magnitude (linear unitless), and 3rd phase (deg). folder
            also includes setup.json which includes the gain (g), Helmholtz
            radius (r), resister measured across (R_p) and number of loops
            in the probe (N).

        Raises:
            ValueError: Makes sure each probe was calibrated with the same
            set of frequencies.
        """
        self.folder = folder
        
        with open(f'{folder}/setup.json', 'r') as file:
            d = json.load(file)
            self.factor = ((d['g'] * d['N'] * mu_0 * 16) / 
                           (d['R_p'] * d['r'] * (5**1.5)))
            self.N = d['N']
        file.close()

        # Re(PjBi) = const * a_ij * f_j(omega)
        # X -> 0, Y -> 1, Z -> 2
        # self.*_ij is data corresponding to PjBi
        f_00, self.r_00, self.i_00 = super()._load(f'{folder}/PXBX.TXT')
        f_10, self.r_10, self.i_10 = super()._load(f'{folder}/PXBY.TXT')
        f_20, self.r_20, self.i_20 = super()._load(f'{folder}/PXBZ.TXT')

        # corresponds to second column of A
        f_01, self.r_01, self.i_01 = super()._load(f'{folder}/PYBX.TXT')
        f_11, self.r_11, self.i_11 = super()._load(f'{folder}/PYBY.TXT')
        f_21, self.r_21, self.i_21 = super()._load(f'{folder}/PYBZ.TXT')

        # corresponds to third column of A
        f_02, self.r_02, self.i_02 = super()._load(f'{folder}/PZBX.TXT')
        f_12, self.r_12, self.i_12 = super()._load(f'{folder}/PZBY.TXT')
        f_22, self.r_22, self.i_22 = super()._load(f'{folder}/PZBZ.TXT')

        if not (np.array_equal(f_00, f_01) and
                np.array_equal(f_00, f_02) and
                np.array_equal(f_00, f_10) and
                np.array_equal(f_00, f_11) and
                np.array_equal(f_00, f_12) and
                np.array_equal(f_00, f_20) and
                np.array_equal(f_00, f_21) and
                np.array_equal(f_00, f_22)
               ):
            raise ValueError('All probes must be sampled at the same \
                             frequencies')
        else:
            self.f = f_00




    def graph_raw_data(self, 
                       fig=False, 
                       show=False) -> matplotlib.figure.Figure:
       
        if not fig:
            fig = plt.figure(figsize=(8.5,8), constrained_layout=True)
        subfigs = fig.subfigures(nrows=3, ncols=1)
        titles = ['On axis x', 'On axis y', 'On axis z']
        y_re = [self.r_00, self.r_11, self.r_22]
        y_im = [self.i_00, self.i_11, self.i_22]

        im_bound = np.max(np.abs(np.array(y_im)))
        re_bound = np.max(np.abs(np.array(y_re)))

        for i, subfig in enumerate(subfigs):
            ax_re = subfig.add_subplot()
            ax_re.plot(self.f*1e-6, y_re[i], label='Real part', color='red')
            ax_re.plot(self.f*1e-6, np.sqrt(y_re[i]**2 + y_im[i]**2), 
                       label='Magnitude', linestyle='--', color='green')
            ax_re.set_title(titles[i])
            ax_re.set_ylim(-re_bound, re_bound)
            ax_re.set_xlabel('Angular frequency (Mrad/s)')
            ax_re.set_ylabel('Linear units', color='red')
            ax_im = ax_re.twinx()
            ax_im.plot(self.f*1e-6, y_im[i], label='Imag part', color='blue')
            ax_im.set_ylabel('Linear units', color='blue')
            ax_im.set_ylim(-im_bound, im_bound)
            ax_im.axhline(color='black', linestyle='--', alpha=0.5)
            subfig.legend()
    
        if show:
            plt.show()

        return fig



    def clip(self, low: int=10, high: int=-1):
        """Clips freq, re, im data to remove noisy data at the high or low 
           ends to help clean up data before calibrating.

        Args:
            low (int, optional): Lower bound for clipping (inclusive). 
            Defaults to 10.
            high (int, optional): Upper bound for clipping (exclusive). 
            Defaults to -1.
        """

        self.r_00 = self.r_00[low:high]
        self.i_00 = self.i_00[low:high]
        self.r_10 = self.r_10[low:high]
        self.i_10 = self.i_10[low:high]
        self.r_20 = self.r_20[low:high]
        self.i_20 = self.i_20[low:high]
        self.r_01 = self.r_01[low:high]
        self.i_01 = self.i_01[low:high]
        self.r_11 = self.r_11[low:high]
        self.i_11 = self.i_11[low:high]
        self.r_21 = self.r_21[low:high]
        self.i_21 = self.i_21[low:high]
        self.r_02 = self.r_02[low:high]
        self.i_02 = self.i_02[low:high]
        self.r_12 = self.r_12[low:high]
        self.i_12 = self.i_12[low:high]
        self.r_22 = self.r_22[low:high]
        self.i_22 = self.i_22[low:high]
        self.f = self.f[low:high]

        pass

    
    def _objective(self, 
                    params, 
                    freq, 
                    v_re_true, 
                    v_im_true) -> np.array:
        
        """Objective function to be minimized"""
        a1 = params['a_0']
        a2 = params['a_1']
        a3 = params['a_2']
        tau = params['tau']
        tau_s = params['tau_s']

        predict_vec_re = np.array([
            super()._re_curve_meinecke(freq, a1, tau, tau_s),
            super()._re_curve_meinecke(freq, a2, tau, tau_s),
            super()._re_curve_meinecke(freq, a3, tau, tau_s)
        ])

        predict_vec_im = np.array([
            super()._im_curve_meinecke(freq, a1, tau, tau_s),
            super()._im_curve_meinecke(freq, a2, tau, tau_s),
            super()._im_curve_meinecke(freq, a3, tau, tau_s)
        ])

        resid_re = (v_re_true - predict_vec_re).flatten()
        resid_im = (v_im_true - predict_vec_im).flatten()

        return np.concat((resid_re, resid_im))


    
        
    

    

    
    

    def calibrate(self, 
                  save: bool=False, 
                  verbose: bool=False, 
                  overwrite: bool=False,
                  notes: str=''):
        """Routine to calibrate the probe by minimizing the residues of
           _objective()

        Args:
            save (bool, optional): Save the probe parameters to 
               "./bdot_data/bdot_probe_data/probe_k.json" where k is the 
               probe number. Defaults to False.
            verbose (bool, optional): Whether to print calibration results.
               Defaults to False.
            overwrite (bool, optional): Whether to overwrite probe_k.json 
               if it already exists. Defaults to False.

        Raises:
            AttributeError: If the probe calibration data already exists.

        Returns:
            tuple: A_mat, Tau, Tau_s
        """
        
        re_vec_j_is_0 = np.array([self.r_00, self.r_01, self.r_02])
        re_vec_j_is_1 = np.array([self.r_10, self.r_11, self.r_12])
        re_vec_j_is_2 = np.array([self.r_20, self.r_21, self.r_22])

        im_vec_j_is_0 = np.array([self.i_00, self.i_01, self.i_02])
        im_vec_j_is_1 = np.array([self.i_10, self.i_11, self.i_12])
        im_vec_j_is_2 = np.array([self.i_20, self.i_21, self.i_22])


        ### j = 0
        params_j_is_0 = lmfit.Parameters()
        params_j_is_0.add_many(('a_0', 1e-6), ('a_1', 1e-6), ('a_2', 1e-6),
                               ('tau', 1e-8), ('tau_s', 1e-8))
        
        params_j_is_1 = lmfit.Parameters()
        params_j_is_1.add_many(('a_0', 1e-6), ('a_1', 1e-6), ('a_2', 1e-6), 
                               ('tau', 1e-8), ('tau_s', 1e-8))
        
        params_j_is_2 = lmfit.Parameters()
        params_j_is_2.add_many(('a_0', 1e-6), ('a_1', 1e-6), ('a_2', 1e-6),
                               ('tau', 1e-8), ('tau_s', 1e-8))
        
        result_j_is_0 = lmfit.minimize(self._objective,
                                       params_j_is_0,
                                       args=(self.f, re_vec_j_is_0, 
                                             im_vec_j_is_0))
        
        result_j_is_1 = lmfit.minimize(self._objective,
                                       params_j_is_1,
                                       args=(self.f, re_vec_j_is_1, 
                                            im_vec_j_is_1))
        
        result_j_is_2 = lmfit.minimize(self._objective,
                                       params_j_is_2,
                                       args=(self.f, re_vec_j_is_2, 
                                             im_vec_j_is_2))
        
        a_00, a_10, a_20, tau_0, tau_s0 = (result_j_is_0.params
                                            .valuesdict().values())
        a_01, a_11, a_21, tau_1, tau_s1 = (result_j_is_1.params
                                            .valuesdict().values())
        a_02, a_12, a_22, tau_2, tau_s2 = (result_j_is_2.params
                                            .valuesdict().values())
        
        self.j0_report = lmfit.fit_report(result_j_is_0)
        self.j1_report = lmfit.fit_report(result_j_is_1)
        self.j2_report = lmfit.fit_report(result_j_is_2)
        
        eps = 1e-10
        A_mat = np.array([
            [a_00, a_01, a_02],
            [a_10, a_11, a_12],
            [a_20, a_21, a_22],
        ])

        A_mat[np.abs(A_mat) < eps] = 0
        self.a = A_mat

        Tau_vec = np.array([tau_0, tau_1, tau_2])
        Tau_vec[np.abs(Tau_vec)< eps] = 0
        self.tau = Tau_vec

        Tau_s_vec = np.array([tau_s0, tau_s1, tau_s2])
        Tau_s_vec[np.abs(Tau_s_vec)< eps] = 0
        self.tau_s = Tau_s_vec


        if save:
            save_data = {
                "num": self.num,
                "name": self.name,
                "a": self.a.tolist(),
                "tau": self.tau.tolist(),
                "tau_s": self.tau_s.tolist(),
                "N":self.N,
                "type": '1mm',
                "calibration_info":{
                    'calibration_time': time.strftime('%X %x %Z', 
                                                      time.localtime()),
                    'calibration_notes': notes,
                    'calibration_results_j_is_0': self.j0_report,
                    'calibration_results_j_is_1': self.j1_report,
                    'calibration_results_j_is_2': self.j2_report,
                }
            }
            save_path = f'bdot_data/params/probe_{self.num}.json'

            if not overwrite and os.path.exists(save_path):
                raise AttributeError(f'A file already exists at {save_path} '
                                     f'and overwrite = false')
            else:
                with open(save_path, 'w') as save_file:
                    json.dump(save_data, save_file, indent=4)

        if verbose:
            print(f'{'-'*80}{'\n'}REPORT FOR PROBE X AXIS (j=0)')
            print(lmfit.fit_report(result_j_is_0))
            print(f'{'-'*80}{'\n'}REPORT FOR PROBE Y AXIS (j=1)')
            print(lmfit.fit_report(result_j_is_1))
            print(f'{'-'*80}{'\n'}REPORT FOR PROBE Z AXIS (j=2)')
            print(lmfit.fit_report(result_j_is_2))
            print(f'{'-'*80} Full probe parameters:')
            print(f'A={self.A}')
            print(f'Tau={self.Tau}')
            print(f'Tau_s={self.Tau_s}')

        self.calibrated = True
        
        return self.a, self.tau, self.tau_s
    
    def graph(self, raw=False, results=False, save=True):

        if self.calibrated and not results:
            raise ValueError('Graphs can only be generated '
                'for calibrated probes')
        

        if raw:
            fig_raw = plt.figure(figsize=(5,5))
            fig_raw.suptitle('Trace readouts for re, im data')

            axs = fig_raw.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
            sequence_list = ['PXBX', 'PXBY', ]
            for ax in axs.flatten():
                ax.set_xlabel('Angular frequency (w)')
                ax.set_ylabel('V_measure/V_ref (mU)')
                ax.legend()
            
            axs[0,0].set_title('PXBX')
            axs[0,0].plot(self.freq_00, self.re_00)
            axs[0,0].plot(self.freq_00, self.im_00)
            
        pass

    def _predict_indiv(self, freq, i, j):
        a = self.a[i,j]
        tau = self.tau[j]
        tau_s = self.tau_s[j]

        re_predict = self._re_curve_meinecke(freq, a, tau, tau_s)
        im_predict = self._im_curve_meinecke(freq, a, tau, tau_s)

        return (re_predict, im_predict)

        

    def gen_probe_report(self):
        
        with PdfPages(f'bdot_data/reports/probe_{self.num}_report.pdf') as pdf:
            page1 = plt.figure(figsize=(8.5,11))
            header, plot_fig, footer = page1.subfigures(nrows=3, 
                                               ncols=1, 
                                               height_ratios=[3, 7, 1])
            header.text(0.5, 0.5, f'Calibration data for probe number '
                        f'{self.num} ({self.name})', wrap=True, ha='center', 
                        fontvariant='small-caps', fontsize='x-large')
            header.text(0.5, 0.4, f'Calibrated on {time.strftime('%X %x %Z', 
                        time.localtime())}', ha='center')
            plot_fig = self.graph_raw_data(plot_fig, show=False)
            plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2)
            footer.text(0.5,0.4, s='1', ha='center')
            pdf.savefig()
            plt.close()

            data_true = [[self.r_00, self.i_00],
                         [self.r_11, self.i_11],
                         [self.r_22, self.i_22]]
            data_pred = [self._predict_indiv(self.f, 0,0),
                         self._predict_indiv(self.f, 1,1),
                         self._predict_indiv(self.f, 2,2)]

            for i in range(3):
                axis = ['x', 'y', 'z']
                page_i = plt.figure(figsize=(8.5, 11))
                header, plot_fig = page_i.subfigures(nrows=2, ncols=1, 
                                                height_ratios=[4, 7])
                header.text(0.5, 0.82, f'Fit results for probe on {axis[i]} '
                            f'axis', ha='center', fontsize='large')
                header.text(0.5,0, self.j0_report, ha='center', 
                            ma='left', fontsize='small')
                
                true = data_true[i]
                pred = data_pred[i]
                
                
                axs = plot_fig.subplots(2, 1, sharex=True)            
                title = ['Re', 'Im']
                
                for j, ax in enumerate(axs):
                    ax.set_title(f'Data v. Predicted Fit for {title[j]} '
                                 f'Component of on axis for {axis[i]} probe')
                    ax.set_xlabel('Angular frequency (Mrad/s)')
                    ax.set_ylabel('Linear unitless')
                    ax.plot(self.f*1e-6, true[j], color='blue', 
                            label='Data')
                    ax.plot(self.f*1e-6, pred[j], color='red', 
                            linestyle='--', label='Predicted fit')
                    ax.grid(color='darkgray')
                    ax.minorticks_on()
                    ax.grid(which='minor', linestyle='--', color='lightgray')
                    ax.legend()
                plot_fig.text(0.5,0.05, s=(i+2), ha='center')
                plt.subplots_adjust(0.15, 0.15, 0.85, 0.9, hspace=0.25)



                pdf.savefig()
                plt.close()


    def load_params(self,
                    path,
                    change_num=False):
        with open(path, 'r') as file:
            data = json.load(file)
        
        if self.num != data['num'] and not change_num:
            raise ValueError(f'Attempting to load data from probe'
                             f' {data['num']} into probe {self.num}')
        
        self.params_loaded = True
    
        self.a = data['a']
        self.tau = data['tau']
        self.tau_s = data['tau_s']
        self.N = data['N']
        pass
    

    def reconstruct_field(self, 
                         v_x: np.array,
                         v_y: np.array,
                         v_z: np.array, 
                         times: np.array,
                         g: float,
                         correct_drift: bool = False,
                         b_0: np.array = None,
                    ) -> tuple:
        """Reconstruct three dimensional magnetic field from probe
           measurements for an individual run. Takes into accont the off-axis 
           cross sectional areas. 

        Args:
            v_x (np.array): voltage readings from the x-axis
            v_y (np.array): voltage readings from the y-axis
            v_z (np.array): voltage readings from the z-axis
            times (np.array): time values for each individual voltage
                measurement
            g (float): scope gain
            correct_drift (bool, optional): Whether or not to add an offset to
                the x, y, and z voltages equal to the average of the first
                0.04 elements. Defaults to False.
            b_0 (np.array | None, optional): 3 element array containing
                initial field values for x, y, and z axes. Defaults to None.

        Raises:
            ValueError: Raised if the voltages and times are not of the
                same length

        Returns:
            tuple: three np.arrays corresponding to the reconstructed magnetic
                field for the x, y, and z axes, respectively.            
        """
        if not self.a:
            raise ValueError('Probe must be calibrated before \
	    reconstructing fields')
        if len(v_x) != len(v_y) != len(v_z) != len(times):
            raise IndexError(f'x, y, z traces and times must have the same'
                             f'length, but arrays of length {len(v_x)}, '
                             f'{len(v_y)}, {len(v_z)}, and {len(times)} were '
                             f'passed.')
        
        if correct_drift:
            num_timesteps = times.shape[0]
            v_x -= np.average(v_x[:int(num_timesteps*0.04)])
            v_y -= np.average(v_y[:int(num_timesteps*0.04)])
            v_z -= np.average(v_z[:int(num_timesteps*0.04)])

        field = np.zeros((len(times),3))
        if b_0:
            field[0] = b_0

        v_x_int = cumulative_trapezoid(v_x, times)
        v_y_int = cumulative_trapezoid(v_y, times)
        v_z_int = cumulative_trapezoid(v_z, times)

        v_vec = np.vstack((v_x, v_y, v_z))
        v_int_vec = np.vstack((v_x_int, v_y_int, v_z_int))

        A_inv = np.linalg.inv(self.a) / (self.N * g)

        ts = self.tau_s

        for i in range(len(v_x_int)):
            field[i+1] = A_inv @ (v_int_vec[:,i] 
                                + np.multiply(ts, (v_vec[:,i] - v_vec[:,0])))   

        return field.T[0], field.T[1], field.T[2]
    

    def reconstruct_array(self,
                          v_x_arr: np.array,
                          v_y_arr: np.array,
                          v_z_arr: np.array,
                          time_arr: np.array,
                          gain: float,
                          **kwargs,
                        ) -> tuple:
        """Calles reconstruct_field on each run (e.g., row) in an array of
           voltage readings. See reconstruct_field for implimentation details.

        Args:
            v_x_arr (np.array): x voltage array, where axis one is to be 
                integrated along.
            v_y_arr (np.array): y voltage array, where axis one is to be 
                integrated along.
            v_z_arr (np.array): z voltage array, where axis one is to be 
                integrated along.
            time_arr (np.array): time array, where axis one corresponds to 
                the time values for each voltage measurment.
            gain (float): scope gain.
            **kwargs (dict): passed to reconstruct_field. See reconstruct_field
                for information on additional arguments.

        Returns:
            tuple: three np.arrays for the reconstructed field along the 
                x, y, and z axes (respectively). Each row corresponds to a
                different shot.
        """
        

        field_x_arr = np.zeros_like(v_x_arr)
        field_y_arr = np.zeros_like(v_y_arr)
        field_z_arr = np.zeros_like(v_z_arr)

        for i in time_arr.shape[0]:
            x, y, z = self.reconstruct_field(v_x_arr[i],
                                             v_y_arr[i],
                                             v_z_arr[i],
                                             time_arr[i],
                                             gain,
                                             **kwargs)
            field_x_arr[i] = x
            field_y_arr[i] = y
            field_z_arr[i] = z

        return field_x_arr, field_y_arr, field_z_arr
        

    




if __name__ == '__main__':

    probe = OneAxisProbe(name = '1IN Probe New', number = 0)
    probe.load_data('bdot_data/1in_probe-05-28')
    probe.clip(10)
    probe.graph_raw_data(show=True)

    probe.calibrate(save=True, overwrite=True)
    # probe.gen_probe_report()

