import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.optimize import curve_fit


class TSResult(object):

	def __init__(self):
		pass

	# Raw and notched spectrum
	@property
	def spectrum_full(self):
		return self._spectrum_full
	
	@spectrum_full.setter
	def spectrum_full(self, arr: np.array):
		self._spectrum_full = arr

	@property
	def spectrum_notched(self):
		return self._spectrum_notched
	
	@spectrum_notched.setter
	def spectrum_notched(self, arr: np.array):
		self._spectrum_notched = arr

	# Programmed and real delay
	@property
	def set_delay(self):
		return self._set_delay

	@set_delay.setter
	def set_delay(self, value):
		self._set_delay = value
		pass

	@property
	def real_delay(self):
		return self._real_delay

	@real_delay.setter
	def real_delay(self, value: tuple):
		self._real_delay = value
		pass

	# Fit parameters with error from regression
	@property
	def fit_mean(self):
		return self._fit_mean
	
	@fit_mean.setter
	def fit_mean(self, value: tuple):
		self._fit_mean = value

	@property
	def fit_std(self):
		return self._fit_std
	
	@fit_std.setter
	def fit_std(self, value: tuple):
		self._fit_std = value

	@property
	def fit_amplitude(self):
		return self._fit_amplitude
	
	@fit_amplitude.setter
	def fit_amplitude(self, value: tuple):
		self._fit_amplitude = value

	# Calculate electron temp and density from fit parameters
	@property
	def T_e(self):
		# Electron temperature in eV
		return np.abs(0.903 * self._fit_std[0])

	@property
	def n_e(self):
		# Electron density in counts per cm^3
		return 5.6e16 * 1e-6 * self._fit_amplitude[0] * 512 / 19




class TSAnalyzer():

	NOTCH_LOW = 529.8
	NOTCH_LOW = 531
	NOTCH_HIGH = 534.2
	NOTCH_HIGH = 533
	RAYLEIGH = 1

	
	def __init__(self, file: str | h5py._hl.files.File):

		self.wavelengths = (np.arange(512) * 19.80636 / 511) + 522.918

		if type(file) == 'str':
			self._file = h5py.File(file)
		else:
			self._file = file

		ts = np.array(self._file['LeCroy:Ch2:Trace'])
		heater = np.array(self._file['LeCroy:Ch1:Trace'])
		times = np.array(self._file['LeCroy:Time'])
		delays = np.array(self._file['actionlist/TS:1w2wDelay'])
		images = [np.array(self._file[f'13PICAM1:Pva1:Image/image {n}']) for n in range(100)]

		unique_delays = np.unique(delays)

		dt = np.round(np.average(times[:,1:] - times[:,:-1]), 10)
		ts_max_idx = np.argmax(np.gradient(ts, axis=1), axis=1)
		heater_max_idx = np.argmax(np.gradient(heater, axis=1), axis=1)
		real_delays = (ts_max_idx - heater_max_idx) * dt * 1e9
		
		ts_max = np.max(ts, axis=1)
		ts_shutter = np.array((ts_max > 2), dtype=int)
		heater_max = np.max(heater, axis=1)
		heater_shutter = np.array([a != heater_max[i-1] for i, a in enumerate(heater_max)]).astype(int)

		shutter_delay_arr = np.array([ts_shutter, heater_shutter, delays, real_delays]).T
		info_arr = np.array(sorted(shutter_delay_arr, key=lambda x: x[2]))
		
		self.results = []
		
		for delay in unique_delays:

			delay_result = TSResult()

			fg_idxs = np.argwhere(np.all(info_arr[:,:3] == (1, 0, delay), axis=1)).T[0]
			bg_idxs = np.argwhere(np.all(info_arr[:,:3] == (0, 0, delay), axis=1)).T[0]
			fg_imgs = np.average([images[i] for i in fg_idxs], axis=0)
			bg_imgs = np.average([images[i] for i in bg_idxs], axis=0)

			shot_delay = info_arr[fg_idxs, 3]

			spectrum_full = np.sum((fg_imgs - bg_imgs) [215:375], axis=0)

			wspan = (np.max(self.wavelengths) - np.min(self.wavelengths)) / 2
			eval_w = np.linspace(-wspan, wspan, num=self.wavelengths.size)
			inst_func_arr = TSAnalyzer.gauss(
				x = eval_w,
				mu = 0,
				sigma = 0.3 / (2 * np.sqrt(2 * np.log(2))),
				amplitude = 1
			)
			inst_func_arr /= np.sum(inst_func_arr)

			spectrum_conv = np.convolve(spectrum_full, inst_func_arr, mode='same')
			notched_spectrum = self.remove_notch(spectrum_conv)		

			# TO DO:
			# Impliment a more robust curve fitting routine using RANSAC
			# Investigate the usefullness of a supergaussian fit (i.e., the power p (usually 2), is also a fitable parameter)
			popt, pcov = curve_fit(self.gauss, self.remove_notch(self.wavelengths), notched_spectrum, p0 = [532, 5, 1000], nan_policy='omit')
			mean, std_dev, amplitude = popt
			mean_err, std_dev_err, amplitude_err = np.square(np.diag(pcov))

			delay_result.set_delay = delay
			delay_result.real_delay = shot_delay
			delay_result.spectrum_full = spectrum_full
			delay_result.spectrum_notched = notched_spectrum
			delay_result.fit_mean = (mean, mean_err)
			delay_result.fit_std = (np.abs(std_dev), std_dev_err)
			delay_result.fit_amplitude = (amplitude, amplitude_err)

			self.results.append(delay_result)


	def plot_spectra(self, title: str = None, show: bool = False, save: str = None):

		fig = plt.figure(constrained_layout=True, figsize=(12, 8))
		fig.suptitle(title, fontsize=16)
		axs = fig.subplots(2, 3, sharex=True, sharey=True)
		for i, ax in enumerate(axs.flatten()):
			if i == 5:
				ax.set_visible(False)
			else:
				data = self.results[i]

				# Data plotting
				ax.plot(self.wavelengths, data.spectrum_full, color='b', alpha=0.25)
				ax.plot(self.wavelengths, data.spectrum_notched, color='b')
				fit = TSAnalyzer.gauss(self.wavelengths, data.fit_mean[0], data.fit_std[0], data.fit_amplitude[0])
				ax.plot(self.wavelengths, fit, color='r')

				# Fit reporting
				s = f'$T_e = {data.T_e:.2f}$ eV {'\n'}$n_e$ = {data.n_e:.2e} cm$^{{{-3}}}$'
				ax.text(0.05,0.95, s, transform=ax.transAxes, va='top', bbox=dict(edgecolor='k', facecolor='none'))



				# Graph formatting
				ax.set_title(f'$t_{{del}} = {data.set_delay}$ ns (actual = ${np.average(data.real_delay):.1f}±{np.std(data.real_delay):.1f}$ ns)')
				ax.set_xlabel(r'$\lambda$ (nm)')
				ax.set_ylabel('counts')
				ax.tick_params(labelbottom=True)
				ax.axhline(color='k')
				ax.set_xlim(min(self.wavelengths), max(self.wavelengths))
				loc = plticker.MultipleLocator(base=2)
				ax.xaxis.set_major_locator(loc)

				
		if show:
			plt.show()
		if save:
			plt.savefig(save)
	

	def remove_notch(self, arr):
		idx_low = np.argmin(np.abs(self.wavelengths - TSAnalyzer.NOTCH_LOW))
		idx_high = np.argmin(np.abs(self.wavelengths - TSAnalyzer.NOTCH_HIGH))
		output_arr = np.copy(arr)
		output_arr[idx_low:idx_high] = np.nan
		return output_arr

	@staticmethod
	def gauss(x, mu, sigma, amplitude):
		return (amplitude / np.sqrt(2 * np.pi * sigma ** 2) * 
		  np.exp( - np.square(x - mu) / (2 * sigma**2)))
	
	@staticmethod
	def inst_func(wavelengths):
		fwhm_factor = 2 * np.sqrt(2 * np.log(2))
		return TSAnalyzer.gauss(wavelengths, 0, 0.3 / fwhm_factor, 1)






if __name__ == '__main__':

	file = h5py.File('06-05/TS_CH_Location1_25V-2025-06-05.h5')
	data = TSAnalyzer(file)
	data.plot_spectra(show=True, title='CH, 25V, Location 1')





