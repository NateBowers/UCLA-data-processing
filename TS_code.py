import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.optimize import curve_fit
from scipy import integrate
from itertools import product


class TSResult(object):
	"""Data class to hold results from Thomson data analysis. Includes the
	following data points:

	- `spectrum_full`: An array containing the raw Thomson spectrum.
	- `spectrum_notched`: An array containing spectrum convolved with the 
	instrument function and with the central notch removed.
	- `set_delay`: The programmed delay.
	- `real_delay`: An array containing the actual delay for the 5 shots, 
	calculated by comparing peaks of the photodiode readouts for the thomson 
	and heater beam.
	- `mean`: A tuple containing the predicted mean and its error for the 
	gaussian fit.
	- `std`: A tuple containing the predicted standard deviation and its 
	error for the gaussian fit.
	- `amp`: A tuple containing the predicted amplitude and its 
	error for the gaussian fit.
	- `T_e`: The predicted electron temperature from the fit in eV.
	- `n_e`: The predicted electron density in cm^-3.

	"""

	def __init__(self):
		pass

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

	@property
	def mean(self):
		return self._mean
	
	@mean.setter
	def mean(self, value: tuple):
		self._mean = value

	@property
	def std(self):
		return self._std
	
	@std.setter
	def std(self, value: tuple):
		self._std = value

	@property
	def amp(self):
		return self._amp
	
	@amp.setter
	def amp(self, value: tuple):
		self._amp = value

	@property
	def T_e(self):
		# Electron temperature in eV
		return np.abs(0.903 * self._std[0])

	@property
	def n_e(self):
		# Electron density in counts per cm^3
		return 5.6e16 * 1e-6 * self._amp[0] * 512 / 19




class TSAnalyzer():

	RAYLEIGH = 1

	
	def __init__(self, 
			  file: str | h5py._hl.files.File,
			  notch_low: float = 531, 
			  notch_high: float = 533,
			  instr_func_fwhm: float = 0.3):
		"""Initialize instance of Thomson scattering analysis class.
		Automatically loads the raw data from an h5 file, finds the signal
		and background image based on the Thomson and heater photodiodes, 
		calculates the spectrum based on those images, convolves the spectrum
		with a gaussian with a FWHM given by `inst_func_fwhm`, removes the
		notch, and fits a gaussian to the resulting data.

		Args:
			file (str | h5py._hl.files.File): Path to h5 file where the raw 
			data is stored or an instance of an h5py File containing the raw 
			data.
			notch_low (float, optional): Wavelength of the low side of the 
			notch. Defaults to 531.
			notch_high (float, optional): Wavelength of the high side of the
			notch. Defaults to 533.
			instr_func_fwhm (float, optional): FWHM of the instrument function.
			It is assumed that the instrument function is gaussian. Defaults 
			to 0.3.
		"""

		if type(file) == 'str':
			self._file = h5py.File(file)
		else:
			self._file = file

		ts = np.array(self._file['LeCroy:Ch2:Trace'])
		heater = np.array(self._file['LeCroy:Ch1:Trace'])
		times = np.array(self._file['LeCroy:Time'])
		delays = np.array(self._file['actionlist/TS:1w2wDelay'])
		images = [np.array(self._file[f'13PICAM1:Pva1:Image/image {n}']) 
			for n in range(100)]

		unique_delays = np.unique(delays)

		dt = np.round(np.average(times[:,1:] - times[:,:-1]), 10)
		ts_max_idx = np.argmax(np.gradient(ts, axis=1), axis=1)
		heater_max_idx = np.argmax(np.gradient(heater, axis=1), axis=1)
		real_delays = (ts_max_idx - heater_max_idx) * dt * 1e9
		
		ts_max = np.max(ts, axis=1)
		ts_shutter = np.array((ts_max > 2), dtype=int)
		heater_max = np.max(heater, axis=1)
		heater_shutter = np.array(
			[a != heater_max[i-1] for i, a in enumerate(heater_max)]
		).astype(int)

		shutter_delay_arr = np.array(
			[
				ts_shutter, 
				heater_shutter, 
				delays, 
				real_delays
			]
		).T
		info_arr = np.array(sorted(shutter_delay_arr, key=lambda x: x[2]))
		
		self.results = []
		
		for delay in unique_delays:

			delay_result = TSResult()

			fg_idxs = np.argwhere(np.all(info_arr[:,:3] == (1, 0, delay), 
								axis=1)).T[0]
			bg_idxs = np.argwhere(np.all(info_arr[:,:3] == (0, 0, delay), 
								axis=1)).T[0]
			fg_imgs = np.average([images[i] for i in fg_idxs], axis=0)
			bg_imgs = np.average([images[i] for i in bg_idxs], axis=0)

			shot_delay = info_arr[fg_idxs, 3]

			spectrum_full = np.sum((fg_imgs - bg_imgs) [215:375], axis=0)

			wspan = (np.max(self.wavelengths) - np.min(self.wavelengths)) / 2
			eval_w = np.linspace(-wspan, wspan, num=self.wavelengths.size)
			inst_func_arr = TSAnalyzer.gauss(
				x = eval_w,
				mu = 0,
				sigma = instr_func_fwhm / (2 * np.sqrt(2 * np.log(2))),
				amplitude = 1
			)
			inst_func_arr /= np.sum(inst_func_arr)

			spectrum_conv = np.convolve(
				spectrum_full, 
				inst_func_arr, 
				mode='same'
			)
			notched_spectrum = self.remove_notch(
				spectrum_conv, 
				notch_low, 
				notch_high
			)		

			# TO DO:
			# Impliment a more robust curve fitting routine using RANSAC
			# Investigate the usefullness of a supergaussian fit (i.e., 
			# the power p (usually 2), is also a fitable parameter)

			notched_wavelengths = self.remove_notch(
				self.wavelengths,
				notch_low,
				notch_high
			)

			mask = ~np.isnan(notched_spectrum)
			x = notched_wavelengths[mask]
			y = notched_spectrum[mask]
			sigma = np.sqrt(np.abs(y))  # Poisson noise estimate

			p0 = [x[np.argmax(y)], np.std(x), np.max(y)]
			popt, pcov = curve_fit(
			    self.gauss, x, y, sigma=sigma, p0=p0, absolute_sigma=True
			)
			mean, std_dev, amplitude = popt
			mean_err, std_dev_err, amplitude_err = np.sqrt(np.diag(pcov))

			delay_result.set_delay = delay
			delay_result.real_delay = shot_delay
			delay_result.spectrum_full = spectrum_full
			delay_result.spectrum_notched = notched_spectrum
			delay_result.mean = (mean, mean_err)
			delay_result.std = (np.abs(std_dev), std_dev_err)
			delay_result.amp = (amplitude, amplitude_err)

			self.results.append(delay_result)


	def plot_spectra(self,
				  title: str = None, 
				  show: bool = False, 
				  save: str = None):
	
		"""For a data set, graph the Thomson scattering spectrum along with
		the fit gaussian and predicted electron temperature/densities.

		Args:
			title (str, optional): title for the plot. Defaults to None.
			show (bool, optional): whether or not to show the image. 
			Defaults to False.
			save (str, optional): If passed, the path to save the spectrum 
			plots at. If none, then no plots are saved. Defaults to None.
		"""
		w = self.wavelengths
		fig = plt.figure(constrained_layout=True, figsize=(12, 8))
		fig.suptitle(title, fontsize=16)
		axs = fig.subplots(2, 3, sharex=True, sharey=True)
		for i, ax in enumerate(axs.flatten()):
			if i == 5:
				ax.set_visible(False)
			else:
				d = self.results[i]

				# Data plotting
				ax.plot(w, d.spectrum_full, color='b', alpha=0.25)
				ax.plot(w, 
						d.spectrum_notched, 
						color='b', 
						label=f'data convolved{'\n'}with inst. func.')
				fit = TSAnalyzer.gauss(w, d.mean[0], d.std[0], d.amp[0])
				ax.plot(w, fit, color='r', label='fit gaussian')

				configs = product(
					(d.mean[0]-d.mean[1], d.mean[0]+d.mean[1]),
					(d.std[0]-d.std[1], d.std[0]+d.std[1]),
					(d.amp[0]-d.amp[1], d.amp[0]+d.amp[1]),
				)
				fit_range = np.vstack([TSAnalyzer.gauss(w, *c) for c in configs])
				fit_low = np.min(fit_range, axis=0)
				fit_high = np.max(fit_range, axis=0)

				ax.fill_between(w, fit_low, fit_high, color='r', alpha=0.25)
				
				ax.legend(loc='upper right')

				# Fit reporting
				s = (f'$T_e = {d.T_e:.2f}$ eV\n'
		 			 f'$n_e$ = {d.n_e:.2e}cm$^{{{-3}}}$')
				ax.text(
					0.05,
					0.95, 
					s, 
					transform=ax.transAxes,
					va='top', 
					bbox=dict(edgecolor='k', facecolor='none'))



				# Graph formatting
				ax.set_title(f'$t_{{del}} = {d.set_delay}$ ns '
				 			 f'(actual = ${np.average(d.real_delay):.1f}±'
							 f'{np.std(d.real_delay):.1f}$ ns)')
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
	

	def remove_notch(self, arr: np.array, low, high) -> np.array:
		"""Remove the values corresponding to wavelengths between `low` and 
		`high` from `arr`."""
		idx_low = np.argmin(np.abs(self.wavelengths - low))
		idx_high = np.argmin(np.abs(self.wavelengths - high))
		output_arr = np.copy(arr)
		output_arr[idx_low:idx_high] = np.nan
		return output_arr

	@staticmethod
	def gauss(x, mu, sigma, amplitude) -> np.array:
		"""Gaussian curve over the data set `x` with parameters `mu`, 
		`sigma`, and `amplitude."""
		return (amplitude / np.sqrt(2 * np.pi * sigma ** 2) * 
		  np.exp( - np.square(x - mu) / (2 * sigma**2)))
	
	@property
	def wavelengths(self) -> np.array:
		"""Returns the wavelengths measured by our spectrometer"""
		return (np.arange(512) * 19.80636 / 511) + 522.918



if __name__ == '__main__':

	file = h5py.File('06-05/TS_CH_Location1_25V-2025-06-05.h5')
	data = TSAnalyzer(file)
	data.plot_spectra(show=True, title='CH, 25V, Location 1')





