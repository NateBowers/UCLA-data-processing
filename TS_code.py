import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import convolve
from skimage.measure import block_reduce
import os
import datetime
import sys

class TSAnalyzer():

	RAYLEIGH = 1
	
	def __init__(self,
			  file: str | h5py._hl.files.File | tuple[str, ...],
			  position: int,
			  voltage: int,
			  material: str,
			  date: str | None = None,
			  notch: None | list[int, int] = [531, 533]
			  ):
		
		# Ensure the arguments passed are right, and load the h5 files
		if type(file) is str:
			if not os.path.exists(file) or os.path.splitext(file)[1] != '.h5':
				raise ValueError(f'No .h5 file exists at {file}')
			else:
				f = h5py.File(file)
		elif type(file) is h5py._hl.files.File:
			f = file
		elif type(file) is tuple:
			f = [h5py.File(e) for e in file]

		else:
			raise ValueError(f'file must be str or h5py._hl.files.File, but '
							 f'file is {type(file)}')
		
		# Set some instance variables 
		self._file = f
		self._position = position
		self._voltage = voltage
		self._material = material

		if date is None:
			self._date = datetime.date.today().strftime("%m-%d")
		else:
			self._date = date
		self._name = f'TS_{material}_loc{position}_{voltage}V-{self._date}'

		# Load data from h5 files. If multiple files have been passed, they're
		# all stacked to have the height be 100 * number of files.
		if type(self._file) == list:
			ts = np.vstack([file['LeCroy:Ch2:Trace'] for file in self._file])
			heater = np.vstack([file['LeCroy:Ch1:Trace'] for file in self._file])
			times = np.vstack([file['LeCroy:Time'] for file in self._file])[0]
			delays = np.hstack([file['actionlist/TS:1w2wDelay'] for file in self._file])
			images = np.vstack([[file[f'13PICAM1:Pva1:Image/image {n}'] for n in range(100)] for file in self._file], dtype=np.int16)
		else:
			ts = file['LeCroy:Ch2:Trace']
			heater = file['LeCroy:Ch1:Trace']
			times = file['LeCroy:Time']
			delays = file['actionlist/TS:1w2wDelay']
			images = [file[f'13PICAM1:Pva1:Image/image {n}'] for n in range(100)]

		unique_delays = np.unique(delays)

		# Calculate the 'real' delays by comparing the maximum of the 
		# derivative of the photodiode readout for the thomson and heater 
		# beams and converting from index differences to nanoseconds with dt.
		self._dt = np.average(times[1::2]-times[:-1:2])
		real_delays = ((np.argmax(np.gradient(ts, axis=1), axis=1) 
				 - np.argmax(np.gradient(heater, axis=1), axis=1)) 
				 * self._dt * 1e9)
		
		# Finds which shots the TS beam fires by seeing if there's a value
		# above 2.
		ts_max = np.max(ts, axis=1)
		ts_shutter = np.array((ts_max > 2), dtype=int)

		# Finds the shots the heater beam is fired by seeing if the readout
		# is different from the previous shot. This works because the LeCroy
		# scope gives the same data if nothing new is recorded.
		heater_max = np.max(heater, axis=1)
		heater_shutter = np.array([a != heater_max[i-1] for i, a in enumerate(heater_max)]).astype(int)


		# Package the shutters and delays, and sort by the programmed delay
		shutter_delay_arr = np.array([
			ts_shutter,
			heater_shutter,
			delays,
			real_delays
		]).T
		info_arr = np.array(sorted(shutter_delay_arr, 
						key=lambda x: x[2]))
		
		# For each delay, calculate the spectrum by finding the foreground
		# images (TS and heater), the background images (no TS but heater),
		# averaging and taking the difference. Also calculate the average,
		# std of the real delays. Also records the image
		imgs = []
		spectra = []
		real_delays = []

		for delay in unique_delays:
			try:
				fg_idxs = np.argwhere(np.all(info_arr[:,:3] == (1, 0, delay), axis=1))
				bg_idxs = np.argwhere(np.all(info_arr[:,:3] == (0, 1, delay), axis=1))
				im_fg = np.average(images[fg_idxs[0]], axis=0)
				im_bg = np.average(images[bg_idxs[0]], axis=0)
				im = im_fg - im_bg
				spectrum = np.sum(im[215:375], axis=0)

				real_delay = info_arr[fg_idxs, 3]
				real_delays.append([np.average(real_delay),
						np.std(real_delay)])

			except IndexError:
				# TO DO
				# Impliment a fall back procedure if the ts blip is not 
				# recorded in LeCroy because its too far out in time
				pass 
				real_delays.append([delay, 0])
			spectra.append(spectrum)
			imgs.append(im)

		# Store several useful things to the class instance
		self.wavelengths = (np.arange(512) * 19.80636 / 511) + 522.918
		self.spectra = np.array(spectra)
		self.images = imgs
		self.real_delays = real_delays
		self.programmed_delays = unique_delays
		self.num_shots = len(unique_delays)


		#############################################
		#                 WARNINGS:                 #
		#                                           #
		# NOTCH IS NOT CORRECTED FOR BEFORE FITTING #
		#     ASSUMES NON-COLLECTIVE SCATTERING     #
		#                                           #
		#############################################


		means = []
		st_devs = []
		scales = []
		for i in range(self.num_shots):
			popt, _ = curve_fit(self.gauss, 
						self.wavelengths, 
						self.spectra[i],
						p0 = [532, 5, 1000])
			
			mean, st_dev, scale = popt
			means.append(mean)
			st_devs.append(st_dev)
			scales.append(scale)

		self.means = np.array(means)
		self.st_devs = np.array(st_devs)
		self.scales = np.array(scales)

		self.density = 5.6e16 * 1e-6 * self.scales * 512 / 19
		self.temp = np.abs(0.903 * self.st_devs)



		# TO DO
		# Add method to filter out extreme outliers (akin to a convolution)
		# Add method to downsample/filter data to smooth





	def plot_spectra(self):

		w = self.wavelengths

		nrows = self.num_shots // 3 + 1
		fig = plt.figure(figsize=(15, 2 + nrows*3))
		fig.suptitle(self._name)
		axs = fig.subplots(nrows=nrows, ncols=3, sharex=True, sharey=True)
		for i, ax in enumerate(axs.flatten()):
			if i >= self.num_shots:
				ax.set_visible(False)
				pass
			else:
				spectrum = self.spectra[i]
				fit_stats = [self.means[i], self.st_devs[i], self.scales[i]]
				fit = TSAnalyzer.gauss(w, *fit_stats)
				ax.plot(w, spectrum, label='data')
				ax.plot(w, fit, color='red', label='fitted curve')

				w_red = block_reduce(w, block_size=2, func=np.mean, cval=np.mean(w))
				a = block_reduce(spectrum, block_size=2, func=np.mean, cval=np.mean(spectrum))
				b = convolve(a, [0.1, 0.2, 0.4, 0.2, 0.1], 'same')
				ax.plot(w_red, b, color='black', label='downsampled data')
				s = f'mean = {fit_stats[0]:.2f}{'\n'}std dev = {fit_stats[1]:.3f}{'\n'}scale = {fit_stats[2]:.1f}{'\n'}'
				ax.text(0.05,0.95, s, fontsize='small', transform=ax.transAxes, va='top')

				s = f'n_e = {self.density[i]:.2g} cm$^{{-3}}${'\n'}T_e = {self.temp[i]:.3f} eV'
				ax.text(0.05,0.75, s, fontsize='small', transform=ax.transAxes, va='top')

				ax.legend()


		plt.show()
		pass

	def plot_shot_timing():
		pass





	@staticmethod
	def gauss(x, mu, sigma, alpha):
		return (alpha / np.sqrt(2 * np.pi * sigma ** 2) * 
		  np.exp( - np.square(x - mu) / (2 * sigma**2)))




if __name__ == '__main__':
	file = (
		'06-06/TS_Al_Location3_0V-2025-06-06.h5',
		# '06-06/TS_Cu_Location4_0V_rerun-2025-06-06.h5',
	)

	# foo = TSAnalyzer.gauss(np.array([1,2,3]), 1, 1, 1)

	data = TSAnalyzer(file, position=1, voltage=400, material='Cu')
	data.plot_spectra()


	# print(foo[0:15])

