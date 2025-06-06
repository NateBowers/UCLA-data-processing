import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
				bg_idxs = np.argwhere(np.all(info_arr[:,:3] == (0, 0, delay), axis=1))
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

		# TO DO
		# Add code to fit (either with a gaussian or plasmapy)
		# Add method to filter out extreme outliers (akin to a convolution)
		# Add method to downsample/filter data to smooth





	def plot_spectra(self):

		nrows = self.num_shots // 3 + 1
		fig = plt.figure(figsize=(12, 2 + nrows*3), constrained_layout=True)
		axs = fig.subplots(nrows=nrows, ncols=3, sharex=True, sharey=True)
		for i, ax in enumerate(axs.flatten()):
			ax.plot(self.wavelengths, self.spectra[i])



		pass

	def plot_shot_timing():
		pass





	@staticmethod
	def gauss(x, mu, sigma, alpha):
		return (alpha * 1 / np.sqrt(2 * np.pi * sigma**2) * 
					np.exp( - (x-mu) **2 / (2 * sigma**2)))




if __name__ == '__main__':
	file = (
		'06-06/TS_Cu_Location2_400V-2025-06-06.h5',
		'06-06/TS_Cu_Location2_400V_rerun-2025-06-06.h5',
		'06-06/TS_Cu_Location2_400V_rerun2-2025-06-06.h5',
	)
	data = TSAnalyzer(file, position=1, voltage=400, material='Cu')
	data.plot_spectra()


	# print(foo[0:15])

