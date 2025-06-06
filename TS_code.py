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
			  ):
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
		
		self._file = f
		self._position = position
		self._voltage = voltage
		self._material = material
		if date is None:
			self._date = datetime.date.today().strftime("%m-%d")
		else:
			self._date = date

		self._name = f'TS_{material}_loc{position}_{voltage}V-{self._date}'


	def loader(self):

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

		self._dt = np.average(times[1::2]-times[:-1:2])
		unique_delays = np.unique(delays)
		ts_max = np.max(ts, axis=1)
		heater_max = np.max(heater, axis=1)
		heater_shutter = np.array([a != heater_max[i-1] for i, a in enumerate(heater_max)]).astype(int)
		ts_shutter = np.array((ts_max > 2), dtype=int)
		real_delays = (np.argmax(ts, axis=1) - np.argmax(heater, axis=1)) * self._dt * 1e9

		shutter_delay_arr = np.array([
			ts_shutter,
			heater_shutter,
			delays,
			real_delays
		]).T

		info_arr = np.array(sorted(shutter_delay_arr, 
						key=lambda x: x[2]))

		for delay in unique_delays:
			fg_idxs = np.argwhere(np.all(info_arr[:3] == (1, 1, delay), axis=1))
			bg_idxs = np.argwhere(np.all(info_arr[:3] == (1, 1, delay), axis=1))

			pass







		return info_arr

		# load LeCroy for heater, ts, times
		# load delays
		# load images



		



if __name__ == '__main__':
	file = '06-05/TS_Cu_Location1_400V_redo-2025-06-05.h5'
	file = (
		'06-06/TS_Cu_Location2_400V-2025-06-06.h5',
		'06-06/TS_Cu_Location2_400V_rerun-2025-06-06.h5',
		'06-06/TS_Cu_Location2_400V_rerun2-2025-06-06.h5',
	)
	data = TSAnalyzer(file, position=1, voltage=400, material='Cu')

	foo = data.loader()
	print(foo[0:15])

