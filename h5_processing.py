import numpy as np
import h5py
import os
import inspect
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class PreProcessH5():

    def __init__(self, file):
        if type(file) is str:
            if not os.path.exists(file) or os.path.splitext(file)[1] != '.h5':
                raise ValueError(f'No .h5 file exists at {file}')
            else:
                file = h5py.File(file)
        elif type(file) is h5py._hl.files.File:
            pass
        else:
            raise ValueError(f'file must be str or h5py._hl.files.File, but '
                             f'file is {type(file)}')
        
        self._num_shots = len(file['epoch'])
        self.data = None
        self._file = file
        self.name = os.path.splitext(os.path.basename(file.filename))[0]
        self._flags = []

    def _motor_to_tcc(self, motor_x, motor_y, motor_z, offsets: np.array):
        """takes motor readout positions, returns TCC pos in cm"""
        x = offsets[0] - motor_x
        y = offsets[1] - motor_y
        z = offsets[2] - motor_z
        return x, y, z

    def read_raw(self) -> dict:
        """From the loaded h5 file (see __init__), reads the raw data for 
        motor x, y, and z positions, laser energy, and chamber pressure, 
        and packages that data in a dictionary.

        Returns:
            dict: dictionary with keys 'motor_x', 'motor_y', 'motor_z',
            'energy', and 'pressure' 
        """
        f = self._file
        raw_data = {
            'motor_x' : np.array(f['Motor4:PositionRead']),
            'motor_y' : np.array(f['Motor5:PositionRead']),
            'motor_z' : np.array(f['Motor6:PositionRead']),
            'energy' : np.array(f['PNGdigitizer:Ch2:Energy']),
            'pressure' : np.array(f['Vacuum:DS:PressureCC'])
        }
        return raw_data
    
    def _unique_positions(self, 
                          *arr, 
                          save_lineout_pos: bool = True,
                          ) -> list[np.array, list]:

        pos_vec = np.vstack((arr))
        unique_pos = np.unique(pos_vec, axis=1).T

        pos_and_idx_list = []
        for pos in unique_pos:
            idx = []
            for i, p in enumerate(pos_vec.T):
                if (pos == p).all():
                    idx.append(i)
            pos_and_idx_list.append([pos, idx])

        pos_sorted = sorted(pos_and_idx_list, key=lambda x :(np.min(x[1])))

        if save_lineout_pos:
            _x_0_pos =[[r[0], i] for i, r in enumerate(pos_sorted) if r[0][0] == 0]
            self.data['y_lineout'] = sorted(_x_0_pos, key=lambda x :(x[0][1]))

        self.data['unique_positions'] = pos_sorted
        return pos_sorted
    

    def avg_over_locations(self, 
                           keys_for_avg: list,
                           data_dict: dict | None = None, 
                           **kwargs):

        if data_dict is None:
            data_dict = self.data
        elif type(data_dict) is not dict:
            raise TypeError(f'data_dict must be a dict, but a'
                            f' {type(data_dict)} was passed')

        if 'offset' in kwargs.keys():
            offset = kwargs['offset']
        else:
            offset = self._offset

        try:
            pos_vec = self._motor_to_tcc(
                data_dict['motor_x'],
                data_dict['motor_y'],
                data_dict['motor_z'],
                offset
            )
            
            x, y, z = np.round(pos_vec, 2)
            data_dict['motor_x'] = x
            data_dict['motor_y'] = y
            data_dict['motor_z'] = z
        except:
            raise ValueError('data_dict must contain "motor_x", "motor_y",'
                             f' and "motor_z"')
        args = [data_dict[key] for key in keys_for_avg]
        pos_arr = self._unique_positions(*args)

        averaged_data = {
            'motor_x' : np.array([p[0][0] for p in pos_arr]),
            'motor_y' : np.array([p[0][1] for p in pos_arr]),
            'motor_z' : np.array([p[0][2] for p in pos_arr]),
            'y_lineout': self.data['y_lineout'],
            'unique_positions': self.data['unique_positions']
        }

        for key, value in data_dict.items():
            if key in ('motor_x', 'motor_y', 'motor_z', 
                       'y_lineout', 'unique_positions'):
                pass

            elif key in ('images', 'ts_timing', 'LeCroy_time', 'MSO_time'):
                avg_vals = []
                for pos in pos_arr:
                    idxs = pos[1]
                    val = np.average(value[idxs], axis=0)
                    avg_vals.append(val)
                averaged_data[key] = np.array(avg_vals)

            else:
                avg_vals = []
                mins = []
                maxes = []
                for pos in pos_arr:
                    idxs = pos[1]
                    val = np.average(value[idxs], axis=0)
                    max = np.max(value[idxs], axis=0)
                    min = np.min(value[idxs], axis=0)

                    avg_vals.append(val)
                    mins.append(min)
                    maxes.append(max)
                averaged_data[key] = np.array(avg_vals)
                averaged_data[f'{key}_mins'] = np.array(mins)
                averaged_data[f'{key}_maxes'] = np.array(maxes)

        
        self.data = averaged_data
        self._flags.append('averaged')

        return self.data
    

    def summary(self, data_type: str, verbose=False):
        """Method to pretty print a brief summary of h5 data"""
        d = self.data
        erg_avg = np.average(d['energy'])
        erg_std = np.std(d['energy'])

        pa_avg = np.average(d['pressure'])
        pa_std = np.std(d['pressure'])

        print(f'{'\n'}{'='*80}')
        print(f'Summary statistics for {self.name}')
        print(f'>{'-'*78}<')
        print('Motor statistics:')
        print(f'  -> Total number of shots: {self._num_shots}')
        if data_type == 'TS':
            print(f'  -> Total number of TS shot sets: {self._num_shots//4}')
        if 'averaged' in self._flags:
            print(f'  -> Number of unique positions: {len(d['motor_x'])}')
            if verbose:
                print('  -> Position list:')
                for i, row in enumerate(d['unique_positions']):
                    r = [float(e) for e in row[0]]
                    print(f'      * Pos {i:02d} at {r} has {len(row[1])} shots')
                print('  -> x=0 lineout positions:')
                for i, row in enumerate(d['y_lineout']):
                    r = [float(e) for e in row[0]]
                    print(f'      * Pos {i:02d} at {r} has indicies {row[1]}')
        print(f'>{'-'*78}<')
        print('Other statistics:')
        print(f'  -> Laser energy: {erg_avg:.3f}±{erg_std:.3f} J')
        print(f'  -> Chamber pressure: {pa_avg*1e3:.3f}±{pa_std*1e3:.3f} mTorr')
        if verbose:
            print(f'>{'-'*78}<')
            print('Pre processing flags:')
            print(f'  -> {self._flags}')
        print('='*80)

    @property
    def lineout_positions(self):
        return np.array([r[0] for r in self.data['y_lineout']])

    @property
    def lineout_idx(self):
        return ([r[1] for r in self.data['y_lineout']])
    
    @property
    def unique_positions(self):
        return self.data['unique_positions']
    
    @property
    def unique_pos_idx(self):
        return ([r[1] for r in self.data['unique_positions']])
    
    @property
    def energy(self):
        return self.data['energy']
    
    @property
    def pressure(self):
        return self.data['pressure']









class PreProcessBdot(PreProcessH5):

    def __init__(self, file, processing='default', **kwargs):
        super().__init__(file)
        
        self._offset= [3.4, 3.6, 1.35]
        self._timedelay = 35 * 1e-9    # given in seconds

        if processing == 'default':
            data = self.read_raw()
            kws = list(inspect.signature(self.align_lecroy).parameters.keys())
            args_for_align = {kw: kwargs[kw] for kw in kws & kwargs.keys()}
            data_aligned = self.align_lecroy(data, **args_for_align)
            data_averaged = self.avg_over_locations(data_aligned)
            self.data = data_averaged

        if processing == 'load_only':
            self.read_raw()                    
        

    def read_raw(self) -> dict:
        
        """Loads the raw data from the h5 file initialized in __init__.
        Only contains data from the motor positions, energy, pressure,
        LeCroy scope, and MSO scope. For the MSO scope, 'MSO_probe' 
        refers to the 1in bdot probe, if used. If not used, that channel 
        is just noise. 'MSO_voltage' refers to the bank voltage.

        Returns:
            dict: dictionary containing the keys in PreProcessH5.read_raw(),
            in addition to 'MSO_current', 'MSO_probe', 'MSO_voltage', 
            'MSO_time', 'LeCroy_x', 'LeCroy_y', 'LeCroy_z', 'LeCroy_time',
            and 'LeCroy_photodiode'.    
        """
        
        f = self._file
        raw_data = super().read_raw()

        raw_data['MSO_current'] = np.array(f['MSO24:Ch1:Trace'])
        raw_data['MSO_probe'] = np.array(f['MSO24:Ch4:Trace'])
        raw_data['MSO_voltage'] = np.array(f['MSO24:Ch2:Trace'])
        raw_data['MSO_time'] = np.array(f['MSO24:Time'])

        raw_data['LeCroy_x'] = np.array(f['LeCroy:Ch4:Trace'])
        raw_data['LeCroy_y'] = np.array(f['LeCroy:Ch3:Trace'])
        raw_data['LeCroy_z'] = np.array(f['LeCroy:Ch2:Trace'])
        raw_data['LeCroy_time'] = np.array(f['LeCroy:Time'])
        raw_data['LeCroy_photodiode'] = np.array(f['LeCroy:Ch1:Trace'])
        
        self.data = raw_data
        self._flags.append('raw_loaded')
        return raw_data

    def avg_over_locations(self, data_dict: dict| None = None):
        return super().avg_over_locations(keys_for_avg=['motor_x', 'motor_y', 
                                         'motor_z'], data_dict=data_dict, 
                                         offsets=self._offset)
    
    def align_lecroy(self, 
                     data_dict: dict | None = None,
                     attenuation: float = 0,
                     gain: float = 1) -> dict:
        
        if data_dict is None:
            data_dict = self.data

        if type(data_dict) is not dict:
            raise TypeError(f"data_dict must be a dictionary, but an object"
                            f"with type {type(data_dict)} was passed")
        
        g = gain * 10 ** (attenuation/20)
        
        ref = data_dict['LeCroy_photodiode']
        times = data_dict['LeCroy_time']
        tot = ref.shape[1]
        ref_diff = np.gradient(ref, axis=1)
        diff_max_idx = np.argmax(ref_diff, axis=1)
        t_shifted = [r - (r[diff_max_idx[i]] + self._timedelay) for i, r in enumerate(times)]
        mn, mx = min(diff_max_idx), max(diff_max_idx)
        l = [idx - mn for idx in diff_max_idx]
        h = [idx + tot - mx for idx in diff_max_idx]

        aligned_dict = data_dict.copy()
        t_align = [row[l[i]: h[i]] for i, row in enumerate(t_shifted)]
        aligned_dict['LeCroy_time'] = np.array(t_align)
        for key in ('LeCroy_x', 'LeCroy_y', 'LeCroy_z', 'LeCroy_photodiode'):
            val_align = [r[l[i]:h[i]] for i, r in enumerate(data_dict[key])]
            val_align_gain = np.array(val_align) * g
            aligned_dict[key] = np.array(val_align_gain)
        
        self.data = aligned_dict

        return aligned_dict
    
    def summary(self, verbose=False):
        super().summary('Bdot', verbose)
            
    
    @property
    def LeCroy(self):
        """Returns LeCroy x, y, z, and time as a tuple"""
        x = self.data['LeCroy_x']
        y = self.data['LeCroy_y']
        z = self.data['LeCroy_z']
        t = self.data['LeCroy_time']
        return x, y, z, t
    
    @property
    def LeCroy_lineout(self):
        """
        Returns LeCroy x, y, z, and time ordered along the lineout
        axis from low y to high y
        """
        idx = [i for i in self.lineout_idx]
        x = self.data['LeCroy_x'][idx]
        y = self.data['LeCroy_y'][idx]
        z = self.data['LeCroy_z'][idx]
        t = self.data['LeCroy_time'][idx]
        return x, y, z, t

    @property
    def MSO(self) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Returns MSO current, bank voltage, probe readout, and time as a tuple
        """
        curr = self.data['MSO_current']
        time = self.data['MSO_time']
        probe = self.data['MSO_probe']
        voltage = self.data['MSO_voltage']
        return curr, voltage, probe, time
    
    def bounds(self, key: str):
        d = self.data
        return d[key], d[f'{key}_mins'], d[f'{key}_maxes']
    








class PreProcessTS(PreProcessH5):
    
    def __init__(self, file, spectrums: bool = False):
        super().__init__(file)

        self._offset = [0, 0, 0]
        self._num_shots = len(self._file['epoch'])
        self.read_raw()
        self.remove_background()
        self.avg_over_locations()
        if spectrums:
            self.spectrums()


    def read_raw(self) -> dict:
        
        """Loads the raw data from the h5 file initialized in __init__.
        Only contains data from the motor positions, energy, pressure, 
        spectra (PICAM1), and thomson scattering delay.

        Returns:
            dict: dictionary containing the keys/values in PreProcessH5.read_raw(),
            in addition to 
                - 'images': dict (contains every individual spectrum. Of the form 
                        'image i' : 512x511 pixel array where the 512 columns 
                        correspond to a different frequency 

                - 'ts_timing': array (delay between heater beam and TS beam 
                        in nano seconds)
        """
        
        file = self._file
        n = self._num_shots
        raw_data = super().read_raw()

        raw_data['images'] = {f'{n}':np.array(file[f'13PICAM1:Pva1:Image/image {n}']) for n in range(n)}
        raw_data['ts_timing'] = file['actionlist/TS:1w2wDelay']

        self.data = raw_data
        self._flags.append('raw_loaded')
        return raw_data
    
    def remove_background(self, data_dict: dict | None = None):
    
        if data_dict is None:
            data_dict = self.data
        
        new_data = {}
        for key, val in data_dict.items():
            if key == 'images':
                images = [
                    np.subtract(np.array(val[f'{4*i+1}']), 
                                np.array(val[f'{4*i+3}']), dtype=float) 
                    for i in range(self._num_shots//4)
                ]

                new_data[key] = np.array(images)
            else:
                new_data[key] = val[1::4]
        self._flags.append('background_removed')
        self.data = new_data
        return new_data
    
    def avg_over_locations(self, data_dict : dict | None = None):
        return super().avg_over_locations(
            keys_for_avg=['motor_x', 'motor_y', 'motor_z', 'ts_timing'],
            data_dict=data_dict
        )
    
    def spectrums(self, 
                  bin_width: int=1
                  ) -> np.array:
        if 'background_removed' not in self._flags:
            raise LookupError('background must be removed before ' \
                'constructing spectrums')
        if bin_width not in (1, 2, 4, 8):
            raise ValueError(f'Bins cannot have width {bin_width},'
                             f'it must be 1, 2, 4, or 8')

        num = len(self.unique_positions)
        spectrums = np.array([np.sum(self.data['images'][n, 215:375,:], axis=0) for n in range(num)])
        binned_spectrums = [
            np.add.reduceat(spectra, np.arange(0, 512, bin_width))
            for spectra in spectrums
        ]

        self.data['spectrums'] = binned_spectrums
        self.data['wavelengths'] = PreProcessTS.wavelengths(bin_width)

        return binned_spectrums
            
    def summary(self, verbose=False):
        super().summary(data_type='TS', verbose=verbose)


    @staticmethod
    def wavelengths(bin_width: int=1) -> np.array:
        """Wavelength range captured in TS spectrum"""
        if bin_width not in (1, 2, 4, 8):
            raise ValueError(f'Bins cannot have width {bin_width},'
                             f'it must be 1, 2, 4, or 8')
        
        num_vals = 512 / bin_width
        return (np.arange(num_vals) * bin_width * 19.80636 / 511) + 522.918
    
    @staticmethod
    def bin_to_wavelength(bin):
        """Converts bin (pixel) to wavelength (nm)"""
        return bin * 0.03876




if __name__ == '__main__':
    loader = PreProcessTS('06-02/TS_lineout_500V-2025-06-02.h5')
    images = loader.data['images']
    spectrums = loader.spectrums()
    wavelengths = loader.data['wavelengths']

    print(len(images))

    fig = plt.figure(figsize=(9, 54), constrained_layout=True)
    axs = fig.subplots(18, 3)

    for im, ax, spectra in zip(images, axs.flatten(), spectrums):
        ax.matshow(im)
        ax1 = ax.twinx()
        ax1.plot(spectra)
        ax.axhline(215, color='red')
        ax.axhline(375, color='red')

    plt.savefig('bar.pdf')



