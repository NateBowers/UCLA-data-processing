import numpy as np
import h5py
import os


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
        self.data = None
        self.file = file
        self.name = os.path.splitext(os.path.basename(file.filename))[0]
        self.flags = []

    def _motor_to_tcc(self, motor_x, motor_y, motor_z, offsets: np.array):
        """takes motor readout positions, returns TCC pos in cm"""
        x = offsets[0] - motor_x
        y = offsets[1] - motor_y
        z = offsets[2] - motor_z
        return x, y, z

    def read_raw(self, readout_type):

        f = self.file
        raw_data = {}
        if readout_type in ('all', 'Motor'):
            raw_data['motor_x'] = np.array(f['Motor4:PositionRead'])
            raw_data['motor_y'] = np.array(f['Motor5:PositionRead'])
            raw_data['motor_z'] = np.array(f['Motor6:PositionRead'])
        if readout_type in ('all', 'Stats'):
            raw_data['energy'] = np.array(f['PNGdigitizer:Ch2:Energy'])
            raw_data['pressure'] = np.array(f['Vacuum:DS:PressureCC'])
        return raw_data
    
    
    def _unique_positions(self, 
        motor_x: np.array, 
        motor_y: np.array, 
        motor_z: np.array,
        **kwargs
        ) -> list:
        """
        Find the unique motor positions, and return those positions with the 
        indicies that they appear at
        """
        self.total_pos = len(motor_x)
        corr_pos = self._motor_to_tcc(motor_x, motor_y, motor_z, **kwargs)
        x, y, z = np.round(corr_pos, 2)
        pos_vec = np.vstack((x, y, z))
        unique_pos = np.unique(pos_vec, axis=1).T

        pos_and_idx_list = []
        for pos in unique_pos:
            idx = []
            for i, p in enumerate(pos_vec.T):
                if (pos == p).all():
                    idx.append(i)
            pos_and_idx_list.append([pos, idx])

        pos_sorted = []
        for i in range(len(pos_and_idx_list)):
            for e in pos_and_idx_list:
                if (i*3) in e[1]:
                    pos_sorted.append(e)
    
        return pos_sorted
    
    def avg_over_locations(self, 
                           data_dict: dict | None = None, 
                           **kwargs):

        if data_dict is None:
            data_dict = self.read_raw('all')
        elif type(data_dict) is not dict:
            raise TypeError(f'data_dict must be a dict, but a'
                            f' {type(data_dict)} was passed')

        try:
            x = data_dict['motor_x']
            y = data_dict['motor_y']
            z = data_dict['motor_z']
        except:
            raise ValueError('data_dict must contain "motor_x", "motor_y",'
                             f' and "motor_z"')

        pos_arr = self._unique_positions(x, y, z, **kwargs)
        motor_x = [p[0][0] for p in pos_arr]
        motor_y = [p[0][1] for p in pos_arr]
        motor_z = [p[0][2] for p in pos_arr]

        avg_dict = {}
        for key, value in data_dict.items():
            if key in ('motor_x', 'motor_y', 'motor_z'):
                avg_dict[key] = locals()[key]
            else:
                avg_vals = []
                for pos in pos_arr:
                    idxs = pos[1]
                    if np.ndim(value) == 2:
                        val = np.average(value[idxs], axis=0)
                    elif np.ndim(value) == 1:
                        val = np.average(value[idxs])
                    avg_vals.append(val)
                avg_dict[key] = np.array(avg_vals)
        
        self.data = avg_dict
        self.flags.append('averaged')

        return avg_dict


class PreProcessBdot(PreProcessH5):

    def __init__(self, file, processing='default'):
        super().__init__(file)
        
        self.offsets = [3.4, 3.6, 1.35]

        if processing == None:
            pass
        if processing == 'default':
            _data = self.avg_over_locations()
            data = self.align_lecroy(_data)
        

    def read_raw(self, readout_type: str='all'):

        if readout_type not in ['all', 'LeCroy', 'MSO', 'Motor', 'Stats']:
            raise ValueError
        
        raw_data = {}
        f = self.file
        self.flags.append(readout_type)

        if readout_type in ('all', 'Motor', 'Stats'):
            raw_data.update(super().read_raw(readout_type))

        if readout_type in ('all', 'MSO'):
            raw_data['MSO_current'] = np.array(f['MSO24:Ch1:Trace'])
            raw_data['MSO_probe'] = np.array(f['MSO24:Ch4:Trace'])
            raw_data['MSO_voltage'] = np.array(f['MSO24:Ch2:Trace'])
            raw_data['MSO_time'] = np.array(f['MSO24:Time'])

        if readout_type in ('all', 'LeCroy'):
            raw_data['LeCroy_x'] = np.array(f['LeCroy:Ch4:Trace'])
            raw_data['LeCroy_y'] = np.array(f['LeCroy:Ch3:Trace'])
            raw_data['LeCroy_z'] = np.array(f['LeCroy:Ch2:Trace'])
            raw_data['LeCroy_time'] = np.array(f['LeCroy:Time'])
            raw_data['LeCroy_photodiode'] = np.array(f['LeCroy:Ch1:Trace'])
        
        self.data = raw_data
        self.flags.append('raw_loaded')
        return raw_data
    

    def motor_to_tcc(self, motor_x, motor_y, motor_z):
        return super()._motor_to_tcc(motor_x, motor_y, motor_z, self.offsets)

    def avg_over_locations(self, data_dict: dict| None = None):
        self.flags.append('avg')
        return super().avg_over_locations(data_dict, offsets=self.offsets)
    
    def align_lecroy(self, data_dict):
    
        if type(data_dict) is not dict:
            raise TypeError(f"data_dict must be a dictionary, but an object"
                            f"with type {type(data_dict)} was passed")
        elif 'all' not in self.flags and 'lecroy' not in self.flags:
            raise ValueError('LeCroy must be loaded before alinging')
        
        ref = data_dict['LeCroy_photodiode']
        times = data_dict['LeCroy_time']
        tot = ref.shape[1]
        ref_diff = np.gradient(ref, axis=1)
        diff_max_idx = np.argmax(ref_diff, axis=1)
        t_shifted = [row - row[diff_max_idx[i]] for i, row in enumerate(times)]
        mn, mx = min(diff_max_idx), max(diff_max_idx)
        l = [idx - mn for idx in diff_max_idx]
        h = [idx + tot - mx for idx in diff_max_idx]

        aligned_dict = data_dict.copy()
        t_align = [row[l[i]: h[i]] for i, row in enumerate(t_shifted)]
        aligned_dict['LeCroy_time'] = np.array(t_align)
        for key in ('LeCroy_x', 'LeCroy_y', 'LeCroy_z', 'LeCroy_photodiode'):
            val_align = [row[l[i]:h[i]] for i, row in enumerate(data_dict[key])]
            aligned_dict[key] = np.array(val_align)
        
        self.data = aligned_dict

        return aligned_dict
            
    def summary(self, verbose=False):
        
        d = self.data
        erg_avg = np.average(d['energy'])
        erg_std = np.std(d['energy'])

        pa_avg = np.average(d['pressure'])
        pa_std = np.std(d['pressure'])

        print(f'{'\n'}{'='*80}')
        print(f'Summary statistics for {self.name}')
        print(f'>{'-'*78}<')
        print('Motor statistics:')
        print(f'  -> Total number of positions: {self.total_pos}')
        if 'avg' in self.flags:
            print(f'  -> Number of unique positions: {len(d['motor_x'])}')
        print(f'>{'-'*78}<')
        print('Other statistics:')
        print(f'  -> Laser energy: {erg_avg:.3f}±{erg_std:.3f} J')
        print(f'  -> Chamber pressure: {pa_avg*1e3:.3f}±{pa_std*1e3:.3f} mTorr')
        if verbose:
            print(f'>{'-'*78}<')
            print('Pre processing done:')
            print(f'  -> Laser energy: {erg_avg:.3f}±{erg_std:.3f} J')
            print(f'  -> Chamber pressure: {pa_avg*1e3:.3f}±{pa_std*1e3:.3f} mTorr')
        print('='*80)
    
    

class PreProcessTS(PreProcessH5):
    def __init__(self):
        raise NotImplementedError



if __name__ == '__main__':
    file = h5py.File('05-27/Probe2_100V-2025-05-27.h5')
    data = PreProcessBdot(file)
    data.summary()