import numpy as np
import h5py
import os

def _correct_time(ref, times, *arrs):
    ref = ref
    tot = ref.shape[1]
    ref_diff = np.gradient(ref, axis=1)
    diff_max_idx = np.argmax(ref_diff, axis=1)
    time_shifted = [row - row[diff_max_idx[i]] for i, row in enumerate(times)]
    mn, mx = min(diff_max_idx), max(diff_max_idx)
    new_l = [idx - mn for idx in diff_max_idx]
    new_h = [idx + tot - mx for idx in diff_max_idx]

    arrs_out = [
        np.array([row[new_l[i]: new_h[i]] for i, row in enumerate(ref)]),
        np.array([row[new_l[i]: new_h[i]] for i, row in enumerate(time_shifted)])
    ]
    for arr in arrs:
        arr = np.array([row[new_l[i]: new_h[i]] for i, row in enumerate(arr)])
        arrs_out.append(arr)
    return arrs_out

def _motor_to_tcc(v):
    """v = [x,y,z] is motor position vector in cm"""
    x = 3.2 - v[0]
    y = 3.73 - v[1]
    z = 1.1 - v[2]
    vec = np.round([x,y,z], 3)
    return vec

def _average_over_locations(motor_x,
                           motor_y,
                           motor_z,
                           data_dict):
    
    all_pos = np.vstack((
          motor_x,
          motor_y,
          motor_z
    )).T
    all_pos_TCC = np.apply_along_axis(motor_to_tcc, axis=1, arr=all_pos)
    unique_pos = np.unique(all_pos_TCC, axis=0)
   
    output_dict = {}
    for key, value in data_dict.items():
        temp = []
        for pos in unique_pos:
            pos_idx = []
            for i, p in enumerate(all_pos_TCC):
                if (pos == p).all():
                    pos_idx.append(i)
        
            temp.append(np.average(value[pos_idx], axis=0))
        output_dict[key] = temp       
    return output_dict, unique_pos




def pre_process(file: h5py._hl.files.File):


    ## Load relevent data
    motor_x = file['Motor4:PositionRead']
    motor_y = file['Motor5:PositionRead']
    motor_z = file['Motor6:PositionRead']

    p = file['LeCroy:Ch1:Trace']
    x = file['LeCroy:Ch4:Trace']
    y = file['LeCroy:Ch3:Trace']
    z = file['LeCroy:Ch2:Trace']
    t = file['LeCroy:Time']

    # e = file['PNG']


    pass


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
            raise ValueError(f'file must be str or h5py._hl.files.File, but file is {type(file)}')

        self.file = file

    def read_raw(self, readout_type):

        f = self.file
        raw_data = {}
        if readout_type in ('all', 'Motor'):
            raw_data['motor_x'] = np.array(f['Motor4:PositionRead'])
            raw_data['motor_y'] = np.array(f['Motor5:PositionRead'])
            raw_data['motor_z'] = np.array(f['Motor6:PositionRead'])
        if readout_type in ('all', 'Other'):
            raw_data['energy'] = np.array(f['PNGdigitizer:Ch2:Energy'])
            raw_data['pressure'] = np.array(f['Vacuum:DS:PressureCC'])
        return raw_data


class PreProcessBdot(PreProcessH5):

    def __init__(self, file):
        super().__init__(file)

        pos = self.read_raw('Motor')
        x = pos['motor_x']
        y = pos['motor_y']
        z = pos['motor_z']

        self.pos_unique = self.unique_positions(x, y, z)
        

    def read_raw(self, readout_type: str='all'):

        if readout_type not in ['all', 'LeCroy', 'MSO', 'Motor', 'Other']:
            raise ValueError
        raw_data = {}
        f = self.file

        if readout_type in ('all', 'Motor'):
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

        if readout_type in ('all', 'Other'):
            raw_data.update(super().read_raw(readout_type))

        return raw_data

    def _motor_to_tcc(self, motor_x, motor_y, motor_z):
        """takes motor readout positions, returns TCC pos in cm"""
        x = 3.4 - motor_x
        y = 3.6 - motor_y
        z = 1.35 - motor_z
        return x, y, z
    
    def _unique_positions(self, 
        motor_x: np.array, 
        motor_y: np.array, 
        motor_z: np.array) -> list:

        

        x, y, z = np.round(self._motor_to_tcc(motor_x, motor_y, motor_z), 2)
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





if __name__ == '__main__':
    path = '05-27/Probe2_0V-2025-05-27.h5'
    # file = h5py.File(path)

    data = PreProcessBdot(path).pos_unique
    for d in data:
        print(d)
