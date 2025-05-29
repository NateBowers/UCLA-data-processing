# UCLA-data


This code was written and tested in python 3.13.1. See requirements.txt for the packages and the versions used.

I *highly* recommend using a virtual enviornment for running this code. I like using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) and [pyenv](https://github.com/pyenv/pyenv), but [venv](https://docs.python.org/3/library/venv.html) is easier to setup and use.

Contents:
1. Capabilities
   1.1 HDF5 Pre-processing
   1.2 BDOT probe calibration
2. To-do
3. Contact

# 1. Capabilities

## 1.1 HDF5 pre-processing

### Code

The file `h5_processing.py` contains classes for preforming some initial processing on .h5 files collected from 

## 1.2 BDot probe calibration

### Code

The file `bdot_code.py` contains two classes which contain all the useful functionality for an individual bdot probe including calibrating, reconstructing fields, generating reports, and loading saved calibration parameters.

The folder `bdot_data/` contains the raw data used for calibration, the setup json files, and the already calibrated parameters.

### Sample use

**Probe calibration:**
```
from bdot_code import ThreeAxisProbe

probe = ThreeAxisProbe(5, 'name')
probe.load_data('path_to_folder_for_probe_5_calibration_data')

probe.clip(50)

probe.calibrate(save=True, overwrite=True)
probe.gen_probe_report()
```

*Explanation*:
When calling ThreeAxisProbe, you need to include the probe number, and, optionally, the name. The calibration data is expected to be nine files titles PjBi.TXT for i, j in (X,Y,Z) where the first 15 rows are header and, in order from left to right, columns for frequency (Hz), magnitude (unitless), and phase (deg). The first 50 data points are clipped because the phase signal tends to be extra noisy there. When calibrating, there setting `overwrite=True` allows any old data to be updated.

**Field Reconstruction**
```
from bdot_code import ThreeAxisProbe

probe = ThreeAxisProbe(2)
probe.load_params('bdot_data/params/probe_2.json)
field_vec = probe.reconstruct_field(probe_x, probe_y, probe_z, times, gain)

x_field, y_field, z_field = field_vec
```

# 2. To do

- [ ] improve documentation for bdot_code
- [ ] add auto detection for voltage array sizes for field reconstruction in bdot_code
- [ ] add shell script to automatically process data
- [ ] improve documentation for h5_processing
- [ ] add TS specific aspects to h5_processing
- [ ] add general graphing capabilities (unsure what they would look like)
- [ ] add instructions for h5_processing to README


# 3. Contact
Nathaniel Bowers - bowena02@gettysburg.edu

