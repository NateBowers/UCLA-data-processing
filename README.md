# UCLA-data


This code was written and tested in python 3.13.1. See requirements.txt for the packages and the versions used.

I *highly* recommend using a virtual enviornment for running this code. I like using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) and [pyenv](https://github.com/pyenv/pyenv), but [venv](https://docs.python.org/3/library/venv.html) is easier to setup and use.


# Capabilities

## HDF5 pre-processing

### Code

The file `h5_processing.py` contains classes for preforming some initial processing on .h5 files collected from the laser control system. The PreProcessing classes automatically average data over positions and align LeCroy readouts in time. Pass `processing='load_only` to just load the raw data into the object `processing=None` to simply initialize it.

### Sample use

Call `PreProcessBdot` either on an already loaded h5py instance or the path to an .h5 file. It will automatically collect information on LeCroy, MSO, motor positions, laser energy, and chamber pressure. It will then automatically average over positions and align LeCroy in time. Calling `LeCroy`, `MSO`, `energy`, or `pressure` gives the relevant data (the type hints should make clear what the output is). In addition, `LeCroy_lineout` gives the LeCroy data only for the y-axis lineout. `unique_positions` gives a series an array of tupes for each position containing the TCC coordinates and indexes of the shots which were taken at that point, and `unique_pos_idx` gives just the indexes of each unique position. Finally, `lineout_positions` gives the positions of the y-lineout and `lineout_idx` gives the index of the lineout positions *in the averaged data*.

The following snipped loads in LeCroy data that is aligned and averaged:
```
file = h5py.File('path/to_file.h5)
loader = PreProcessBdot(file)
x, y, z, t = loader.LeCroy
```

To graph the average x probe reading at the closest point on the y-lineout (typically (0, 0.9, 0.85)), we can run the following code:
```
x_line, _, _, t_line = loader.LeCroy_lineout
x_pos0 = x_line[0]
t_pos0 = t_line[0]
plt.plot(t_pos0, x_pos0)
plt.show()
```

Finally, calling `loader.summary()` gives prints the following output with information on the h5 file. Passing the argument `verbose=True` prints substantial additional information on the positions
```
================================================================================
Summary statistics for to_file
>------------------------------------------------------------------------------<
Motor statistics:
  -> Total number of positions: 57
  -> Number of unique positions: 18
>------------------------------------------------------------------------------<
Other statistics:
  -> Laser energy: 9.368±0.079 J
  -> Chamber pressure: 96.839±1.203 mTorr
================================================================================
```


## BDot probe calibration

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

# To do

- [x] improve documentation for bdot_code
- [ ] add auto detection for voltage array sizes for field reconstruction in bdot_code
- [ ] add shell script to automatically process data
- [ ] improve documentation for h5_processing
- [ ] add TS specific aspects to h5_processing
- [ ] add general graphing capabilities (unsure what they would look like)
- [x] add instructions for h5_processing to README


# Contact
Nathaniel Bowers - bowena02@gettysburg.edu

