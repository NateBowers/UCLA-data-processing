General overview of the Thomson scattering processing code:

1. `ts_paths.csv` lists where each of the h5 files are locally on my (Nate's) laptop, along with the location, material, and voltage
2. `ts_save_raw_as_tiff.py` reads through all the h5 files in `ts_paths.csv` and exports the the CH data. Sorts foreground from background, calculates the actual delay, saves the images as .tiff files, and generates two .csv files (foreground and background, respectively) which have all the relevant information on the shots.
3. `ts_neon_calibration.py` calibrates the PICAM spectrum to a known spectrum from a Ne lamp. Fits 4 Gaussians to the four primary peaks.
4. `ts_model.py` is the primary function for calculating the temperature and density from a series of images. Uses `plasmapy` for the Thomson response. 
5. `ts_system_sigma.py` calculates the PICAM noise.
