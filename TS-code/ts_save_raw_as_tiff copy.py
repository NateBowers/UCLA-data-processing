import numpy as np
import h5py
import pandas as pd
import os
from PIL import Image
from PIL.ExifTags import TAGS
import csv
import shutil

MATERIAL = 'CH'


def process_h5(data):
    # calculates real delay
    # returns array of foreground and background information
    # each represented as [image, real_delay, programmed_delay]
    # For background, real_delay = None

    ts = np.array(data['LeCroy:Ch2:Trace'])
    heater = np.array(data['LeCroy:Ch1:Trace'])
    times = np.array(data['LeCroy:Time'])
    delays = np.array(data['actionlist/TS:1w2wDelay'])
    images = [np.array(data[f'13PICAM1:Pva1:Image/image {n}'], dtype=np.int32) for n in range(len(data['13PICAM1:Pva1:Image']))]

    dt = np.round(np.average(times[:,1:] - times[:,:-1]), 10)
    ts_max_idx = np.argmax(np.gradient(ts, axis=1), axis=1)
    heater_max_idx = np.argmax(np.gradient(heater, axis=1), axis=1)
    real_delays = (ts_max_idx - heater_max_idx) * dt * 1e9 - 10.1

    ts_max = np.max(ts, axis=1)
    ts_shutter = np.array((ts_max > 2), dtype=int)
    heater_max = np.max(heater, axis=1)
    heater_shutter = np.array([a != heater_max[i-1] for i, a in enumerate(heater_max)]).astype(int)

    fg_arr = []
    bg_arr = []

    for i in range(len(images)):
        if ts_shutter[i] == 1 and heater_shutter[i] == 1:
            fg_arr.append([images[i], real_delays[i], delays[i]])
        if ts_shutter[i] == 0 and heater_shutter[i] == 1:
            bg_arr.append([images[i], 0, delays[i]])
    return fg_arr, bg_arr


dir = 'ts_tiff_img_CH'
subdirs = ['fg', 'bg']
shutil.rmtree(dir, ignore_errors=True)

for d in subdirs:
    d_path = os.path.join(dir, d)
    os.makedirs(d_path, exist_ok=True)
    csv_path = os.path.join(dir, f"{d}.csv")
    header = ['save_name', 'date', 'location', 'voltage', 'number', 'calculated_delay', 'programmed_delay', 'local_path']
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

df = pd.read_csv('ts_paths.csv')
df_CH = df.loc[df['material'] == MATERIAL]
saved_dict = {}

for row in df_CH.itertuples(index=False):
    date = row.path[2:7]
    data = h5py.File(row.path)
    fg_data, bg_data = process_h5(data)

    for dataset, ground in zip((fg_data, bg_data), ('fg', 'bg')):
        for element in dataset:
            save_name_prefix = f'CH_{ground}_loc{row.location}_{row.voltage}V'
            if save_name_prefix in saved_dict.keys():
                val = saved_dict[save_name_prefix]
                save_name = f'{save_name_prefix}_#{val}.tiff'
            else:
                save_name = f'{save_name_prefix}_#0.tiff'
                val = 0

            im = element[0]
            calc_delay = np.round(element[1], decimals=3)
            prog_delay = element[2]

            im_info_arr = [save_name, date, row.location, row.voltage, val, calc_delay, prog_delay, row.path]
            saved_dict[save_name_prefix] = (val + 1)
            save_path = os.path.join(dir, ground, save_name)

            tiff = Image.fromarray(im, mode='I')
            tiff.save(save_path, format='TIFF', compression='tiff_lzw')
            with open(os.path.join(dir, f'{ground}.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(im_info_arr)