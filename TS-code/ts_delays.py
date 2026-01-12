import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel
from scipy.stats import norm

wavelengths = (np.arange(512) * 19.80636 / 511) + 522.918



def find_delays(file: str):
    data = h5py.File(file)

    ts = np.array(data['LeCroy:Ch2:Trace'])
    heater = np.array(data['LeCroy:Ch1:Trace'])
    times = np.array(data['LeCroy:Time'])
    delays = np.array(data['actionlist/TS:1w2wDelay'])

    dt = np.round(np.average(times[:,1:] - times[:,:-1]), 10)
    ts_max_idx = np.argmax(np.gradient(ts, axis=1), axis=1)
    heater_max_idx = np.argmax(np.gradient(heater, axis=1), axis=1)
    real_delays = (ts_max_idx - heater_max_idx) * dt * 1e9

    ts_max = np.max(ts, axis=1)
    ts_shutter = np.array((ts_max > 2), dtype=int)
    heater_max = np.max(heater, axis=1)
    heater_shutter = np.array([a != heater_max[i-1] for i, a in enumerate(heater_max)]).astype(int)

    shutter_delay_arr = np.array(
        [
            ts_shutter, 
            heater_shutter, 
            delays, 
            real_delays
        ]
    ).T
    
    info_arr = np.array(sorted(shutter_delay_arr, key=lambda x: x[2]))

    fg_idxs = np.argwhere(np.all(info_arr[:,:2] == (1, 1), axis=1)).T[0]
    bg_idxs = np.argwhere(np.all(info_arr[:,:2] == (0, 1), axis=1)).T[0]

    fg_real_delays = real_delays[fg_idxs]
    fg_delays = delays[fg_idxs]

    return fg_real_delays, fg_delays



df = pd.read_csv('ts_paths.csv')
CH_df = df[df['material'] == 'CH']


paths = ['./06-05/TS_CH_Location3_0V-2025-06-05.h5', './06-06/TS_Cu_Location4_100V-2025-06-06.h5']
# all_real_delays = []
# all_delays = []

# for path in CH_df['path']:
#     real_delay, delay = find_delays(path)
#     all_real_delays.append(real_delay)
#     all_delays.append(delay)


# real_delays_flat = np.array([item for sublist in all_real_delays for item in sublist])
# delays_flat = np.array([item for sublist in all_delays for item in sublist])

# print(len(delays_flat))

# def f(x, offset): return x + offset

# popt, pcov = curve_fit(f, real_delays_flat, delays_flat)
# print(popt)
# print(pcov)

# # plt.scatter(real_delays_flat, delays_flat, c='k', s=2)
# # offset = popt
# # plt.plot([np.min(real_delays_flat) - offset, np.max(real_delays_flat) - offset], [np.min(real_delays_flat), np.max(real_delays_flat)], linestyle='--', c='b', lw=1, label='5.229 ns offset')
# # plt.plot([np.min(real_delays_flat), np.max(real_delays_flat)], [np.min(real_delays_flat), np.max(real_delays_flat)], c='r', lw=1, label=f'$x=y$')
# # plt.legend()
# # plt.xlabel('Thomson beam peak - heater beam peak (ns)')
# # plt.ylabel('Programmed delay (ns)')
# # plt.savefig('delay_actual_predict.png', dpi=300)


# adjusted_delays = real_delays_flat - delays_flat
# # Fit a Gaussian to the adjusted_delays histogram


# mu, std = norm.fit(adjusted_delays)
# print(mu)
# print(std)
# xmin = np.min(adjusted_delays)
# xmax = np.max(adjusted_delays)
# # xmin, xmax = -20, 20
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std) * len(adjusted_delays) * (xmax - xmin) / 50  # scale to histogram

# plt.plot(x, p, 'r--', linewidth=2, label=f'Gaussian fit\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')
# plt.legend()
# plt.hist(adjusted_delays, bins=50, color='skyblue', edgecolor='black')
# plt.xlabel('Difference from programmed delay (ns)')
# plt.ylabel('Count')
# # plt.title('Histogram of Adjusted Delays')
# plt.savefig('delay_histogram.png', dpi=300)






def find_background_w_delay(file: str):
    data = h5py.File(file)

    ts = np.array(data['LeCroy:Ch2:Trace'])
    heater = np.array(data['LeCroy:Ch1:Trace'])
    times = np.array(data['LeCroy:Time'])
    delays = np.array(data['actionlist/TS:1w2wDelay'])
    images = np.array([np.array(data[f'13PICAM1:Pva1:Image/image {n}'], dtype=int) for n in range(len(data['13PICAM1:Pva1:Image']))])


    dt = np.round(np.average(times[:,1:] - times[:,:-1]), 10)
    ts_max_idx = np.argmax(np.gradient(ts, axis=1), axis=1)
    heater_max_idx = np.argmax(np.gradient(heater, axis=1), axis=1)
    real_delays = (ts_max_idx - heater_max_idx) * dt * 1e9

    ts_max = np.max(ts, axis=1)
    ts_shutter = np.array((ts_max > 2), dtype=int)
    heater_max = np.max(heater, axis=1)
    heater_shutter = np.array([a != heater_max[i-1] for i, a in enumerate(heater_max)]).astype(int)

    info_arr = np.array([ts_shutter, heater_shutter]).T


    bg_idxs = np.argwhere(np.all(info_arr[:,:] == (0, 1), axis=1)).T[0]
    bg_imgs = images[bg_idxs]
    bg_delays = delays[bg_idxs]


    return bg_imgs, bg_delays

def find_foreground_w_delay(file: str):
    data = h5py.File(file)

    ts = np.array(data['LeCroy:Ch2:Trace'])
    heater = np.array(data['LeCroy:Ch1:Trace'])
    times = np.array(data['LeCroy:Time'])
    delays = np.array(data['actionlist/TS:1w2wDelay'])
    images = np.array([np.array(data[f'13PICAM1:Pva1:Image/image {n}'], dtype=int) for n in range(len(data['13PICAM1:Pva1:Image']))])


    dt = np.round(np.average(times[:,1:] - times[:,:-1]), 10)
    ts_max_idx = np.argmax(np.gradient(ts, axis=1), axis=1)
    heater_max_idx = np.argmax(np.gradient(heater, axis=1), axis=1)
    real_delays = (ts_max_idx - heater_max_idx) * dt * 1e9

    ts_max = np.max(ts, axis=1)
    ts_shutter = np.array((ts_max > 2), dtype=int)
    heater_max = np.max(heater, axis=1)
    heater_shutter = np.array([a != heater_max[i-1] for i, a in enumerate(heater_max)]).astype(int)

    info_arr = np.array([ts_shutter, heater_shutter]).T


    fg_idxs = np.argwhere(np.all(info_arr[:,:] == (1, 1), axis=1)).T[0]
    fg_imgs = images[fg_idxs]
    fg_delays = delays[fg_idxs]


    return fg_imgs, fg_delays


file = '06-05/TS_CH_Location1_0V-2025-06-05.h5'
bg, bg_delays = find_background_w_delay(file)
fg, fg_delays = find_foreground_w_delay(file)


unique_delays = np.unique(bg_delays)

# fig = plt.figure()
# fig.suptitle(f'{delay} ns delay')
# ax1, ax2, ax3 = fig.subplots(3, 1)

# foo = np.zeros(512)

# print(bg.shape)

# bg_spectra = np.sum(bg, axis=1)
# bg_std = np.std(bg_spectra, axis=0)
# bg_avg = np.average(bg_spectra, axis=0)

# fg_spectra = np.sum(fg, axis=1)
# fg_std = np.std(fg_spectra, axis=0)
# fg_avg = np.average(fg_spectra, axis=0)

# s_avg = fg_avg - bg_avg
# s_std = np.sqrt(np.square(fg_std) + np.square(bg_std))

# ax3.plot(s_avg, c='b')
# ax3.fill_between(np.arange(len(s_avg)), s_avg - s_std, s_avg + s_std, alpha=0.5)


delays = [160]

bg_delay_idxs = np.where(np.isin(bg_delays, delays))[0]
fg_delay_idxs = np.where(np.isin(fg_delays, delays))[0]

bg_delay_idxs = bg_delay_idxs[1]
fg_delay_idxs = fg_delay_idxs[1]

bg_delay_imgs = bg[bg_delay_idxs]
fg_delay_imgs = fg[fg_delay_idxs]

bg_delay_spectra = np.sum(bg_delay_imgs, axis=0)
fg_delay_spectra = np.sum(fg_delay_imgs, axis=0)

spectra = fg_delay_spectra - bg_delay_spectra
# spectra_avg = np.average(spectra, axis=0)
spectra_avg = spectra
# spectra_max = np.max(spectra, axis=0)
# spectra_min = np.min(spectra, axis=0)

# bg_std = np.std(bg_delay_spectra, axis=0)
# fg_std = np.std(fg_delay_spectra, axis=0)

# spectra_std = np.sqrt(np.square(bg_std) + np.square(fg_std))

# data = np.vstack((spectra_avg, spectra_std))
np.savez('sample_data2.npz', spectra_avg = spectra_avg)


plt.plot(wavelengths, spectra_avg)
# plt.plot(spectra_max)
# plt.plot(spectra_min)
# plt.fill_between(wavelengths, spectra_avg - spectra_std, spectra_avg + spectra_std, alpha=0.5)
plt.show()



# for im in bg_delay_imgs:
#     bg_spectrum = np.sum(im, axis=0)
#     ax_scat.scatter(wavelengths, bg_spectrum, s=1, c='b')

# for im in fg_delay_imgs:
#     fg_spectrum = np.sum(im, axis=0)
#     ax_scat.scatter(wavelengths, fg_spectrum, s=1, c='r')

# bg_spectrum = np.sum(np.average(bg_delay_imgs, axis=0), axis=0)
# fg_spectrum = np.sum(np.average(fg_delay_imgs, axis=0), axis=0)

# ax1.plot(bg_spectrum[10:-10], label=f'bg, {delay} ns')
# ax1.plot(fg_spectrum[10:-10], label=f'fg, {delay} ns')

# spectrum = fg_spectrum - bg_spectrum
# ax2.plot(spectrum, label=f'{delay} ns')

# foo += spectrum
# print(bg_delay_imgs.shape)
# avg_bg = np.average(bg_delay_imgs, axis=0)
# std_bg = np.std(bg_delay_imgs, axis=0)



# ax2.legend()
# ax1.legend()
plt.show()
# print(delays)