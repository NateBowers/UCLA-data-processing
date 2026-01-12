import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from lmfit.models import GaussianModel, ConstantModel, VoigtModel
import lmfit

'''
Calibrate spectral response for Thomson Scattering

'''

file = h5py.File('neon-2025-06-11.h5')
images = file['13PICAM1:Pva1:Image']

imgs = np.array([images[f'image {i}'][230:320] for i in range(10)], dtype=float)
spectra = np.sum(imgs, axis=1)

avg = np.average(spectra, axis=0)
std = np.std(spectra, axis=0)
max_val = np.max(avg)
std /= max_val
avg /= max_val




res = np.polyfit(x = avg, y = std, deg = 1)

plt.plot(std)
plt.plot(res[0] * avg + res[1])
plt.show()


print(res * max_val)









# plt.plot(std / avg)
# plt.show()

# print(std)

# x = np.arange(512)
# # plt.plot(x, avg)
# plt.fill_between(x, avg-std, avg+std)
# plt.show()
# avg = np.average(spectra, axis=0)
# avg /= np.max(avg)
# for spectrum in spectra:
#     spectrum /= np.max(spectrum)
#     print(np.argmax(spectrum))
#     # plt.plot(spectrum)
#     plt.plot(np.square(spectrum - avg))

# # plt.plot(avg, linestyle='--')
# plt.show()


# spectra: shape (10, N)
# means = np.mean(spectra, axis=0)
# vars = np.var(spectra, axis=0, ddof=1)
# vars = np.square(np.std(spectra, axis=0))














spectra_avg = np.average(spectra, axis=0)[1: -1]
spectra_std = np.std(spectra, axis=0)


plt.plot(spectra_std / spectra_std)
plt.show()



# background = np.average(np.hstack((spectra_avg[0:240],spectra_avg[504:-1])))
# spectra_avg -= background
spectra_avg /= np.max(spectra_avg)



model = GaussianModel(prefix='peak1') + GaussianModel(prefix='peak2')  + GaussianModel(prefix='peak3')  + GaussianModel(prefix='peak4') + ConstantModel(prefix='background')
pixels = np.arange(len(spectra_avg)) + 1

params = lmfit.Parameters()
params.add('peak1amplitude', value = 2, min = 0)
params.add('peak1center', value = 263, min=0)
params.add('peak1sigma', 2, min=0)

params.add('peak2amplitude', value = 2, min = 0)
params.add('peak2center', value = 290, min=0)
params.add('peak2sigma', 2, min=0)

params.add('peak3amplitude', value = 1, min = 0)
params.add('peak3center', value = 300, min=0)
params.add('peak3sigma', 2, min=0)

params.add('peak4amplitude', value = 4, min = 0)
params.add('peak4center', value = 444, min=0)
params.add('peak4sigma', 2, min=0)

params.add('backgroundc', 0.1)


# plt.plot(model.eval(params, x = pixels))
# plt.show()
# par_guess = model.guess(spectra_avg, x=pixels)
res = model.fit(
    spectra_avg,
    x=pixels,
    params = params,
    # weights = 1 / spectra_std,
)
print(res.fit_report(min_correl=0.25))
print(res.ci_report())

fig = plt.figure(figsize=(12, 8), constrained_layout=True)
res.plot(fig = fig, title='Four gaussians fitted to Ne lamp data', fitfmt='k--')

# plt.savefig('neon_fitted_gauss.png')
plt.show()
# print(background)

# peak1_idx = (429, 464)
# peak2_idx = (245, 283)
# peak3_idx = (273, 315)

# peak1 = spectra_avg[peak1_idx[0]: peak1_idx[1]]
# y = peak1
# x = np.arange(len(peak1))

# model = GaussianModel()
# pars = model.guess(y, x=x)
# out = model.fit(y, pars, x=x)

# print(out.fit_report(min_correl=0.25))









# def gauss(arr, mid, fwhm, max):
#     sigma = fwhm / 2.355
#     res = norm.pdf(x = arr, loc = mid, scale = sigma)
#     res /= np.max(res)
#     res *= max
#     return res

# image_pixel_range = np.arange(512)

# peak_maxes = [
#     1,
#     0.4698092620905653,
#     0.4523084134527071
# ]

# peaks_fwhm = [
#     4.5920,
#     4.5043,
#     4.9222,
# ]

# peaks_mid_px = [444.3050, 263.3897, 290.3282]
# peaks_mid_nm = [540.06, 533.08, 534.11]

# three_peak_arr_gauss = np.zeros_like(image_pixel_range, dtype=float)
# for i in range(3):
#     three_peak_arr_gauss += gauss(
#         image_pixel_range, 
#         mid = peaks_mid_px[i],
#         fwhm = peaks_fwhm[i],
#         max = peak_maxes[i]
#     )



# # plt.scatter(peaks_mid_px, peaks_mid_nm)
# # plt.show()



# def find_fwhm_x(p1, p2):
#     x1, y1 = p1
#     x2, y2 = p2
#     m = (y2-y1) / (x2-x1)
#     fwhm_x = (0.5 - y1 + m * x1) / m
#     return fwhm_x






# peak1_idx = (429, 464)
# peak2_idx = (245, 283)
# peak3_idx = (273, 315)

# peak_idxs = peak2_idx
# peak = spectra_avg[peak_idxs[0]: peak_idxs[1]]
# peak_pixels = np.arange(peak_idxs[0], peak_idxs[1])
# peak_front = peak[0:10]
# peak_back = peak[-10: -1]
# peak_baseline = np.average(np.hstack((peak_front, peak_back)))
# peak -= peak_baseline
# peak /= np.max(peak)

# eps = 0
# fwhm_idxs = [0]
# while len(fwhm_idxs) != 4:
#     fwhm_idxs = np.argwhere(np.abs(peak - 0.5) < eps)
#     eps += 0.001
#     if len(fwhm_idxs) > 4:
#         raise('something happened, more than four fwhm indexes')
    
# fwhm_idxs = np.array([287, 288, 292, 293]) - peak_idxs[0]

# fwhm_x1 = find_fwhm_x(
#     p1 = (fwhm_idxs[0] + peak_idxs[0], peak[fwhm_idxs[0]]),
#     p2 = (fwhm_idxs[1] + peak_idxs[0], peak[fwhm_idxs[1]]),
# )

# fwhm_x2 = find_fwhm_x(
#     p1 = (fwhm_idxs[2] + peak_idxs[0], peak[fwhm_idxs[2]]),
#     p2 = (fwhm_idxs[3] + peak_idxs[0], peak[fwhm_idxs[3]]),
# )

# plt.plot(spectra_avg)
# plt.plot(three_peak_arr_gauss, linestyle='--', c='k')
# plt.show()









# print(fwhm_idxs)



# plt.axvline(13 + peak_idxs[0])
# plt.axvline(17 + peak_idxs[0])
# plt.axvline(18 + peak_idxs[0])

# plt.axvline(444.3047, c='r')


# baseline = np.hstack((
#     spectra_avg[5:245],
#     spectra_avg[307:429],
#     spectra_avg[464:504]
# ))

# baseline_idx = np.hstack((
#     np.arange(5, 245),
#     np.arange(307, 429),
#     np.arange(464, 504)

# ))







# fig = plt.figure(constrained_layout=True, figsize=(6, 6))
# ax1, ax2 = fig.subplots(2, height_ratios=(0.8, 0.2), sharex=True)

# ax2.plot(baseline_idx, baseline, label='Spectrum without peaks')
# # ax2.set_title('Neon spectra without peaks')
# ax2.set_xlabel('Pixels')
# ax2.legend()


# ax1.plot(spectra_avg, c='k', label='Average spectra')
# ax1.set_title('Neon spectra')
# ax1.set_ylabel('Normalized intensity')
# ax1.fill_between(np.arange(512), spectra_avg - spectra_std, spectra_avg + spectra_std, color='r', label='Statistical standard deviation')
# ax1.legend()
# plt.savefig('neon spectrum.png')



# sample_im = np.array(images['image 0'], dtype=int)
# sample_im_clipped = sample_im.copy()
# sample_im_clipped[0:230] = -3000
# sample_im_clipped[320:-1] = -3000

# plt.matshow(sample_im_clipped, cmap='magma')
# # plt.show()
# # plt.savefig('neon_clip_region.png')


# fig = plt.figure(constrained_layout=True, figsize=(9, 4))
# axs = fig.subplots(2, 5)

# for i, ax in enumerate(axs.flatten()):
#     im = np.array(images[f'image {i}'], dtype=int)
#     im[0,0] = -3000
#     ax.matshow(im, aspect='equal', cmap='magma')
#     ax.axis('off')
#     ax.set_title(f'shot {i}')

# plt.show()
# plt.savefig('neon_lamp_raw.png')