import numpy as np
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt

path = 'archive/ts_tiff_images_full/no_heater_no_ts/CH'

# Get all TIFF files in the directory
tiff_files = sorted(glob.glob(os.path.join(path, '*.tiff')))

if not tiff_files:
    print(f"No TIFF files found in {path}")
else:
    spectra = []
    
    # Read and process each TIFF file
    for filepath in tiff_files:
        img = np.array(Image.open(filepath))
        
        # Clip to pixels 220-380 (rows)
        clipped = img[220:381,]
        
        # Average vertically to create spectrum (mean across rows)
        spectrum = np.mean(clipped, axis=0)
        spectra.append(spectrum)
    
    # Convert to 2D array (n_files x n_wavelengths)
    spectra = np.array(spectra)
    
    # Calculate detector uncertainty (standard deviation across all spectra)
    detector_sigma = np.std(spectra, axis=0)
    mean_spectrum = np.mean(spectra, axis=0)
    
    print(f"Number of images processed: {len(tiff_files)}")
    print(f"Mean uncertainty per pixel: {np.mean(detector_sigma):.4f}")
    print(f"Max uncertainty per pixel: {np.max(detector_sigma):.4f}")

    print(f"Avg percent error: {np.mean(detector_sigma / mean_spectrum) * 100:.4f}%")

    plt.axhline(0.2455, c='k', label='Average S/N (0.2469%)')
    plt.plot((detector_sigma / mean_spectrum) * 100)
    plt.xlabel('Pixel')
    plt.ylabel('Signal to noise ratio (%)')
    plt.grid(alpha=0.3)
    plt.xlim(0, 512)
    plt.legend()
    plt.title('Average Signal/Noise Across 1596 Blank Shots')
    plt.show()
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean spectrum
    axes[0, 0].plot(mean_spectrum, linewidth=1.5)
    axes[0, 0].set_title('Mean Spectrum (Blank Data)')
    axes[0, 0].set_xlabel('Pixel')
    axes[0, 0].set_ylabel('Intensity')
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Detector uncertainty per pixel
    axes[0, 1].plot(detector_sigma, linewidth=1.5, color='red')
    axes[0, 1].set_title('Detector Uncertainty (σ) per Pixel')
    axes[0, 1].set_xlabel('Pixel')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Sample of individual spectra
    for i in range(min(10, len(spectra))):
        axes[1, 0].plot(spectra[i], alpha=0.5, linewidth=0.8)
    axes[1, 0].set_title(f'Sample of {min(10, len(spectra))} Individual Spectra')
    axes[1, 0].set_xlabel('Pixel')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Histogram of uncertainty values
    axes[1, 1].hist(detector_sigma, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Distribution of Pixel Uncertainties')
    axes[1, 1].set_xlabel('Standard Deviation')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('detector_uncertainty_analysis.png', dpi=150)
    plt.show()

