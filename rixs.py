"""
Mila Fitzgerald, Oct 2025

RIXS analysis on SACLA cluster

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.special import wofz
from matplotlib.colors import LogNorm
import matplotlib.cm as cm

class SpectrometerRIXS:
    """
    Class to load spectrometer .tif data, process calibration shots,
    and determine the dispersion axis from emission line positions.
    """

    def __init__(self, base_path, fssr_name, detector_size=(512, 2048)):
        """
        Initialize calibration class.
        
        Parameters
        ----------
        base_path : str
            Path to the main analysis folder containing subdirs for spectrometers.
        fssr_name : str
            Identifier for spectrometer (used in subfolder names).
        detector_size : tuple
            Expected shape of detector images (height, width).
        """
        self.base_path = base_path
        self.fssr = fssr_name
        self.detector_size = detector_size
        self.data = None
        self.dark = None
        self.image_avg = None
        self.dispersion_fit = None

        # Define path for tif files and create directory if missing
        self.tif_dir = os.path.join(self.base_path, self.fssr, "tif")
        if not os.path.exists(self.tif_dir):
            os.makedirs(self.tif_dir)
            print(f"Created directory: {self.tif_dir}")
        else:
            print(f"Using existing directory: {self.tif_dir}")
            
    # ----------------------------------------------------------
    # Create hot pixel mask
    # ---------------------------------------------------------- 
    @staticmethod
    def otsu_threshold(image):
        """
        Compute Otsu's threshold for a grayscale image using pure NumPy.
    
        Parameters
        ----------
        image : np.ndarray
            2D image array (float or int).
    
        Returns
        -------
        threshold : float
            Intensity threshold that separates background and foreground.
        """
        # Flatten and normalize image
        img = image.ravel()
        img = img[~np.isnan(img)]  # remove NaNs if any
        img = img[img >= 0]        # ignore negative values (if relevant)
        
        # Compute histogram
        hist, bin_edges = np.histogram(img, bins=256, range=(img.min(), img.max()))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
        # Compute normalized histogram (probabilities)
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
    
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    
        # Compute inter-class variance
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2
    
        # Find threshold that maximizes variance
        idx = np.argmax(variance12)
        threshold = bin_centers[idx]
        return threshold
    
    def create_hotpixel_mask(self, runs, dark_run=None, n_dark=1, use_otsu=True, mean=False, plot=True):
        """
        Load multiple .tif images for calibration runs and compute average.
        Optionally subtract a dark run.

        Parameters
        ----------
        runs : list[int]
            List of run numbers to average.
        dark_run : int or None
            Run number for dark image (optional).
        n_dark : int
            Number of dark images averaged into dark_run (for normalization).
        """
        tif_dir = self.tif_dir

        # Load calibration runs
        imgs = []
        for r in runs:
            path = os.path.join(tif_dir, f"{r}_{self.fssr}.tif")
            imgs.append(np.array(Image.open(path), dtype=float))
            
        self.hotpixelIm = np.mean(imgs, axis=0) if mean else np.sum(imgs, axis=0)

        # Load dark frame if provided
        if dark_run is not None:
            dark_path = os.path.join(tif_dir, f"{dark_run}_{self.fssr}.tif")
            self.dark = np.array(Image.open(dark_path), dtype=float) / n_dark
            self.hotpixelIm -= self.dark

        if use_otsu:
            print("Applying otsu threshold filter")
            # Apply Otsu thresholding
            otsu_thresh = self.otsu_threshold(self.hotpixelIm)
            self.hotpixel_mask = self.hotpixelIm > otsu_thresh

            print(f"Otsu threshold: {otsu_thresh:.3f}")
            
            thresh = otsu_thresh
            
        else:
            print("Filtering using (summed mean + 4 * std deviation)")
            summed_mean = np.mean(self.hotpixelIm)
            std_deviation = np.std(self.hotpixelIm)
            thresh = summed_mean + 4 * std_deviation 

            self.hotpixel_mask = self.hotpixelIm > thresh
            
            print(f"Hot pixels detected: {np.sum(self.hotpixel_mask)}")

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            ax0, ax1, ax2 = axes
    
            ax0.imshow(self.hotpixelIm, cmap='inferno')
            ax0.set_title("Combined calibration image")
            ax0.axis('off')
    
            ax1.hist(self.hotpixelIm.ravel(), bins=200, color='gray', alpha=0.7)
            ax1.axvline(thresh, color='r', linestyle='--', label=f"Threshold = {thresh:.2f}")
            ax1.legend()
            ax1.set_title("Histogram and threshold")
    
            ax2.imshow(self.hotpixel_mask, cmap='gray')
            ax2.set_title("Hot pixel mask")
            ax2.axis('off')
    
            plt.tight_layout()
            plt.show()
            
    
        return self.hotpixel_mask
            
    # ----------------------------------------------------------
    # Load & average/sum images
    # ----------------------------------------------------------
    def load_images(self, runs, dark_run=None, n_dark=1, hotpixelmask_filepath = None, mean=True, plot=True):
        """
        Load multiple .tif images for calibration runs and compute average.
        Optionally subtract a dark run.

        Parameters
        ----------
        runs : list[int]
            List of run numbers to average.
        dark_run : int or None
            Run number for dark image (optional).
        n_dark : int
            Number of dark images averaged into dark_run (for normalization).
        """
        tif_dir = self.tif_dir

        # Load calibration runs
        imgs = []
        for r in runs:
            path = os.path.join(tif_dir, f"{r}_{self.fssr}.tif")
            imgs.append(np.array(Image.open(path), dtype=float))

        self.image_avg = np.mean(imgs, axis=0) if mean else np.sum(imgs, axis=0)

        # Load dark frame if provided
        if dark_run is not None:
            dark_path = os.path.join(tif_dir, f"{dark_run}_{self.fssr}.tif")
            self.dark = np.array(Image.open(dark_path), dtype=float) / n_dark
            self.image_avg -= self.dark

        self.data = self.image_avg.copy()
        
        if hotpixelmask_filepath is not None:
            print(f"🔥 Hot pixel mask applied")
            hotpixelmask = np.load(hotpixelmask_filepath)
            masked_img = np.where(hotpixelmask, img, 0)
            self.data = clean_image_avg.copy()
            
        print(f"✅ Loaded and summed {len(runs)} runs for {self.fssr}.")
        
        if plot:
            average_profile = np.nanmean(self.image_avg, axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(self.data, cmap='inferno', origin='lower', norm=LogNorm(vmin=1.0, vmax=self.data.max()),
                       aspect='auto')
            plt.colorbar(label='Intensity (a.u.)')
            plt.plot(average_profile, color='yellow', label='Average intensity over ROI')#,label=savename)
            plt.xlabel('Pixel x')
            plt.ylabel('Pixel y')
            plt.legend()
            plt.grid(True)
            
    # ----------------------------------------------------------
    # Extract lineouts along detector Y axis
    # ----------------------------------------------------------  
    @staticmethod
    def lorentzian(x, x0, gamma, A, offset):
        return A * gamma**2 / ((x - x0)**2 + gamma**2) + offset
        
    def find_vertical_peaks_and_fit(self, n_bins=40, plot=True):
        """
        Bin the image into vertical strips, fit Lorentzian to the integrated signal
        in each strip, then perform a weighted linear fit of peak position vs strip center.
    
        Parameters
        ----------
        n_bins : int
            Number of vertical bins (across x-direction).
        plot : bool
            Whether to show the 2D image with fits and the final weighted linear fit.
        """
    
        if self.data is None:
            raise ValueError("No image data loaded.")
    
        img = self.data
        h, w = img.shape
        x_bins = np.linspace(0, w, n_bins + 1, dtype=int)
        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    
        y = np.arange(h)
        peak_positions = np.full(n_bins, np.nan)
        peak_amplitudes = np.full(n_bins, np.nan)
    
        for i in range(n_bins):
            x1, x2 = x_bins[i], x_bins[i + 1]
            if x2 <= x1:
                continue
            profile = np.sum(img[:, x1:x2], axis=1)  # integrate vertically
    
            # initial guesses
            y0_guess = np.argmax(profile)
            A_guess = profile[y0_guess] - np.median(profile)
            gamma_guess = 10
            offset_guess = np.median(profile)
            p0 = [y0_guess, gamma_guess, A_guess, offset_guess]
    
            try:
                popt, _ = curve_fit(self.lorentzian, y, profile, p0=p0, maxfev=5000)
                y0_fit, gamma_fit, A_fit, offset_fit = popt
                peak_positions[i] = y0_fit
                peak_amplitudes[i] = A_fit
            except RuntimeError:
                continue  # skip bad fits
    
        # remove NaN fits
        valid = ~np.isnan(peak_positions) & ~np.isnan(peak_amplitudes)
        x_valid = x_centers[valid]
        y_valid = peak_positions[valid]
        weights = 1 / peak_amplitudes[valid]
    
        # weighted linear fit
        coeffs = np.polyfit(x_valid, y_valid, deg=1, w=weights)
        fit_fn = np.poly1d(coeffs)
        slope, intercept = coeffs
    
        # store results
        self.peak_positions = peak_positions
        self.peak_amplitudes = peak_amplitudes
        self.fit_coeffs = coeffs
    
        if plot:
            # Create figure with two columns — shared y-axis
            fig, (ax_img, ax_prof) = plt.subplots(
                1, 2, figsize=(12, 6), sharey=True, 
                gridspec_kw={'width_ratios': [2.5, 1]}
            )
        
            # --- Left panel: 2D image with Lorentzian fits ---
            im = ax_img.imshow(
                img, cmap='inferno', origin='lower',
                norm=LogNorm(vmin=1, vmax=img.max()), aspect='auto'
            )
            cbar = plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
            cbar.set_label("Intensity (a.u.)")
        
            ax_img.scatter(x_valid, y_valid, marker='x', c='cyan', label='Lorentzian centers')
            ax_img.plot(x_valid, fit_fn(x_valid), 'w--', label='Weighted linear fit')
        
            ax_img.set_xlabel("Pixel (dispersion direction)")
            ax_img.set_ylabel("Pixel (spatial direction)")
            ax_img.set_title("Vertical bins with Lorentzian-fitted peak positions")
            ax_img.legend(loc='upper right')
        
            # --- Right panel: Integrated vertical profiles ---
            colors = cm.coolwarm(np.linspace(0, 1, len(x_valid)))
            for i, xv in enumerate(x_valid):
                ax_prof.plot(
                    np.sum(img[:, x_bins[i]:x_bins[i + 1]], axis=1),
                    y, color=colors[i], alpha=0.5
                )
        
            ax_prof.set_xlabel("Integrated intensity")
            ax_prof.set_title("Integrated vertical profiles")
            ax_prof.invert_xaxis()  # optional: makes direction intuitive next to image
            ax_prof.grid(alpha=0.3)
        
            # --- Final layout tweaks ---
            plt.tight_layout()
            plt.show()
    
        print(f"Weighted fit: y = {slope:.4f} * x + {intercept:.2f}")
        # return slope, intercept, peak_positions, peak_amplitudes
    
    # ----------------------------------------------------------
    # Generate dispersion axis mask
    # ---------------------------------------------------------- 
    def generate_dispersion_axis_mask(self, width=5, plot=True):
        """
        Create a boolean mask selecting pixels along the fitted dispersion axis.
    
        Parameters
        ----------
        width : int
            Half-width (in pixels) around the fitted line to include in the mask.
        plot : bool
            If True, plot the mask overlayed on the image.
    
        Returns
        -------
        mask : np.ndarray (bool)
            Boolean mask the same size as the image, with True along the dispersion axis.
        """
        
        if not hasattr(self, "fit_coeffs"):
            raise AttributeError("No dispersion fit found. Run find_vertical_peaks_and_fit() first.")

        if self.data is None:
            raise ValueError("No image data loaded.")
            
        img = self.data
        h, w = img.shape
        slope, intercept = self.fit_coeffs
    
        # Create pixel coordinate grids
        y_indices = np.arange(h)
        x_indices = np.arange(w)
        X, Y = np.meshgrid(x_indices, y_indices)
    
        # Compute the expected y position of the dispersion line for each x
        y_fit = slope * X + intercept
    
        # Create mask where the actual Y is within ±width of the fitted line
        self.dispersion_mask = np.abs(Y - y_fit) <= width

        masked_img = np.where(self.dispersion_mask, img, 0)
    
        if plot:
            plt.figure(figsize=(10, 6))
            plt.imshow(img, cmap='inferno', origin='lower', norm=LogNorm(vmin=1, vmax=img.max()),
                       aspect='auto')
            plt.contour(self.dispersion_mask, colors='cyan', linewidths=0.6)
            plt.title(f"Dispersion mask (±{width} px around fitted line)")
            plt.xlabel("X (dispersion direction)")
            plt.ylabel("Y (spatial direction)")
            plt.tight_layout()

            plt.figure(figsize=(10, 2))
            plt.imshow(masked_img, cmap='inferno', origin='lower', norm=LogNorm(vmin=1, vmax=masked_img.max()),
                       aspect='auto')
            plt.title(f"Masked image")
            plt.xlabel("X (dispersion direction)")
            plt.ylabel("Y (spatial direction)")
            plt.show()
    
        print(f"Created dispersion mask ±{width} pixels wide around fitted line.")
        return self.dispersion_mask
        
    # ----------------------------------------------------------
    # Fit to calibration shots to find pixel -> eV conversion
    # ---------------------------------------------------------- 
    def calculate_spatial_energy_axis_conversion(self, peakenergies_input=None, dispersionmask_filepath=None, gaussian_blur=1.0, plot=True):
        """
        Integrate the hotpixel and dispersion masked calibration image to find 
        the pixel -> energy conversion.
    
        Parameters
        ----------
        peakenergies_list : list
            List of expected line energies based on functions in calibration.py.
        """

        if self.data is None:
            raise ValueError("No image data loaded.")
    
        img = self.data.copy()

        if dispersionmask_filepath is not None:
            dispresionaxismask = np.load(dispersionmask_filepath)
            masked_img = np.where(dispresionaxismask, img, 0)
            img = masked_img.copy()

        profile = np.sum(img, axis=0)  # integrate horizontally

        x = np.arange(profile.size)
        y = profile.astype(float)

        if peakenergies_input is None:
            raise ValueError("Please provide peakenergies_input (expected energies in keV).")
        
        element_array = peakenergies_input["Element"] 
        element_list = np.asarray(list(element_array), dtype=str)
        line_array = peakenergies_input["Line"]
        line_list = np.asarray(list(line_array), dtype=str)

        peakenergies_array = peakenergies_input["Energy_keV"]
        peakenergies_list = np.asarray(list(peakenergies_array), dtype=float)
        n_peaks = len(peakenergies_list)
        colors = cm.jet(np.linspace(0, 1, n_peaks))
        print(f"Expecting {n_peaks} peaks, at {peakenergies_list}")

        # Smoothing to help peak detection (small sigma)
        y_s = gaussian_filter1d(y, sigma=gaussian_blur)

        plt.figure(figsize=(10, 6))
        plt.title(f"Gaussian filter ({gaussian_blur}) applied to data")
        plt.plot(profile)
        plt.plot(y_s)
        plt.yscale('log')
        plt.xlabel("Energy (keV)")
        plt.ylabel("Integrated intensity")
        plt.show()

        # detect candidate peaks (tune prominence/distance if needed)
        # Use a conservative prominence to avoid tiny noise peaks
        prominence = 0.05 * (y_s.max() - y_s.min())
        distance = max(3, int(0.5 * profile.size / 200))  # heuristic
        cand_idx, props = find_peaks(y_s, prominence=prominence, distance=distance)
        if cand_idx.size == 0:
            # fallback: take top-n pixels
            cand_idx = np.argsort(y)[-max(n_peaks, 6):]

        # Sort candidate peaks by pixel position (monotonic dispersion assumption)
        cand_idx_sorted = np.sort(cand_idx)

        # Sort energies ascending and pair with sorted candidate pixels
        # If more candidates than expected, select the n_peaks strongest in each neighborhood.
        # Simple approach: choose the n_peaks candidates centered across array:
        if cand_idx_sorted.size < n_peaks:
            # If too few candidates, fall back to taking top n pixels by height
            cand_idx_sorted = np.argsort(y)[-n_peaks:]
            cand_idx_sorted = np.sort(cand_idx_sorted)
    
        # If there are more candidates than peaks, choose the ones that best sample the axis:
        if cand_idx_sorted.size > n_peaks:
            # pick the n_peaks with highest local peak heights but keep order
            heights = y_s[cand_idx_sorted]
            top_order = np.argsort(heights)[-n_peaks:]
            chosen = np.sort(cand_idx_sorted[top_order])
        else:
            chosen = cand_idx_sorted

        ## Now pair by order: sort energies ascending -> pair with chosen pixels 
        # energy_sorted_idx = np.argsort(peakenergies_list) # ascending energies
        energy_sorted_idx = np.argsort(peakenergies_list)[::-1]   # descending energies

        energies_sorted = peakenergies_list[energy_sorted_idx]
        pixels_sorted = np.sort(chosen)
    
        if pixels_sorted.size != energies_sorted.size:
            # if counts mismatch, try to pick 'n_peaks' strongest candidates and pair
            top_cand = cand_idx[np.argsort(props.get('prominences', np.ones_like(cand_idx)))][-n_peaks:]
            pixels_sorted = np.sort(top_cand)
            # if still mismatch, truncate or pad (rare)
            pixels_sorted = np.resize(pixels_sorted, energies_sorted.shape)

        # Build initial guesses: amplitude from profile, sigma/gamma defaults
        init_centers = pixels_sorted
        init_amps = y_s[init_centers] - np.median(y_s)
        init_sig = np.full(n_peaks, 1.0)   # px, instrument-dependent
        init_gam = np.full(n_peaks, 0.5)

        def voigt_profile(xv, amplitude, center, sigma, gamma):
            z = ((xv - center) + 1j * gamma) / (sigma * np.sqrt(2))
            return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

        def multi_voigt(xv, *params_flat):
            # params_flat: [amp0, cen0, sig0, gam0, amp1, cen1, ... , b0, b1]
            n = (len(params_flat) - 2) // 4
            y_model = np.zeros_like(xv, dtype=float)
            for j in range(n):
                a = params_flat[4*j + 0]
                c = params_flat[4*j + 1]
                s = params_flat[4*j + 2]
                g = params_flat[4*j + 3]
                y_model += voigt_profile(xv, a, c, s, g)
            b0 = params_flat[-2]
            b1 = params_flat[-1]
            return y_model + b0 + b1 * xv

        p0 = []
        lower = []
        upper = []
        center_window = 3.0
        for i in range(n_peaks):
            a0 = max(1e-6, float(init_amps[i]))
            c0 = float(init_centers[i])
            s0 = float(init_sig[i])
            g0 = float(init_gam[i])
            p0 += [a0, c0, s0, g0]
            lower += [0.0, c0 - center_window, 0.01, 0.01]
            upper += [np.inf, c0 + center_window, 10.0, 10.0]
        p0 += [np.median(y), 0.0]
        lower += [-np.inf, -np.inf]
        upper += [np.inf, np.inf]

        popt, pcov = curve_fit(multi_voigt, x, y, p0=p0, bounds=(lower, upper), maxfev=20000)
        model_y = multi_voigt(x, *popt)

        fitted_peaks = []
        for i in range(n_peaks):
            cen = popt[4*i + 1]
            cen_err = np.sqrt(pcov[4*i + 1, 4*i + 1]) if pcov is not None else np.nan
            amp = popt[4*i + 0]
            fitted_peaks.append({
                'energy_keV': float(energies_sorted[i]),
                'pixel': float(cen),
                'pixel_err': float(cen_err),
                'amplitude': float(amp)
            })
        fit_out = (popt, pcov)

        # Build pixel -> energy linear calibration (fit energy = m*pixel + b)
        pixels = np.array([p['pixel'] for p in fitted_peaks])
        pixel_errs = np.array([p['pixel_err'] for p in fitted_peaks])
        energies = np.array([p['energy_keV'] for p in fitted_peaks])

        # Simple weighted linear fit using np.polyfit with weights = 1/pixel_err (if available)
        # But pixel_err sometimes NaN -> fall back to unweighted fit
        good = ~np.isnan(pixel_errs) & (pixel_errs > 0)
        if good.sum() >= 2:
            w = 1.0 / pixel_errs[good]
            # fit energy = m * pixel + b
            m, b = np.polyfit(pixels[good], energies[good], deg=1, w=w)
            # Estimate covariance from residuals (approx)
            # Compute residual variance
            res = energies[good] - (m * pixels[good] + b)
            dof = max(1, good.sum() - 2)
            s2 = np.sum((w * res)**2) / dof
            # approximate covariance matrix
            # use standard linear regression normal equations for weighted case
            X = np.vstack([pixels[good], np.ones_like(pixels[good])]).T
            W = np.diag(w**2)
            try:
                cov = np.linalg.inv(X.T @ W @ X) * s2
            except np.linalg.LinAlgError:
                cov = np.full((2,2), np.nan)
        else:
            # unweighted
            m, b = np.polyfit(pixels, energies, deg=1)
            # covariance from np.polyfit not directly accessible; estimate using residuals
            res = energies - (m * pixels + b)
            dof = max(1, pixels.size - 2)
            s2 = np.sum(res**2) / dof
            X = np.vstack([pixels, np.ones_like(pixels)]).T
            try:
                cov = np.linalg.inv(X.T @ X) * s2
            except np.linalg.LinAlgError:
                cov = np.full((2,2), np.nan)
    
        pixel_to_energy = {'slope_keV_per_px': float(m), 'intercept_keV': float(b)}

        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            ax = axes[0]
            ax.plot(x, y, label='data', lw=1.5)
            ax.plot(x, model_y, label='fit', lw=0.5)
            # individual components (if lmfit, we used composite access; for scipy rebuild)
            for i, p in enumerate(fitted_peaks):
                # approximate component (use fitted amp/cent/sig/gam if available)
                try:
                    if use_lmfit:
                        a = fit_out.params[f'amp_{i}'].value
                        c = fit_out.params[f'cen_{i}'].value
                        s = fit_out.params[f'sig_{i}'].value
                        g = fit_out.params[f'gam_{i}'].value
                    else:
                        a = popt[4*i + 0]; c = popt[4*i + 1]; s = popt[4*i + 2]; g = popt[4*i + 3]
                    comp = voigt_profile(x, a, c, s, g)
                    ax.plot(x, comp + (b + m * x)*0 + np.median(y)*0, '--', alpha=0.8)  # plotted as isolated components
                except Exception:
                    pass
    
            ax.set_ylabel('Summed intensity')
            ax.legend()
            ax.set_title('Multi-Voigt fit to dispersion-integrated profile')
    
            axr = axes[1]
            axr.plot(x, y - model_y, label='residual')
            axr.axhline(0, color='k', lw=0.5)
            axr.set_xlabel('Pixel (dispersion axis)')
            axr.set_ylabel('Residual')
    
            plt.tight_layout()
            plt.show()
    
            # Pixel->Energy visual check
            plt.figure(figsize=(6,4))
            plt.errorbar(pixels, energies, xerr=pixel_errs, fmt='o', label='fitted peaks')
            xs = np.linspace(pixels.min()-5, pixels.max()+5, 100)
            plt.plot(xs, m*xs + b, '-', label=f'Energy = {m:.6f}*pixel + {b:.6f}')
            plt.xlabel('Pixel')
            plt.ylabel('Energy (keV)')
            plt.legend()
            plt.title('Pixel -> Energy calibration')
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.title("Check peak positions")
            plt.plot(m*range(len(y_s))+b, y_s)
            for i, line in enumerate(peakenergies_list):
                plt.axvline(line, label=f"{element_list[i]} {line_list[i]}", color=colors[i], lw=0.5)
            plt.legend()
            plt.yscale('log')
            plt.xlabel("Energy (keV)")
            plt.ylabel("Integrated intensity")
            plt.show()

        result = {
            'fitted_peaks': fitted_peaks,
            'pixel_to_energy': pixel_to_energy,
            'fit_full': fit_out
        }
        return result

    def manual_peak_picker(self, peakenergies_list, gaussian_blur=1.0, plot=True):
        """
        Manually select peak positions for calibration by clicking on the plot.
        
        
        Parameters
        ----------
        peakenergies_list : list[float]
        Known energies of the peaks in keV.
        gaussian_blur : float
        Sigma for Gaussian blur applied to smooth the profile.
        plot : bool
        Whether to show plots during selection and fitting.
        
        
        Returns
        -------
        calib_dict : dict
        Dictionary containing calibration parameters.
        """
        
        
        if self.data is None:
            raise ValueError("No image data loaded.")
        
        
        profile = np.sum(self.data, axis=0)
        profile_smooth = gaussian_filter1d(profile, sigma=gaussian_blur)
        
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(profile_smooth, label='Smoothed profile')
        ax.set_title('Click to select peak positions (close plot when done)')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Integrated intensity')
        ax.legend()
        
        
        # Manual selection of peak positions
        picked_points = plt.ginput(len(peakenergies_list), timeout=-1)
        plt.close(fig)
        
        
        selected_pixels = np.array([pt[0] for pt in picked_points])
        
        
        # Fit a linear calibration (pixel to energy)
        energies = np.array(sorted(peakenergies_list)) # Sort energies ascending
        pixels_sorted = np.sort(selected_pixels)
        
        
        # Perform linear fit (Energy = slope * pixel + intercept)
        slope, intercept = np.polyfit(pixels_sorted, energies, deg=1)
        
        
        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(selected_pixels, energies, 'o', label='Selected Peaks')
            pixel_fit = np.linspace(min(pixels_sorted)-10, max(pixels_sorted)+10, 100)
            plt.plot(pixel_fit, slope * pixel_fit + intercept, '-', label=f'Fit: E = {slope:.4f}*px + {intercept:.4f}')
            plt.xlabel('Pixel')
            plt.ylabel('Energy (keV)')
            plt.legend()
            plt.title('Manual Pixel → Energy Calibration')
            plt.grid(True)
            plt.show()
            
            
        calib_dict = {
        'slope_keV_per_px': float(slope),
        'intercept_keV': float(intercept),
        'selected_pixels': selected_pixels.tolist(),
        'energies_keV': energies.tolist(),
        }
        
        
        return calib_dict

    
    def save_pixel_to_energy(self, calib_dict, save_dir="."):
        """
        Save pixel->energy calibration (slope/intercept) to a JSON file.
        """
        filepath = os.path.join(save_dir, f"pixel_to_energy_{self.fssr}.json")
        with open(filepath, "w") as f:
            json.dump(calib_dict, f, indent=2)
        print(f"✅ Calibration saved to {filepath}")




        
    
        


