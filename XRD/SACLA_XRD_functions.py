# SACLA XRD functions
# A series of functions to retreive XRD images and plot them
# creator: tom stevens
# created: 2025/11/29

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_XRD_image(run_number, experiment_path = "/work/kmiyanishi/userdata/2025b/fitzgerald2025b/renamed_files", plot=False):
    
    fpd1_folderpath = os.path.join(experiment_path, 'fpd1')
    
    dark_path_fpd1 = os.path.join(fpd1_folderpath, f"{run_number}_fpd1-b.tif")
    dark_im_fpd1 = np.array(Image.open(dark_path_fpd1), dtype=float)
    
    file_path_fpd1 = os.path.join(fpd1_folderpath, f"{run_number}_fpd1.tif")
    im_fpd1 = np.array(Image.open(file_path_fpd1), dtype=float)
    
    dark_subtracted_im_fpd1 = im_fpd1 - dark_im_fpd1
    
    if plot:
        plt.pcolormesh(im_fpd1)
        plt.title('Raw image')
        plt.show()
        plt.pcolormesh(dark_im_fpd1)
        plt.title('Dark image')
        plt.show()
        plt.pcolormesh(dark_subtracted_im_fpd1)
        plt.title('Dark subtracted image')
        plt.show()
    return dark_subtracted_im_fpd1

def get_XRD_shot_preshot(shot_run_number, preshot_run_number, experiment_path = "/work/kmiyanishi/userdata/2025b/fitzgerald2025b/renamed_files", plot=False):
    fpd1_folderpath = os.path.join(experiment_path, 'fpd1')
    
    dark_path_fpd1_shot = os.path.join(fpd1_folderpath, f"{shot_run_number}_fpd1-b.tif")
    dark_im_fpd1_shot = np.array(Image.open(dark_path_fpd1_shot), dtype=float)
    
    file_path_fpd1_shot = os.path.join(fpd1_folderpath, f"{shot_run_number}_fpd1.tif")
    im_fpd1_shot = np.array(Image.open(file_path_fpd1_shot), dtype=float)
    
    dark_subtracted_im_fpd1_shot = im_fpd1_shot - dark_im_fpd1_shot

    dark_path_fpd1_preshot = os.path.join(fpd1_folderpath, f"{preshot_run_number}_fpd1-b.tif")
    dark_im_fpd1_preshot = np.array(Image.open(dark_path_fpd1_preshot), dtype=float)
    
    file_path_fpd1_preshot = os.path.join(fpd1_folderpath, f"{preshot_run_number}_fpd1.tif")
    im_fpd1_preshot = np.array(Image.open(file_path_fpd1_preshot), dtype=float)
    
    dark_subtracted_im_fpd1_preshot = im_fpd1_preshot - dark_im_fpd1_preshot
    
    if plot:
        plt.pcolormesh(im_fpd1_preshot)
        plt.title('Raw image preshot')
        plt.show()
        plt.pcolormesh(dark_im_fpd1_preshot)
        plt.title('Dark image preshot')
        plt.show()
        plt.pcolormesh(dark_subtracted_im_fpd1_preshot)
        plt.title('Dark subtracted image preshot')
        plt.show()
        plt.pcolormesh(im_fpd1_shot)
        plt.title('Raw image shot')
        plt.show()
        plt.pcolormesh(dark_im_fpd1_shot)
        plt.title('Dark image shot')
        plt.show()
        plt.pcolormesh(dark_subtracted_im_fpd1_shot)
        plt.title('Dark subtracted image shot')
        plt.show()
    return dark_subtracted_im_fpd1_shot, dark_subtracted_im_fpd1_preshot
    

def plot(fpd1_CAKE, fpd1_INT, run_number = None):
    # This is all to plot the below graphs
    plt.figure(figsize = (25, 8))
    if run_number is not None:
        plt.title(f"Run: {run_number}")
    # Plot of the cakes
    plt.subplot(121)
    plt.pcolormesh(fpd1_CAKE.radial, fpd1_CAKE.azimuthal, np.log10(fpd1_CAKE.intensity), cmap = 'plasma')
    plt.xlabel('Theta (Degrees) or Q ($\\AA^{-1}$)')
    plt.ylabel('Phi (Degrees)')
    
    # Plot of the 1D lineouts
    plt.subplot(122)
    plt.plot(fpd1_INT.radial, fpd1_INT.intensity)
    # plt.semilogy()
    plt.xlabel('Theta (Degrees) or Q ($\\AA^{-1}$)')
    plt.ylabel('Arbitrary Intensity')
    # plt.xlim(15, 20)
    plt.show()

def plot_with_preshot(fpd1_CAKE, fpd1_INT, fpd1_INT_preshot, offset = 20, run_number = None):
    # This is all to plot the below graphs
    plt.figure(figsize = (25, 8))
    if run_number is not None:
        plt.title(f"Run: {run_number}")
    # Plot of the cakes
    plt.subplot(121)
    plt.pcolormesh(fpd1_CAKE.radial, fpd1_CAKE.azimuthal, np.log10(fpd1_CAKE.intensity), cmap = 'plasma')
    plt.xlabel('Theta (Degrees) or Q ($\\AA^{-1}$)')
    plt.ylabel('Phi (Degrees)')
    
    # Plot of the 1D lineouts
    plt.subplot(122)
    plt.plot(fpd1_INT.radial, fpd1_INT.intensity)
    plt.plot(fpd1_INT_preshot.radial, fpd1_INT_preshot.intensity+offset)
    # plt.semilogy()
    plt.xlabel('Theta (Degrees) or Q ($\\AA^{-1}$)')
    plt.ylabel('Arbitrary Intensity')
    # plt.xlim(15, 20)
    plt.show()

def process_many_for_waterfall(pyFAI_dict, previously_processed_runs, run_numbers, good_runs, export=False, offset=20, plot_all=False, plot_good=False):
    fpd1_poni, fpd1_mask, fpd1_map, npt_rad, npt_azim, radial_range, azim_range, units = pyFAI_dict['poni'], pyFAI_dict['mask'], pyFAI_dict['map'], pyFAI_dict['npt_rad'], pyFAI_dict['npt_azim'], pyFAI_dict['radial_range'], pyFAI_dict['azim_range'], pyFAI_dict['units']
    processed_runs = previously_processed_runs
    ims_fpd1 = False
    fpd1_INTs = False
    fpd1_INTs_no_offset = False
    new_runs = {}
    for idx, run_number in enumerate(run_numbers):
        try:
            fpd1_CAKE, fpd1_INT = processed_runs[run_number]['CAKE'], processed_runs[run_number]['INT']
        except:
            im_fpd1 = get_XRD_image(run_number, plot=False)
        
            if isinstance(ims_fpd1, bool):
                ims_fpd1 = im_fpd1
            else:
                ims_fpd1 = np.dstack((ims_fpd1, im_fpd1))
        
            # pyFAI integration for VAREX 1
            fpd1_CAKE = fpd1_poni.integrate2d_ng(im_fpd1 * fpd1_map, npt_rad, npt_azim, azimuth_range=azim_range, radial_range=radial_range, correctSolidAngle=True, polarization_factor=1, unit=units, mask=fpd1_mask)
            fpd1_INT = fpd1_poni.integrate1d_ng(im_fpd1 * fpd1_map, npt_rad, azimuth_range=azim_range, radial_range=radial_range, correctSolidAngle=True, polarization_factor=1, unit=units, mask=fpd1_mask)
    
            processed_runs[run_number] = {}
            processed_runs[run_number]['CAKE'] = fpd1_CAKE
            processed_runs[run_number]['INT'] = fpd1_INT
            new_runs[run_number] = processed_runs[run_number]
        
        if plot_all:
            plot(fpd1_CAKE, fpd1_INT, run_number=run_number)
        elif plot_good:
            if run_number in good_runs:
                plot(fpd1_CAKE, fpd1_INT, run_number=run_number)
            elif run_number in new_runs:
                plot(fpd1_CAKE, fpd1_INT, run_number=run_number)
        else:
            if run_number in new_runs:
                plot(fpd1_CAKE, fpd1_INT, run_number=run_number)
    
        if isinstance(fpd1_INTs, bool):
            fpd1_INTs = fpd1_INT.intensity + (idx*offset)
            fpd1_INTs_no_offset = fpd1_INT.intensity
        else:
            fpd1_INTs = np.vstack((fpd1_INTs, fpd1_INT.intensity + (idx*offset)))
            fpd1_INTs_no_offset = np.vstack((fpd1_INTs_no_offset, fpd1_INT.intensity))
    
        if export:
            lineout_exportrun = np.stack((fpd1_INT.radial, fpd1_INT.intensity),axis=1)
            np.savetxt(f'xys/{run_number}.xy',lineout_exportrun)
    return processed_runs, new_runs, fpd1_INTs, fpd1_INTs_no_offset, fpd1_INT