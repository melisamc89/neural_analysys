
"""
@author: Melisa
Created on Wed 27 mei

Create a video with frames of source extraded correleation images and neurons
for multiple trials.

"""
import os
import numpy as np
import pandas as pd
import pickle
import math
import matplotlib.pyplot as plt

import src.configuration
import caiman as cm
from caiman.base.rois import com
from caiman.source_extraction.cnmf.cnmf import load_CNMF

mouse = 32364

decoding_v = 1
motion_correction = 100
alignment_version = 1
equalization_version = 0
source_extraction_version = 1
component_evaluation_version = 1

file_directory1 = os.environ['PROJECT_DIR'] + 'neural_analysis/data/correlation/'
file_directory2 = os.environ['PROJECT_DIR'] + 'neural_analysis/data/sources/'
figure_path = os.environ['PROJECT_DIR'] +'neural_analysis/data/process/figures/corr_images/'


for session in [1,2]:
    for trial in range(1,22):
        figure, axes = plt.subplots(2,2)
        for i in range(2):
            for j in range(2):
                cropping_v = i*2+j+1
                file_name_corr_image = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_'+f'{trial}'+'_R_v' + f'{decoding_v}' + \
                                  '.'+f'{cropping_v}' + '.'+ f'{100}' + '.1_gSig_5.npy'
                corr_image = np.load(file_directory1 + file_name_corr_image)
                file_name_components = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_'+f'{trial}'+'_R_v' + f'{decoding_v}' + \
                                  '.'+f'{cropping_v}' + '.'+ f'{100}' + '.1.0.1.1.hdf5'
                cnm = load_CNMF(file_directory2 + file_name_components)
                axes[i,j].imshow(corr_image)
                coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(corr_image), 0.2, 'max')
                for c in coordinates:
                    v = c['coordinates']
                    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
                    axes[i,j].plot(*v.T, c='b')
        figure.suptitle('Trial = ' + f'{trial}' + ' Rest')
        figure_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_'+f'{trial}'+'_R_v' + f'{decoding_v}' + \
                          '.'+f'{1}' + '.'+ f'{100}' + '.1.0.1.1.png'
        figure.savefig(figure_path + figure_name)
