
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

mouse = 401714

decoding_v = 1
motion_correction = 20
alignment_version = 3
equalization_version = 0
source_extraction_version = 1
component_evaluation_version = 0

#file_directory1 = os.environ['PROJECT_DIR'] + 'neural_analysis/data/correlation/'
#file_directory2 = os.environ['PROJECT_DIR'] + 'neural_analysis/data/sources/'
#file_directory3 = os.environ['PROJECT_DIR'] + 'neural_analysis/data/registration/'
file_directory1 = os.environ['PROJECT_DIR'] + 'calcium_imaging_analysis/data/interim/source_extraction/session_wise/meta/corr/'
file_directory2 = os.environ['PROJECT_DIR'] + 'calcium_imaging_analysis/data/interim/component_evaluation/session_wise/main/'
file_directory3 = os.environ['PROJECT_DIR'] + 'neural_analysis/data/registration/'

figure_path = os.environ['PROJECT_DIR'] +'neural_analysis/data/process/figures/corr_images/'


for session in [1]:
    for trial in [1,6,11,16]:
        figure, axes = plt.subplots(2,2)
        for i in range(2):
            for j in range(2):
                cropping_v = i*2+j+1
                file_name_corr_image = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_'+f'{trial}'+'_v' + f'{decoding_v}' + \
                                  '.'+f'{cropping_v}' + '.'+ f'{20}' + '.3_gSig_5.npy'
                corr_image = np.load(file_directory1 + file_name_corr_image)
                file_name_components = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_'+f'{trial}'+'_v' + f'{decoding_v}' + \
                                  '.'+f'{cropping_v}' + '.'+ f'{20}' + '.3.0.1.1.hdf5'
                cnm = load_CNMF(file_directory2 + file_name_components)

                #file_registration_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + f'{1}' + '_v' + f'{decoding_v}' + \
                #                         '.' + f'{cropping_v}' + '.' + f'{20}' + '.1.0.1.1.1.pkl'

                #registration_file = open(file_directory3 + file_registration_name, 'rb')
                #registration_info = pickle.load(registration_file)

                axes[i,j].imshow(corr_image)
                coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(corr_image), 0.2, 'max')
                for c in coordinates:
                    v = c['coordinates']
                    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
                    axes[i,j].plot(*v.T, c='b')
        figure.suptitle('Trial = ' + f'{trial}' )
        figure_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_'+f'{trial}'+'_R_v' + f'{decoding_v}' + \
                          '.'+f'{1}' + '.'+ f'{20}' + '.3.0.1.0.png'
        figure.savefig(figure_path + figure_name)
