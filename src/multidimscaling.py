'''
Created on Thrus 20 Feb 2020
Author: Melisa

Multidimentional scaling in raw data and in pca data
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pickle
import configuration
import general_statistics as stats
import matplotlib.cm as cm
import figures as figs
from sklearn.decomposition import PCA
import scipy
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
cmap = cm.jet


mouse = 56165             ### mouse number id
decoding_v = 1            ## decoding version, normaly equal to one
motion_correction_v = 100 ### 100 means everything was aligned, 200 means it was also equalized
alignment_v = 1           ## alignment version
equalization_v = 0        ## equalization version
source_extraction_v = 1   ## source extraction version
component_evaluation_v = 1 ## component evaluation version
registration_v = 1        ## registration version
sf = 10                   ## sampling frequency of the original signal
re_sf= 50                 ## value of resampling

sessions = [1,2,4]       ## sessions for this particular mouse
session_now = 2          ## session that will run

## define task for plotting. This will cahnge for other mice!!!!
if session_now == 1:
    task = 'OVERLAPPING'
else:
    if session_now == 2:
        task = 'STABLE'
    else:
        task = 'RANDOM'

file_directory = os.environ['PROJECT_DIR'] + 'neural_analysis/data/calcium_activity/'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'neural_analysis/data/timeline/'
behaviour_dir = os.environ['PROJECT_DIR'] + 'calcium_imaging_behaviour/data/scoring_time_vector/'
objects_dir= os.environ['PROJECT_DIR'] + 'calcium_imaging_behaviour/data/object_positions/'
figures_path = os.environ['PROJECT_DIR'] +'neural_analysis/data/process/figures/MDS/'

#%%
# define all relevant files names
session = session_now
file_name_session = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_1_v' + f'{decoding_v}' + '.4.' + f'{100}' + \
                      '.' + f'{alignment_v}' + '.' + f'{equalization_v}' + '.' + f'{source_extraction_v}' + '.' + \
                      f'{component_evaluation_v}' + '.' + f'{registration_v}' + '.npy'
time_file_session = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_1_v' + f'{decoding_v}' + '.1.' + f'{1}' + \
                      '.' + f'{0}' + '.pkl'
beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_event_' + f'{re_sf}' + '.npy'

##load activity and timeline
activity = np.load(file_directory + file_name_session)
timeline_file = open(timeline_file_dir + time_file_session, 'rb')
timeline_info = pickle.load(timeline_file)
##normalize neural activity
neural_activity, timeline = stats.normalize_neural_activity(activity=activity, timeline=timeline_info)
##downsample neural activity
resample_neural_activity_mean, resample_neural_activity_std = stats.resample_matrix(neural_activity=neural_activity,
                                                                                    re_sf=re_sf)

#%%

embedding = MDS(n_components=3)
neural_activity_transformed = embedding.fit_transform(resample_neural_activity_mean.T)

file_name = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/MDS/mouse_' + f'{mouse}' + \
            '_session_' + f'{session}' + '_binsize_'+f'{re_sf}'+'.npy'
np.save(file_name, neural_activity_transformed)
