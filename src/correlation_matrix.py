'''
Created on Wed 19 Feb 2020
Author: Melisa

Checking whether correlation matrix for different exploring conditions change

'''
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pickle
import configuration
import general_statistics as stats
import figures as figs
import matplotlib.cm as cm
from matplotlib import colors
from scipy import signal
import scipy
cmap = cm.jet

mouse = 56165
decoding_v = 1
motion_correction_v = 100 ### means everything was aligned
alignment_v = 1
equalization_v = 0
source_extraction_v = 1
component_evaluation_v = 1
registration_v = 1

## load source extracted calcium traces condition SESSION 1
file_directory = os.environ['PROJECT_DIR'] + 'neural_analysis/data/calcium_activity/'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'neural_analysis/data/timeline/'
behaviour_dir = os.environ['PROJECT_DIR'] + 'calcium_imaging_behaviour/data/scoring_time_vector/'

sf = 10
re_sf= 20

#%% Correlation in conditions
## session files
correlation_matrix = []
task_list = []
for session in [1,2,4]:

    if session == 1:
        task = 'OVERLAPPING'
    else:
        if session == 2:
            task = 'STABLE'
        else:
            task = 'RANDOM'

    file_name_session_1 = 'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_1_v'+ f'{decoding_v}'+'.4.'+f'{100}'+\
                          '.'+f'{alignment_v}'+'.'+ f'{equalization_v}' +'.' + f'{source_extraction_v}'+'.' + \
                          f'{component_evaluation_v}' +'.'+ f'{registration_v}' + '.npy'
    time_file_session_1 =  'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_1_v'+ f'{decoding_v}'+'.1.'+f'{1}'+\
                          '.'+f'{0}'+ '.pkl'
    beh_file_name_1 = 'mouse_'+f'{mouse}'+'_session_'+f'{session}'+'_event_'+f'{re_sf}'+'.npy'

    ##load activity and timeline
    activity = np.load(file_directory + file_name_session_1)
    timeline_file= open(timeline_file_dir + time_file_session_1,'rb')
    timeline_info = pickle.load(timeline_file)
    ##normalize neural activity
    neural_activity1, timeline_1 = stats.normalize_neural_activity(activity=activity, timeline=timeline_info)
    ##downsample neural activity
    resample_neural_activity_mean, resample_neural_activity_std = stats.resample_matrix(neural_activity=neural_activity1,re_sf=re_sf)
    correlation_matrix.append(stats.corr_matrix(neural_activity = resample_neural_activity_mean))
    task_list.append(task)

#%% plotting corr matrix for all conditions
correlation_fig_path =  os.environ['PROJECT_DIR'] + 'neural_analysis/data/process/figures/' \
                                                    'correlation_matrix_mouse_'+f'{mouse}'+'_session_'+f'{session}'+\
                        '_binsize_'+f'{re_sf}'+'.png'
figs.plot_correlation_matrix_conditions(matrix_list = correlation_matrix, save_path = correlation_fig_path,
                                        title = 'CorrelationMatrix' , conditions = task_list)
