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

#%% In days correlation matrix, data loading
session = 4
if session == 1:
    task = 'OVERLAPPING'
else:
    if session == 2:
        task = 'STABLE'
    else:
        task = 'RANDOM'

file_name_session_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_1_v' + f'{decoding_v}' + '.4.' + f'{100}' + \
                      '.' + f'{alignment_v}' + '.' + f'{equalization_v}' + '.' + f'{source_extraction_v}' + '.' + \
                      f'{component_evaluation_v}' + '.' + f'{registration_v}' + '.npy'

time_file_session_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_1_v' + f'{decoding_v}' + '.1.' + f'{1}' + \
                      '.' + f'{0}' + '.pkl'

beh_file_name_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_event_' + f'{re_sf}' + '.npy'

##load activity and timeline
activity = np.load(file_directory + file_name_session_1)
timeline_file = open(timeline_file_dir + time_file_session_1, 'rb')
timeline_info = pickle.load(timeline_file)
##normalize neural activity
neural_activity1, timeline_1 = stats.normalize_neural_activity(activity=activity, timeline=timeline_info)
##downsample neural activity
resample_neural_activity_mean, resample_neural_activity_std = stats.resample_matrix(neural_activity=neural_activity1,
                                                                                    re_sf=re_sf)
## LOAD BEHAVIOUR
behaviour = np.load(behaviour_dir + beh_file_name_1)
#c = np.linspace(0, 20, len(behaviour))
neural_activity_new= []
testing = []
reshape_behaviour = np.reshape(behaviour[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
resample_beh1 = np.reshape(scipy.stats.mode(reshape_behaviour,axis=1)[0],reshape_behaviour.shape[0])
resample_timeline = timeline_1/re_sf

#%% in days correlation matrix, data reshape

neural_activity1_days = []
time_length = np.diff(resample_timeline)
for i in range(4):
    day_matrix = np.zeros((neural_activity1.shape[0],int(np.sum(time_length[i*10:(i+1)*10:2]))))
    start_time = 0
    for j in range(0,10,2):
        trial = i*10 + j
        day_matrix[:,start_time:start_time+ int(time_length[trial])] = \
            neural_activity1[:,int(resample_timeline[trial]):int(resample_timeline[trial]) + int(time_length[trial])]
        start_time = start_time + int(time_length[trial])
    neural_activity1_days.append(day_matrix)

neural_activity1_resting_days = []
time_length = np.diff(resample_timeline)
for i in range(4):
    day_matrix = np.zeros((neural_activity1.shape[0],int(np.sum(time_length[i*10+1:(i+1)*10+1:2]))))
    start_time = 0
    for j in range(1,10,2):
        trial = i*10 + j
        day_matrix[:,start_time:start_time+ int(time_length[trial])] = \
            neural_activity1[:,int(resample_timeline[trial]):int(resample_timeline[trial]) + int(time_length[trial])]
        start_time = start_time + int(time_length[trial])
    neural_activity1_resting_days.append(day_matrix)

neural_activity1_resting_testing = neural_activity1[:,int(resample_timeline[-2]):int(resample_timeline[-1])]
neural_activity1_testing = neural_activity1[:,int(resample_timeline[-3]):int(resample_timeline[-2])]

#%%
corr_matrix_days = []
for i in range(4):
    corr_matrix_days.append(stats.corr_matrix(neural_activity = neural_activity1_days[i]))
    #corr_matrix_days.append(stats.cov_matrix(neural_activity = neural_activity1_days[i]))

corr_matrix_days.append(stats.corr_matrix(neural_activity = neural_activity1_testing))
#corr_matrix_days.append(stats.cov_matrix(neural_activity = neural_activity1_testing))

corr_matrix_resting_days = []
for i in range(4):
    corr_matrix_resting_days.append(stats.corr_matrix(neural_activity = neural_activity1_resting_days[i]))
    #corr_matrix_resting_days.append(stats.cov_matrix(neural_activity = neural_activity1_resting_days[i]))
corr_matrix_resting_days.append(stats.corr_matrix(neural_activity = neural_activity1_resting_testing))
#corr_matrix_resting_days.append(stats.cov_matrix(neural_activity = neural_activity1_resting_testing))

#%% In days correlation matrix, plotting

correlation_path = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/' \
                   'correlation_mouse_'+f'{mouse}'+'_session_'+f'{session}'+\
                             '_binsize_'+f'{re_sf}'+'.png'
figs.plot_correlation_statistics_learning(corr_matrix1=corr_matrix_days, corr_matrix2=corr_matrix_resting_days,
                                     path_save=correlation_path, title=task)

#%% Correlation in trials, data reshape
neural_activity1_days = []
time_length = np.diff(timeline_1)
for i in range(0,42,2):
    trial_matrix = neural_activity1[:,int(timeline_1[i]):int(timeline_1[i]) + int(time_length[i])]
    neural_activity1_days.append(trial_matrix)

neural_activity1_resting_days = []
for i in range(1,42,2):
    trial_matrix = neural_activity1[:,int(timeline_1[i]):int(timeline_1[i]) + int(time_length[i])]
    neural_activity1_resting_days.append(trial_matrix)

#%%
corr_matrix_days = []
for i in range(21):
    corr_matrix_days.append(stats.corr_matrix(neural_activity = neural_activity1_days[i]))
    #corr_matrix_days.append(stats.cov_matrix(neural_activity = neural_activity1_days[i]))

corr_matrix_resting_days = []
for i in range(21):
    corr_matrix_resting_days.append(stats.corr_matrix(neural_activity = neural_activity1_resting_days[i]))
    #corr_matrix_resting_days.append(stats.cov_matrix(neural_activity = neural_activity1_resting_days[i]))

#%%
correlation_path = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/' \
                   'correlation_mouse_trials_'+f'{mouse}'+'_session_'+f'{session}'+'_binsize_'+f'{re_sf}'+'.png'

figs.plot_correlation_statistics_trials(corr_matrix1=corr_matrix_days, corr_matrix2=corr_matrix_resting_days,
                                     path_save=correlation_path, title=task)
