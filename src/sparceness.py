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
import matplotlib.cm as cm
from scipy import signal
cmap = cm.jet

mouse = 56166
sessions = [1,2]

## load source extracted calcium traces condition SESSION 1
file_directory = os.environ['PROJECT_DIR'] + 'neural_analysis/data/calcium_activity/'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'neural_analysis/data/timeline/'
behaviour_dir = os.environ['PROJECT_DIR'] + 'calcium_imaging_behaviour/data/scoring_time_vector/'

decoding_v = 1
motion_correction_v = 100 ### means everything was aligned
alignment_v = 1
equalization_v = 0
source_extraction_v = 1
component_evaluation_v = 1
registration_v = 1

bin_vector = [1, 2, 5, 10, 25, 50, 75, 100, 150]
bin_vector = np.arange(1, 150, 1)
sf = 10
figure, axes = plt.subplots(3,1)
## session 1 files
plotting = 0
for session in sessions:
    file_name_session_1 = 'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_1_v'+ f'{decoding_v}'+'.4.'+f'{100}'+\
                          '.'+f'{alignment_v}'+'.'+ f'{equalization_v}' +'.' + f'{source_extraction_v}'+'.' + \
                          f'{component_evaluation_v}' +'.'+ f'{registration_v}' + '.npy'
    time_file_session_1 =  'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_1_v'+ f'{decoding_v}'+'.4.'+f'{1}'+\
                          '.'+f'{0}'+ '.pkl'
    beh_file_name_1 = 'mouse_'+f'{mouse}'+'_session_'+f'{session}'+'.npy'

    ##session 1
    activity = np.load(file_directory + file_name_session_1)
    timeline_file= open(timeline_file_dir + time_file_session_1,'rb')
    timeline_info = pickle.load(timeline_file)
    timeline_1 = np.zeros(42+1)
    for i in range(42):
        timeline_1[i] = timeline_info[i][1]
    timeline_1[42] = activity.shape[1]
    ### do analysis corr, PCA
    ## normalize activity within trial and for each neuron
    activity_normalized = np.zeros((activity.shape))
    for j in range(activity.shape[0]):
        for i in range(0,len(timeline_1)-1):
            activity_segment = activity[j,int(timeline_1[i]):int(timeline_1[i+1])]
            activity_segment = activity_segment - min(activity_segment)
            if max(activity_segment):
                activity_segment_normalized = activity_segment/max(activity_segment)
                activity_normalized[j,int(timeline_1[i]):int(timeline_1[i+1])] =activity_segment_normalized
    neural_activity1 = activity[1:,:]
    corr_matrix1 = stats.corr_matrix(neural_activity = neural_activity1)
    neural_activity1 = activity_normalized[1:,:]

    sparceness_mean = []
    sparceness_std = []

    for re_sf in bin_vector:
        reshape_neural_activity = np.reshape(neural_activity1[:,:int(int(neural_activity1.shape[1]/re_sf )*re_sf) ],
                                              (neural_activity1 .shape[0],int(neural_activity1.shape[1]/re_sf ),re_sf))
        resample_neural_activity = np.mean(reshape_neural_activity,axis= 2)
        denominator = np.mean(np.multiply(resample_neural_activity,resample_neural_activity),axis=1)
        nominator = np.multiply(np.mean(resample_neural_activity,axis=1),np.mean(resample_neural_activity,axis=1))
        x = np.where(denominator)
        denominator = denominator[x]
        nominator = nominator[x]
        sparceness = np.divide(nominator,denominator)
        sparceness_mean.append(np.mean(sparceness))
        sparceness_std.append(np.std(sparceness))

    axes[plotting].plot(bin_vector, sparceness_mean)
    plotting = plotting +1

figure.show()

axes[0].set_title('OVERLAPPING')
axes[1].set_title('STABLE')
axes[2].set_title('RANDOM')
axes[2].set_xlabel('Bin size [frames]')
axes[0].set_ylabel(' <f>^2  /  <f^2>  ')
axes[1].set_ylabel(' <f>^2  /  <f^2>  ')
axes[2].set_ylabel('  <f>^2 / < <f^2>   ')

figure.show()
figure.savefig('/home/melisa/Documents/neural_analysis/data/process/figures/sparcity_mouse_'+f'{mouse}'+'.png')
