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
from scipy import signal
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
cmap = cm.jet

mouse = 56165

## load source extracted calcium traces condition SESSION 1
file_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity/'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'
behaviour_dir = '/home/sebastian/Documents/Melisa/calcium_imaging_behaviour/data/scoring_time_vector/'

decoding_v = 1
motion_correction_v = 100 ### means everything was aligned
alignment_v = 1
equalization_v = 0
source_extraction_v = 1
component_evaluation_v = 1
registration_v = 1


## session 1 files
session = 2

file_name_session_1 = 'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_1_v'+ f'{decoding_v}'+'.4.'+f'{100}'+\
                      '.'+f'{alignment_v}'+'.'+ f'{equalization_v}' +'.' + f'{source_extraction_v}'+'.' + \
                      f'{component_evaluation_v}' +'.'+ f'{registration_v}' + '.npy'
time_file_session_1 =  'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_1_v'+ f'{decoding_v}'+'.1.'+f'{1}'+\
                      '.'+f'{0}'+ '.pkl'
beh_file_name_1 = 'mouse_'+f'{mouse}'+'_session_'+f'{3}'+'.npy'

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
corr_matrix1_1 = stats.corr_matrix(neural_activity = neural_activity1)
eigenvalues1, eigenvectors1 = stats.compute_PCA(corr_matrix = corr_matrix1)
proj1 = stats.PCA_projection(neural_activity=neural_activity1, eigenvalues=eigenvalues1,
                            eigenvectors=eigenvectors1, n_components=70)

## LOAD BEHAVIOUR
behaviour = np.load(behaviour_dir + beh_file_name_1)

### downsample
sf = 10
new_sf = 0.1
re_sf = int(sf/new_sf)
#mat_proj1 = np.reshape(proj1[:,:int(int(proj1.shape[1]/re_sf )*re_sf) ],(proj1.shape[0],int(proj1.shape[1]/re_sf ),re_sf))
#resample_proj1 = np.mean(mat_proj1,axis = 2)
#std_resample_proj1 = np.std(mat_proj1, axis = 2)
resample_proj1= signal.resample(proj1,int(proj1.shape[1]/re_sf),axis = 1)

embedding = MDS(n_components=3)
proj1_transformed = embedding.fit_transform(resample_proj1.T)

file_name = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/MDS/mouse_' + f'{mouse}' + '_session_' + f'{session}' + '.npy'
np.save(file_name, proj1_transformed)
