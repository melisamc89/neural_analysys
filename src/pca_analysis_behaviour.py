'''
Created on Tue 03 Mar 2020
Author:Melisa
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


mouse = 56165             ### mouse number id
decoding_v = 1            ## decoding version, normaly equal to one
motion_correction_v = 100 ### 100 means everything was aligned, 200 means it was also equalized
alignment_v = 1           ## alignment version
equalization_v = 0        ## equalization version
source_extraction_v = 1   ## source extraction version
component_evaluation_v = 1 ## component evaluation version
registration_v = 1        ## registration version
sf = 10                   ## sampling frequency of the original signal
re_sf= 20                 ## value of resampling

sessions = [1,2,4]       ## sessions for this particular mouse
session_now = 1          ## session that will run
n_components = 10         ## number of projected components

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
figures_path = os.environ['PROJECT_DIR'] +'neural_analysis/data/process/figures/pca/'

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
## run pca analysis on covariance matrix
cov_matrix = stats.cov_matrix(neural_activity = neural_activity) ## compute covariance matrix
eigenvalues, eigenvectors = stats.compute_PCA(corr_matrix = cov_matrix) ## run eigenvalues and eigenvectors analysis
projection = stats.PCA_projection(neural_activity=neural_activity, eigenvalues=eigenvalues,
                            eigenvectors=eigenvectors, n_components=n_components) ## project to n_componets

#%%
## load behavioural file, downsample it and separate different parts of the experiment
behaviour = np.load(behaviour_dir + beh_file_name)
# resample neural activity and behavioural vector
reshape_behaviour = np.reshape(behaviour[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
resample_beh = np.reshape(scipy.stats.mode(reshape_behaviour,axis=1)[0],reshape_behaviour.shape[0])
resample_timeline = timeline/re_sf
color = np.linspace(0, 20, len(resample_beh))

#%%
## separate different behavioural parts of the experiment (in neural activity and in projected activity as well)

# define a variable with only training data (remove testing)
training_data = resample_neural_activity_mean[:, :int(resample_timeline[40])] ## neural activity in training
training_data_beh = resample_beh[:int(resample_timeline[40])]        ## bahavioural vector in training
testing_data = resample_neural_activity_mean[:,int(resample_timeline[40]):]  ## neural activity in testing
testing_data_beh= resample_beh[int(resample_timeline[40]):]   ## behavior in testing

projection_data = projection[:n_components, :int(resample_timeline[40])]
projection_data_testing = projection[:n_components, int(resample_timeline[40]):]

neural_activity_training = [] ## list containing neural activity for different bahavioural parts
neural_activity_testing = []  ## list containing different behaviours in the testing trial

projection_training = []    ## list of projected activity in pca during training
projection_testing = []     ## list of projected activity in pca during testing

color_training=[]           ## arrange of colors. Will be necesary to have a common color criteria when plotting
color_testing = []
for i in range(6):     ## 6 different behaviours defined here
    neural_activity_training.append(training_data[:,np.where(training_data_beh== i)])
    neural_activity_testing.append(testing_data[:,np.where(testing_data_beh == i)])
    projection_training.append(projection_data[:,np.where(training_data_beh == i)])
    projection_testing.append(projection_data_testing[:,np.where(testing_data_beh == i)])
    color_training.append(color[np.where(training_data_beh ==i)])
    color_testing.append(color[np.where(testing_data_beh==i)])

#%%
pca_behavioral_path = figures_path + 'pca_behaviour_training_mouse_'+f'{mouse}' +\
                      '_session_'+f'{session}_binsize_'+f'{re_sf}'+'.png'
figs.plot_pca_behavioral_representation(components_list = projection_training, color = color_training,
                                        title = task + ' Training', path_save = pca_behavioral_path)

#pca_behavioral_path_test = figures_path + 'pca_behaviour_testing_mouse_'+f'{mouse}' +'_session_'+f'{session}_binsize_'+f'{re_sf}'+'.png'
#figs.plot_pca_behavioral_representation(components_list = projection_testing, color = color_testing, title = task + ' Testing',
#                                        path_save = pca_behavioral_path)

#%%
eigenvectors_list = []
eigenvalues_list = []
for i in range(len(neural_activity_training)):
    cov_matrix = stats.cov_matrix(neural_activity=neural_activity_training[i][:,0,:])  ## compute covariance matrix
    eigVal, eigVec = stats.compute_PCA(corr_matrix=cov_matrix)  ## run eigenvalues and eigenvectors analysis
    eigenvectors_list.append(eigVec)
    eigenvalues_list.append(eigVal)
#%%
pca_fig_path =  figures_path + 'pca_spectrum_behaviour_mouse_'+f'{mouse}' +'_session_'+\
                f'{session}_binsize_'+f'{re_sf}'+'.png'
figs.plot_pca_spectrum_behaviour(eigenvalues = eigenvalues_list, n_components = n_components ,
                                 title = task, path_save = pca_fig_path)

#%%
pca_fig_path =  figures_path + 'pca_eigenvector_distance_behaviour_mouse_'+f'{mouse}' +'_session_'\
                +f'{session}_binsize_'+f'{re_sf}'+'.png'
figs.plot_pca_eigenvector_distance_behaviour(eigenvectors=eigenvectors_list, n_components=n_components,
                                             title=task, path_save=pca_fig_path)

#%%
pca_fig_path =  figures_path + 'pca_eigenvector_distance_dist_behaviour_mouse_'+f'{mouse}'\
                +'_session_'+f'{session}_binsize_'+f'{re_sf}'+'.png'
figs.plot_pca_eigenvector_distance_distribution_behaviour(eigenvectors=eigenvectors_list, n_components=n_components,
                                                          title=task, path_save=pca_fig_path)

#%%