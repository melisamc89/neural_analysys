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
session = 4
file_name_session_1 = 'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_1_v'+ f'{decoding_v}'+'.4.'+f'{100}'+\
                      '.'+f'{alignment_v}'+'.'+ f'{equalization_v}' +'.' + f'{source_extraction_v}'+'.' + \
                      f'{component_evaluation_v}' +'.'+ f'{registration_v}' + '.npy'
time_file_session_1 =  'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_1_v'+ f'{decoding_v}'+'.1.'+f'{1}'+\
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
corr_matrix1_1 = stats.corr_matrix(neural_activity = neural_activity1)
#eigenvalues1, eigenvectors1 = stats.compute_PCA(corr_matrix = corr_matrix1)
#proj1 = stats.PCA_projection(neural_activity=neural_activity1, eigenvalues=eigenvalues1,
#                            eigenvectors=eigenvectors1, n_components=6)


figure, axes = plt.subplots(1,2)
x = axes[0].imshow(np.log(corr_matrix1/max(corr_matrix1.flatten())))
y = axes[1].imshow(np.log(corr_matrix1_1/max(corr_matrix1_1.flatten())))
axes[0].set_title('Raw Calcium traces')
axes[1].set_title('Normalized Calcium traces')
figure.colorbar(x, ax = axes[0])
figure.colorbar(y, ax = axes[1])
figure.suptitle('                             Correlation matrix',fontsize = 15)
figure.show()
figure.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/correlation_matrix_all_mouse_56165_session_4.png')


#%%

mean_neural_activity = np.mean(neural_activity1,axis = 1)
sorted_mean = np.argsort(mean_neural_activity)
sorted_value =  np.sort(mean_neural_activity)
selection_sorted_mean = sorted_mean[np.where(sorted_value >0.02)]
sorted_neural_activity = neural_activity1[selection_sorted_mean[::-1],:]


figure, axes = plt.subplots(1)
C = sorted_neural_activity.copy()
C[1,:] += C[1,:].min()
for j in range(2, len(C)):
    axes.plot(np.arange(0,activity.shape[1]), C[j,:]+j-2)
axes.set_xlabel('t [s]')
axes.set_yticks([])
axes.set_ylabel('activity')
figure.set_size_inches([50., .5 * len(C)])
figure.show()


#%%


## LOAD BEHAVIOUR
behaviour = np.load(behaviour_dir + beh_file_name_1)
#c = np.linspace(0, 20, len(behaviour))
neural_activity_new= []
testing = []
#color1=[]
for i in range(6):
    vector = neural_activity1[:,:int(timeline_1[40])]
    vector_beh = behaviour[:int(timeline_1[40])]
    neural_activity_new.append(vector[:,np.where(vector_beh== i)])
    #color1.append(c[np.where(vector_beh==i)])
    #vector = proj1[:,int(timeline_1[40]):]
    #vector_beh = behaviour[int(timeline_1[40]):]
    #testing_1.append(vector[:,np.where(vector_beh== i)])

mean_neural_activity = []
for i in range(6):
    mean_neural_activity.append(np.mean(neural_activity_new[i],axis = 2))


figure, axes = plt.subplots(1)
for i in range(3):
    for j in range(2):
        [dist, nbins] = np.histogram(mean_neural_activity[i*2+j], bins=15)
        dist = dist / sum(dist)
        axes.plot(nbins[:-1],dist)
figure.show()
#figure.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/mean_activity_task_mouse_56165_session_4.png')


neural_activity1_days = []
time_length = np.diff(timeline_1)
for i in range(4):
    day_matrix = np.zeros((neural_activity1.shape[0],int(np.sum(time_length[i*10:(i+1)*10:2]))))
    start_time = 0
    for j in range(0,10,2):
        trial = i*10 + j
        day_matrix[:,start_time:start_time+ int(time_length[trial])] = \
            neural_activity1[:,int(timeline_1[trial]):int(timeline_1[trial]) + int(time_length[trial])]
        start_time = start_time + int(time_length[trial])
    neural_activity1_days.append(day_matrix)

neural_activity1_resting_days = []
time_length = np.diff(timeline_1)
for i in range(4):
    day_matrix = np.zeros((neural_activity1.shape[0],int(np.sum(time_length[i*10+1:(i+1)*10+1:2]))))
    start_time = 0
    for j in range(1,10,2):
        trial = i*10 + j
        day_matrix[:,start_time:start_time+ int(time_length[trial])] = \
            neural_activity1[:,int(timeline_1[trial]):int(timeline_1[trial]) + int(time_length[trial])]
        start_time = start_time + int(time_length[trial])
    neural_activity1_resting_days.append(day_matrix)

neural_activity1_resting_testing = neural_activity1[:,int(timeline_1[-2]):int(timeline_1[-1])]
neural_activity1_testing = neural_activity1[:,int(timeline_1[-3]):int(timeline_1[-2])]


mean_activity_days = []
for i in range(4):
    mean_activity_days.append(np.mean(neural_activity1_days[i],axis = 1))
mean_activity_days.append(np.mean(neural_activity1_testing, axis = 1))

mean_activity_resting_days = []
for i in range(4):
    mean_activity_resting_days.append(np.mean(neural_activity1_resting_days[i],axis = 1))
mean_activity_resting_days.append(np.mean(neural_activity1_resting_testing, axis= 1))

figure, axes = plt.subplots(2,5)
for i in range(5):
    [dist, nbins] = np.histogram(mean_activity_days[i], bins=15)
    dist = dist / sum(dist)
    axes[0,i].plot(nbins[:-1],dist)
    #axes[0,i].set_xlim([0.001,0.25])
    #axes[0,i].set_ylim([0.00,0.1])
    axes[0,i].set_xlabel('Mean activation')

    [dist, nbins] = np.histogram(mean_activity_resting_days[i], bins=15)
    dist = dist / sum(dist)
    axes[1,i].plot(nbins[:-1],dist)
    #axes[1,i].set_xlim([0.001,0.25])
    #axes[1,i].set_ylim([0.00,0.1])
    axes[0,i].set_title('Trial Day' + f'{i}', fontsize = 12)
    axes[1,i].set_title('Rest Day' + f'{i}', fontsize = 12)
    axes[1,i].set_xlabel('Mean activation')

figure.show()
figure.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/mean_activation_mouse_56165_session_1.png')
