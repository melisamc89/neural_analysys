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
session = 1
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
neural_activity1 = activity_normalized[1:,:]
#neural_activity1 = activity[1:,:]
#corr_matrix1 = stats.corr_matrix(neural_activity = neural_activity1)
#eigenvalues1, eigenvectors1 = stats.compute_PCA(corr_matrix = corr_matrix1)
#proj1 = stats.PCA_projection(neural_activity=neural_activity1, eigenvalues=eigenvalues1,
#                            eigenvectors=eigenvectors1, n_components=6)

#%% In task correlation matrix

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


corr_matrix = []
for i in range(len(neural_activity_new)):
    corr_matrix.append(stats.corr_matrix(neural_activity = neural_activity_new[i][:,0,:]))

figure, axes = plt.subplots(3,2)
for i in range(3):
    for j in range(2):
        x = axes[i,j].imshow(np.log(corr_matrix[i*2+j]/np.max(corr_matrix[i*2+j].flatten())),cmap = 'viridis')

figure.colorbar(x, ax=axes[i,j])
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes[0,0].set_title('Resting', fontsize = 12)
axes[0,1].set_title('Not exploring', fontsize = 12)
axes[1,0].set_title('LR', fontsize = 12)
axes[1,1].set_title('LL', fontsize = 12)
axes[2,0].set_title('UR', fontsize = 12)
axes[2,1].set_title('UL', fontsize = 12)
figure.show()
figure.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/correlation_matrix_task_mouse_56165_session_4.png')


#%% in days correlation matrix
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

corr_matrix_days = []
for i in range(4):
    corr_matrix_days.append(stats.corr_matrix(neural_activity = neural_activity1_days[i]))
corr_matrix_days.append(stats.corr_matrix(neural_activity = neural_activity1_testing))

corr_matrix_resting_days = []
for i in range(4):
    corr_matrix_resting_days.append(stats.corr_matrix(neural_activity = neural_activity1_resting_days[i]))
corr_matrix_resting_days.append(stats.corr_matrix(neural_activity = neural_activity1_resting_testing))


corr_mean = np.zeros(5)
corr_std = np.zeros(5)
corr_mean_resting = np.zeros(5)
corr_std_resting = np.zeros(5)
for i in range(5):
    corr_mean[i] = np.mean(corr_matrix_days[i].flatten())
    corr_std[i] = np.std(corr_matrix_days[i].flatten())
    corr_mean_resting[i]= np.mean(corr_matrix_resting_days[i].flatten())
    corr_std_resting[i]=np.std(corr_matrix_resting_days[i].flatten())


fig, axes = plt.subplots(1)
axes.errorbar(np.arange(1,6),corr_mean,corr_std)
axes.errorbar(np.arange(1,6),corr_mean_resting,corr_std_resting)
axes.set_xlabel('Days',fontsize = 12)
axes.set_ylabel('Correlation')
axes.legend(['Trial','Resting'])
fig.show()
fig.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/correlation_mouse_56165_session_1.png')

figures, axes = plt.subplots(2,5)
for i in range(5):
    x = axes[0,i].imshow(np.log(corr_matrix_days[i]/np.max(corr_matrix_days[i].flatten())),cmap = 'viridis')
    y = axes[1,i].imshow(np.log(corr_matrix_resting_days[i]/np.max(corr_matrix_resting_days[i].flatten())), cmap = 'viridis')
    #figures.colorbar(x, ax=axes[0, i])
    #figures.colorbar(y, ax=axes[1, i])
    axes[0,i].set_title('Trial Day' + f'{i}', fontsize = 12)
    axes[1,i].set_title('Rest Day' + f'{i}', fontsize = 12)
#figures.set_size_inches([75., 25])
figures.colorbar(y, ax=axes[1, i])
figures.suptitle('Activity correlation matrix' , fontsize = 15)
figures.show()

figures.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/correlation_matrix_mouse_56165_session_4.png')