'''
Created on Mon 10 Feb 2020
Author: Melisa
Stating analysis with neural traces and behaviour
Using PCA to see the activity related to different objects and
exploring vs resting periods
For now this program is specific for mouse 56165
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pickle
import configuration
import general_statistics as stats
import matplotlib.cm as cm
cmap = cm.jet

mouse = 56166

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
delete_list = []
sum_activity = np.sum(neural_activity1,axis = 1)
for i in range(neural_activity1.shape[0]):
    if sum_activity[i] == 0:
        delete_list.append(i)
for i in delete_list:
    neural_activity1 = np.delete(neural_activity1,i,0)
corr_matrix1 = stats.corr_matrix(neural_activity = neural_activity1)
eigenvalues1, eigenvectors1 = stats.compute_PCA(corr_matrix = corr_matrix1)
proj1 = stats.PCA_projection(neural_activity=neural_activity1, eigenvalues=eigenvalues1,
                            eigenvectors=eigenvectors1, n_components=6)



## LOAD BEHAVIOUR
behaviour = np.load(behaviour_dir + beh_file_name_1)
c = np.linspace(0, 20, len(behaviour))
elements1 = []
testing_1 = []
color1=[]
for i in range(6):
    vector = proj1[:,:int(timeline_1[40])]
    vector_beh = behaviour[:int(timeline_1[40])]
    elements1.append(vector[:,np.where(vector_beh== i)])
    color1.append(c[np.where(vector_beh==i)])
    vector = proj1[:,int(timeline_1[40]):]
    vector_beh = behaviour[int(timeline_1[40]):]
    testing_1.append(vector[:,np.where(vector_beh== i)])


## session 2 files
session = 3
file_name_session_2 ='mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_1_v'+ f'{decoding_v}'+'.4.'+f'{100}'+\
                      '.'+f'{alignment_v}'+'.'+ f'{equalization_v}' +'.' + f'{source_extraction_v}'+'.' + \
                      f'{component_evaluation_v}' +'.'+ f'{registration_v}' + '.npy'
time_file_session_2 =  'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_1_v'+ f'{decoding_v}'+'.1.'+f'{1}'+\
                      '.'+f'{0}'+ '.pkl'
beh_file_name_2 = 'mouse_'+f'{mouse}'+'_session_'+f'{2}'+'.npy'


## load source extracted calcium traces condition SESSION 3 (REMEMBER TO CORRECT MISTAKE IN EXCEL SHEETS)
activity = np.load(file_directory + file_name_session_2)
timeline_file= open(timeline_file_dir + time_file_session_2,'rb')
timeline_info = pickle.load(timeline_file)
timeline_3 = np.zeros(42+1)
for i in range(42):
    timeline_3[i] = timeline_info[i][1]
timeline_3[42] = activity.shape[1]
### do analysis corr, PCA
activity_normalized = np.zeros((activity.shape))
for j in range(activity.shape[0]):
    for i in range(0,len(timeline_1)-1):
        activity_segment = activity[j,int(timeline_3[i]):int(timeline_3[i+1])]
        activity_segment = activity_segment - min(activity_segment)
        if max(activity_segment):
            activity_segment_normalized = activity_segment/max(activity_segment)
            activity_normalized[j,int(timeline_3[i]):int(timeline_3[i+1])] =activity_segment_normalized
neural_activity3 = activity_normalized[1:,:]
#neural_activity2 = activity[1:,:]
delete_list = []
sum_activity = np.sum(neural_activity3,axis = 1)
for i in range(neural_activity3.shape[0]):
    if sum_activity[i] == 0:
        delete_list.append(i)
for i in delete_list:
    neural_activity3 = np.delete(neural_activity3,i,0)
corr_matrix3 = stats.corr_matrix(neural_activity = neural_activity3)
eigenvalues3, eigenvectors3 = stats.compute_PCA(corr_matrix = corr_matrix3)
proj3 = stats.PCA_projection(neural_activity=neural_activity3, eigenvalues=eigenvalues3,
                            eigenvectors=eigenvectors3, n_components=6)


behaviour = np.load(behaviour_dir + beh_file_name_2)
c = np.linspace(0, 20, len(behaviour))
elements3 = []
testing_3 = []
color3 = []
for i in range(6):
    vector = proj3[:,:int(timeline_3[40])]
    vector_beh = behaviour[:int(timeline_3[40])]
    elements3.append(vector[:,np.where(vector_beh == i)])
    color3.append(c[np.where(vector_beh == i)])
    vector = proj3[:,int(timeline_3[40]):]
    vector_beh = behaviour[int(timeline_3[40]):]
    testing_3.append(vector[:,np.where(vector_beh== i)])


figure, axes = plt.subplots(2,1)
axes[0].scatter(np.arange(0,len(eigenvalues1)),eigenvalues1, color = 'r')
axes[0].set_title('RANDOM')
axes[0].set_xlabel('Order')
axes[0].set_ylabel('Eigenvalue')
axes[1].scatter(np.arange(0,len(eigenvalues3)),eigenvalues3, color = 'b')
axes[1].set_title('OVERLAPING')
axes[1].set_xlabel('Order')
axes[1].set_ylabel('Eigenvalue')
figure.show()


#%%
### do plotting for analysis corr, PCA and correlate with behaviour

fig1 = plt.figure()
axes = fig1.add_subplot(2, 2, 1, projection='3d')
#axes = fig.add_subplot(111, projection='3d')
axes.scatter(elements1[0][0,:,:],elements1[0][1,:,:],elements1[0][2,:,:], c=color1[0], cmap=cmap)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['Resting'])
axes.set_title('RANDOM')

axes = fig1.add_subplot(2, 2, 2, projection='3d')
#axes = fig.add_subplot(111, projection='3d')
axes.scatter(elements1[1][0,:,:],elements1[1][1,:,:],elements1[1][2,:,:], c=color1[1], cmap=cmap)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['Not Exploring'])
axes.set_title('RANDOM')

axes = fig1.add_subplot(2, 2, 3, projection='3d')
axes.scatter(elements3[0][0,:,:],elements3[0][1,:,:],elements3[0][2,:,:],c=color3[0], cmap=cmap)
axes.set_title('OVERLAPPING')
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['Resting'])

axes = fig1.add_subplot(2, 2, 4, projection='3d')
axes.scatter(elements3[1][0,:,:],elements3[1][1,:,:],elements3[1][2,:,:],c=color3[1], cmap=cmap)
axes.set_title('OVERLAPPING')
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['Not Exploring'])

plt.show()

figure_dir = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/resting_vs_exploring_all_conditions_56166.png'
fig1.savefig(figure_dir)


fig2 = plt.figure()
axes = fig2.add_subplot(2, 2, 1, projection='3d')
axes.scatter(elements1[2][0,:,:],elements1[2][1,:,:],elements1[2][2,:,:],c=color1[2], cmap=cmap)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('LR')

axes = fig2.add_subplot(2, 2, 2, projection='3d')
axes.scatter(elements1[3][0,:,:],elements1[3][1,:,:],elements1[3][2,:,:],c=color1[3], cmap=cmap)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('LL')

axes = fig2.add_subplot(2, 2, 3, projection='3d')
axes.scatter(elements1[4][0,:,:],elements1[4][1,:,:],elements1[4][2,:,:],c=color1[4], cmap=cmap)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('UR')

axes = fig2.add_subplot(2, 2, 4, projection='3d')
axes.scatter(elements1[5][0,:,:],elements1[5][1,:,:],elements1[5][2,:,:],c=color1[5], cmap=cmap)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('UL')

fig2.suptitle('RANDOM')
fig2.show()
figure_dir = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/objects_random_32363.png'
fig2.savefig(figure_dir)



fig2 = plt.figure()
axes = fig2.add_subplot(2, 2, 1, projection='3d')
axes.scatter(elements3[2][0,:,:],elements3[2][1,:,:],elements3[2][2,:,:],c=color3[2], cmap=cmap)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('LR')

axes = fig2.add_subplot(2, 2, 2, projection='3d')
axes.scatter(elements3[3][0,:,:],elements3[3][1,:,:],elements3[3][2,:,:],c=color3[3], cmap=cmap)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('LL')

axes = fig2.add_subplot(2, 2, 3, projection='3d')
axes.scatter(elements3[4][0,:,:],elements3[4][1,:,:],elements3[4][2,:,:],c=color3[4], cmap=cmap)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('UR')

axes = fig2.add_subplot(2, 2, 4, projection='3d')
axes.scatter(elements3[5][0,:,:],elements3[5][1,:,:],elements3[5][2,:,:],c=color3[5], cmap=cmap)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('UL')

fig2.suptitle('OVERLAPING')
fig2.show()
figure_dir = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/objects_overlaping_56166.png'
fig2.savefig(figure_dir)

