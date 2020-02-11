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

## load source extracted calcium traces condition SESSION 1
file_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity/'
file_name = 'mouse_56165_session_1_trial_1_v1.4.100.1.0.1.1.1.npy'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'
timeline_file_path = timeline_file_dir +  'mouse_56165_session_1_trial_1_v1.1.1.0.pkl'
activity = np.load(file_directory + file_name)
timeline_file= open(timeline_file_path,'rb')
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
#neural_activity1 = activity_normalized[1:,:]
neural_activity1 = activity[1:,:]
corr_matrix1 = stats.corr_matrix(neural_activity = neural_activity1)
eigenvalues1, eigenvectors1 = stats.compute_PCA(corr_matrix = corr_matrix1)
proj1 = stats.PCA_projection(neural_activity=neural_activity1, eigenvalues=eigenvalues1,
                            eigenvectors=eigenvectors1, n_components=6)



## load source extracted calcium traces condition SESSION 3 (REMEMBER TO CORRECT MISTAKE IN EXCEL SHEETS)
file_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity/'
file_name = 'mouse_56165_session_2_trial_1_v1.4.100.1.0.1.1.1.npy'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'
timeline_file_path = timeline_file_dir +  'mouse_56165_session_2_trial_1_v1.1.1.0.pkl'
activity = np.load(file_directory + file_name)
timeline_file= open(timeline_file_path,'rb')
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
#neural_activity3 = activity_normalized[1:,:]
neural_activity3 = activity[1:,:]
corr_matrix3 = stats.corr_matrix(neural_activity = neural_activity3)
eigenvalues3, eigenvectors3 = stats.compute_PCA(corr_matrix = corr_matrix3)
proj3 = stats.PCA_projection(neural_activity=neural_activity3, eigenvalues=eigenvalues3,
                            eigenvectors=eigenvectors3, n_components=6)


## load source extracted calcium traces condition SESSION 4
file_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity/'
file_name = 'mouse_56165_session_4_trial_1_v1.4.100.1.0.1.1.1.npy'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'
timeline_file_path = timeline_file_dir +  'mouse_56165_session_4_trial_1_v1.1.1.0.pkl'
activity = np.load(file_directory + file_name)
timeline_file= open(timeline_file_path,'rb')
timeline_info = pickle.load(timeline_file)
timeline_4 = np.zeros(42+1)
for i in range(42):
    timeline_4[i] = timeline_info[i][1]
timeline_4[42] = activity.shape[1]
### do analysis corr, PCA
activity_normalized = np.zeros((activity.shape))
for j in range(activity.shape[0]):
    for i in range(0,len(timeline_1)-1):
        activity_segment = activity[j,int(timeline_4[i]):int(timeline_4[i+1])]
        activity_segment = activity_segment - min(activity_segment)
        if max(activity_segment):
            activity_segment_normalized = activity_segment/max(activity_segment)
            activity_normalized[j,int(timeline_4[i]):int(timeline_4[i+1])] =activity_segment_normalized
#neural_activity4 = activity_normalized[1:,:]
neural_activity4 = activity[1:,:]
corr_matrix4 = stats.corr_matrix(neural_activity = neural_activity4)
eigenvalues4, eigenvectors4 = stats.compute_PCA(corr_matrix = corr_matrix4)
proj4 = stats.PCA_projection(neural_activity=neural_activity4, eigenvalues=eigenvalues4,
                            eigenvectors=eigenvectors4, n_components=6)

## LOAD BEHAVIOUR
PROJECT_DIR = '/home/sebastian/Documents/Melisa/calcium_imaging_behaviour/'
directory = PROJECT_DIR + '/data/scoring_time_vector/'
file_name = 'mouse_56165_session_1.npy'
behaviour = np.load(directory + file_name)
elements1 = []
for i in range(6):
    elements1.append(proj1[:,np.where(behaviour == i)])

file_name = 'mouse_56165_session_3.npy'
behaviour = np.load(directory + file_name)
elements3 = []
for i in range(6):
    elements3.append(proj3[:,np.where(behaviour == i)])

file_name = 'mouse_56165_session_4.npy'
behaviour = np.load(directory + file_name)
elements4 = []
for i in range(6):
    elements4.append(proj4[:,np.where(behaviour == i)])

### do analysis corr, PCA and correlate with behaviour


fig1 = plt.figure()
axes = fig1.add_subplot(1, 3, 1, projection='3d')
#axes = fig.add_subplot(111, projection='3d')
axes.scatter(elements1[0][0,:,:],elements1[0][1,:,:],elements1[0][2,:,:])
axes.scatter(elements1[1][0,:,:],elements1[1][1,:,:],elements1[1][2,:,:])
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['Resting', 'Exploration'])
axes.set_title('OVERLAPPING')

axes = fig1.add_subplot(1, 3, 2, projection='3d')
axes.scatter(elements3[0][0,:,:],elements3[0][1,:,:],elements3[0][2,:,:])
axes.scatter(elements3[1][0,:,:],elements3[1][1,:,:],elements3[1][2,:,:])
axes.set_title('STABLE')
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['Resting', 'Exploration'])

axes = fig1.add_subplot(1, 3, 3, projection='3d')
axes.scatter(elements4[0][0,:,:],elements4[0][1,:,:],elements4[0][2,:,:])
axes.scatter(elements4[1][0,:,:],elements4[1][1,:,:],elements4[1][2,:,:])
axes.set_title('RANDOM')
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['Resting', 'Exploration'])
plt.show()

figure_dir = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/resting_vs_exploring_all_conditions.png'
fig1.savefig(figure_dir)


fig2 = plt.figure()
axes = fig2.add_subplot(1, 3, 1, projection='3d')
#axes = fig.add_subplot(111, projection='3d')
#axes.scatter(elements1[0][0,:,:],elements1[0][1,:,:],elements1[0][2,:,:])
#axes.scatter(elements1[1][0,:,:],elements1[1][1,:,:],elements1[1][2,:,:])
axes.scatter(elements1[2][0,:,:],elements1[2][1,:,:],elements1[2][2,:,:])
axes.scatter(elements1[3][0,:,:],elements1[3][1,:,:],elements1[3][2,:,:])
axes.scatter(elements1[4][0,:,:],elements1[4][1,:,:],elements1[4][2,:,:])
axes.scatter(elements1[5][0,:,:],elements1[5][1,:,:],elements1[5][2,:,:])
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('OVERLAPPING')

axes = fig2.add_subplot(1, 3, 2, projection='3d')
#axes.scatter(elements3[0][0,:,:],elements3[0][1,:,:],elements3[0][2,:,:])
#axes.scatter(elements3[1][0,:,:],elements3[1][1,:,:],elements3[1][2,:,:])
axes.scatter(elements3[2][0,:,:],elements3[2][1,:,:],elements3[2][2,:,:])
axes.scatter(elements3[3][0,:,:],elements3[3][1,:,:],elements3[3][2,:,:])
axes.scatter(elements3[4][0,:,:],elements3[4][1,:,:],elements3[4][2,:,:])
axes.scatter(elements3[5][0,:,:],elements3[5][1,:,:],elements3[5][2,:,:])
axes.set_title('STABLE')
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['LR', 'LL', 'UR', 'UL'])

axes = fig2.add_subplot(1, 3, 3, projection='3d')
#axes.scatter(elements4[0][0,:,:],elements4[0][1,:,:],elements4[0][2,:,:])
#axes.scatter(elements4[1][0,:,:],elements4[1][1,:,:],elements4[1][2,:,:])
axes.scatter(elements4[2][0,:,:],elements4[2][1,:,:],elements4[2][2,:,:])
axes.scatter(elements4[3][0,:,:],elements4[3][1,:,:],elements4[3][2,:,:])
axes.scatter(elements4[4][0,:,:],elements4[4][1,:,:],elements4[4][2,:,:])
axes.scatter(elements4[5][0,:,:],elements4[5][1,:,:],elements4[5][2,:,:])
axes.set_title('RANDOM')
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['LR', 'LL', 'UR', 'UL'])

plt.show()
figure_dir = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/objects_all_conditons.png'
fig2.savefig(figure_dir)



figure1 , axes = plt.subplots(1,3)
#axes = fig.add_subplot(111, projection='3d')
axes[0].scatter(elements1[0][0,:,:],elements1[0][1,:,:])
axes[0].scatter(elements1[1][0,:,:],elements1[1][1,:,:])
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend(['Resting', 'Exploration'])
axes[0].set_title('OVERLAPPING')

axes[1].scatter(elements3[0][0,:,:],elements3[0][1,:,:])
axes[1].scatter(elements3[1][0,:,:],elements3[1][1,:,:])
axes[1].set_title('STABLE')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend(['Resting', 'Exploration'])
axes[1].legend(['Resting', 'Exploration'])

axes[2].scatter(elements4[0][0,:,:],elements4[0][1,:,:])
axes[2].scatter(elements4[1][0,:,:],elements4[1][1,:,:])
axes[2].set_title('RANDOM')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
axes[2].legend(['Resting', 'Exploration'])
axes[2].legend(['Resting', 'Exploration'])
figure1.show()



figure2 , axes = plt.subplots(1,3)
axes[0].scatter(elements1[2][0,:,:],elements1[2][1,:,:])
axes[0].scatter(elements1[3][0,:,:],elements1[3][1,:,:])
axes[0].scatter(elements1[4][0,:,:],elements1[4][1,:,:])
axes[0].scatter(elements1[5][0,:,:],elements1[5][1,:,:])
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend(['LR', 'LL', 'UR', 'UL'])
axes[0].set_title('OVERLAPPING')

axes[1].scatter(elements3[2][0,:,:],elements3[2][1,:,:])
axes[1].scatter(elements3[3][0,:,:],elements3[3][1,:,:])
axes[1].scatter(elements3[4][0,:,:],elements3[4][1,:,:])
axes[1].scatter(elements3[5][0,:,:],elements3[5][1,:,:])
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend(['LR', 'LL', 'UR', 'UL'])
axes[1].set_title('STABLE')

axes[2].scatter(elements4[2][0,:,:],elements4[2][1,:,:])
axes[2].scatter(elements4[3][0,:,:],elements4[3][1,:,:])
axes[2].scatter(elements4[4][0,:,:],elements4[4][1,:,:])
axes[2].scatter(elements4[5][0,:,:],elements4[5][1,:,:])
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
axes[2].legend(['LR', 'LL', 'UR', 'UL'])
axes[2].set_title('RANDOM')

figure2.show()


