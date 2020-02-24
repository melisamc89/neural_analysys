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
beh_file_name_1 = 'mouse_'+f'{mouse}'+'_session_'+f'{session}'+'.npy'

##session 1
activity = np.load(file_directory + file_name_session_1)
timeline_file= open(timeline_file_dir + time_file_session_1,'rb')
timeline_info = pickle.load(timeline_file)
timeline_1 = np.zeros(42+1)
for i in range(42):
    timeline_1[i] = timeline_info[i][1]
timeline_1[42] = activity.shape[1]

## LOAD BEHAVIOUR
behaviour = np.load(behaviour_dir + beh_file_name_1)

### downsample
sf = 10
new_sf = 0.1
re_sf = int(sf/new_sf)
#mat_proj1 = np.reshape(proj1[:,:int(int(proj1.shape[1]/re_sf )*re_sf) ],(proj1.shape[0],int(proj1.shape[1]/re_sf ),re_sf))
#resample_proj1 = np.mean(mat_proj1,axis = 2)
#std_resample_proj1 = np.std(mat_proj1, axis = 2)
timeline_1 = timeline_1 / re_sf
positions = np.arange(0,behaviour.shape[0],re_sf)
resample_beh1 = behaviour[positions[:-1]]

file_name = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/MDS/mouse_' + f'{mouse}' + '_session_' + f'{session}' + '.npy'
proj1_transformed = np.load(file_name)


fig1 = plt.figure()
c = np.linspace(0, 20, proj1_transformed.shape[0])
axes = fig1.add_subplot(1, 1, 1, projection='3d')
#axes = fig.add_subplot(111, projection='3d')
#axes.plot3D(proj1_transformed[:,0],proj1_transformed[:,1],proj1_transformed[:,2], color = 'k')
axes.scatter(proj1_transformed[:,0],proj1_transformed[:,1],proj1_transformed[:,2], c=c, cmap=cmap)
axes.set_xlabel('MDS1')
axes.set_ylabel('MDS2')
axes.set_zlabel('MDS3')
fig1.suptitle(task,fontsize = 15)
fig1.show()

fig1.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/'
             'MDS_mouse_'+f'{mouse}'+'_session_'+f'{session}'+'.png')


c = np.linspace(0, 20, len(resample_beh1))
elements1 = []
elements1_transformed = []
testing_1 = []
color1=[]
vector = proj1_transformed[:,:int(timeline_1[40])]
vector_beh = np.round(resample_beh1[:int(timeline_1[40])])
for i in range(6):
    auxiliar = vector[np.where(vector_beh== i),:]
    elements1.append(auxiliar)
    elements1_transformed.append(vector[np.where(vector_beh== i),:][0,:,:])
    color1.append(c[np.where(vector_beh==i)])
    #vector = proj1[:,int(timeline_1[40]):]
    #vector_beh = behaviour[int(timeline_1[40]):]
    #testing_1.append(vector[:,np.where(vector_beh== i)])

#elements1_transformed = []
#for i in range(6):
#    embedding = MDS(n_components=3)
#    elements1_transformed.append(embedding.fit_transform(elements1[i].T))



fig1 = plt.figure()
axes = fig1.add_subplot(1, 2, 1, projection='3d')
#axes = fig1.add_subplot(111, projection='3d')
#axes.plot3D(elements1_transformed[0][:,0],elements1_transformed[0][:,1],elements1_transformed[0][:,2], color = 'k')
axes.scatter(elements1_transformed[0][:,0],elements1_transformed[0][:,1],elements1_transformed[0][:,2], c=color1[0], cmap=cmap)
axes.set_xlabel('MDS1')
axes.set_ylabel('MDS2')
axes.set_zlabel('MDS3')
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
axes.set_zlim([-1, 1])
axes.legend(['Resting'])
axes.set_title(task)

axes = fig1.add_subplot(1, 2, 2, projection='3d')
#axes = fig.add_subplot(111, projection='3d')
#axes.plot3D(elements1_transformed[1][:,0],elements1_transformed[1][:,1],elements1_transformed[1][:,2], color = 'k')
axes.scatter(elements1_transformed[1][:,0],elements1_transformed[1][:,1],elements1_transformed[1][:,2], c=color1[1], cmap=cmap)
axes.set_xlabel('MDS1')
axes.set_ylabel('MDS2')
axes.set_zlabel('MDS3')
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
axes.set_zlim([-1, 1])
axes.legend(['Not Exploring'])
axes.set_title(task)

plt.show()

fig1.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/'
             'resting_vs_exploring_MDS_mouse_'+f'{mouse}'+'_session_'+f'{session}'+'.png')



fig2 = plt.figure()
axes = fig2.add_subplot(2, 2, 1, projection='3d')
#axes.plot3D(elements1_transformed[2][:,0],elements1_transformed[2][:,1],elements1_transformed[2][:,2], color = 'k')
axes.scatter(elements1_transformed[2][:,0],elements1_transformed[2][:,1],elements1_transformed[2][:,2], c=color1[2], cmap=cmap)
axes.set_xlabel('MDS1')
axes.set_ylabel('MDS2')
axes.set_zlabel('MDS3')
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
axes.set_zlim([-1, 1])
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('LR')

axes = fig2.add_subplot(2, 2, 2, projection='3d')
#axes.plot3D(elements1_transformed[3][:,0],elements1_transformed[3][:,1],elements1_transformed[3][:,2], color = 'k')
axes.scatter(elements1_transformed[3][:,0],elements1_transformed[3][:,1],elements1_transformed[3][:,2], c=color1[3], cmap=cmap)
axes.set_xlabel('MDS1')
axes.set_ylabel('MDS2')
axes.set_zlabel('MDS3')
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
axes.set_zlim([-1, 1])
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('LL')

axes = fig2.add_subplot(2, 2, 3, projection='3d')
#axes.plot3D(elements1_transformed[4][:,0],elements1_transformed[4][:,1],elements1_transformed[4][:,2], color = 'k')
axes.scatter(elements1_transformed[4][:,0],elements1_transformed[4][:,1],elements1_transformed[4][:,2], c=color1[4], cmap=cmap)
axes.set_xlabel('MDS1')
axes.set_ylabel('MDS2')
axes.set_zlabel('MDS3')
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
axes.set_zlim([-1, 1])
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('UR')

axes = fig2.add_subplot(2, 2, 4, projection='3d')
#axes.plot3D(elements1_transformed[5][:,0],elements1_transformed[5][:,1],elements1_transformed[5][:,2], color = 'k')
axes.scatter(elements1_transformed[5][:,0],elements1_transformed[5][:,1],elements1_transformed[5][:,2], c=color1[5], cmap=cmap)
axes.set_xlabel('MDS1')
axes.set_ylabel('MDS2')
axes.set_zlabel('MDS3')
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
axes.set_zlim([-1, 1])
#axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('UL')

fig2.suptitle(task)
fig2.show()

fig1.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/'
             'object_task_MDS_mouse_'+f'{mouse}'+'_session_'+f'{session}'+'.png')



