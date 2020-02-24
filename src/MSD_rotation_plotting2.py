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


session = input("session:")
if session == 1:
    task = 'OVERLAPPING'
else:
    if session == 2:
        task = 'STABLE'
    else:
        task = 'RANDOM'

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



fig2 = plt.figure()
axes1 = fig2.add_subplot(1, 2, 1, projection='3d')
#axes = fig1.add_subplot(111, projection='3d')
axes1.scatter(elements1_transformed[0][:,0],elements1_transformed[0][:,1],elements1_transformed[0][:,2], c=color1[0], cmap=cmap)
axes1.set_xlabel('MDS1')
axes1.set_ylabel('MDS2')
axes1.set_zlabel('MDS3')
axes1.set_xlim([-1, 1])
axes1.set_ylim([-1, 1])
axes1.set_zlim([-1, 1])
axes1.legend(['Resting'])


axes2 = fig2.add_subplot(1, 2, 2, projection='3d')
#axes = fig.add_subplot(111, projection='3d')
axes2.scatter(elements1_transformed[1][:,0],elements1_transformed[1][:,1],elements1_transformed[1][:,2], c=color1[1], cmap=cmap)
axes2.set_xlabel('MDS1')
axes2.set_ylabel('MDS2')
axes2.set_zlabel('MDS3')
axes2.set_xlim([-1, 1])
axes2.set_ylim([-1, 1])
axes2.set_zlim([-1, 1])
axes2.legend(['Not Exploring'])

for angle in range(0, 360):
    axes1.view_init(30, angle)
    axes2.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
axes1.set_title(task)

plt.show()
