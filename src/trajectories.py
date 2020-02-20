'''
Created on Mon 10 Feb 2020
Author: Melisa

Plotting and analysis of neural trajectories in the pca domain

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
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
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
corr_matrix1 = stats.corr_matrix(neural_activity = neural_activity1)
eigenvalues1, eigenvectors1 = stats.compute_PCA(corr_matrix = corr_matrix1)
proj1 = stats.PCA_projection(neural_activity=neural_activity1, eigenvalues=eigenvalues1,
                            eigenvectors=eigenvectors1, n_components=6)

## LOAD BEHAVIOUR
behaviour = np.load(behaviour_dir + beh_file_name_1)

resample_proj1= signal.resample(proj1,int(proj1.shape[1]/20),axis = 1)
resample_beh1 = signal.resample(behaviour,int(proj1.shape[1]/20))
timeline_1 = timeline_1 / 20

c = np.linspace(0, 20, len(resample_beh1))
elements1 = []
testing_1 = []
color1=[]
for i in range(6):
    vector = resample_proj1[:,:int(timeline_1[40])]
    vector_beh = resample_beh1[:int(timeline_1[40])]
    elements1.append(vector[:,np.where(vector_beh== i)])
    color1.append(c[np.where(vector_beh==i)])
    vector = resample_proj1[:,int(timeline_1[40]):]
    vector_beh = resample_beh1[int(timeline_1[40]):]
    testing_1.append(vector[:,np.where(vector_beh== i)])


x = elements1[0,:,:]
y = elements1[1,:,:]
z = elements1[2,:,:]

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
# Create a continuous norm to map from data points to colors
norm = plt.Normalize(y.min(), y.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
# Set the values used for colormapping
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs[0].add_collection(lc)
fig.colorbar(line, ax=axs[0])
# Use a boundary norm instead
cmap = ListedColormap(['r', 'g', 'b'])
norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs[1].add_collection(lc)
fig.colorbar(line, ax=axs[1])
axs[0].set_xlim(x.min(), x.max())
axs[0].set_ylim(-1.1, 1.1)
plt.show()
