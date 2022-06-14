

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pickle
import configuration
import general_statistics as stats

## load source extracted calcium traces
file_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity/'
file_name = 'mouse_56166_session_3_trial_1_v1.4.100.1.0.1.1.1.npy'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'
timeline_file_path = timeline_file_dir +  'mouse_56166_session_3_trial_1_v1.1.1.0.pkl'

activity = np.load(file_directory + file_name)
timeline_file= open(timeline_file_path,'rb')
timeline_info = pickle.load(timeline_file)

timeline = np.zeros(42+1)
for i in range(42):
    timeline[i] = timeline_info[i][1]
timeline[42] = activity.shape[1]

##this is just an example
#figure, axes = plt.subplots(1)
#axes.plot(np.arange(0,activity.shape[1])/10,activity[1,:])
#axes.vlines(timeline/10, ymin = 0, ymax = 450, color = 'k', linestyle = '--')
#figure.show()


## normalize activity within trial and for each neuron
activity_normalized = np.zeros((activity.shape))
for j in range(activity.shape[0]):
    for i in range(0,len(timeline)-1):
        activity_segment = activity[j,int(timeline[i]):int(timeline[i+1])]
        activity_segment = activity_segment - min(activity_segment)
        if max(activity_segment):
            activity_segment_normalized = activity_segment/max(activity_segment)
            activity_normalized[j,int(timeline[i]):int(timeline[i+1])] =activity_segment_normalized
neural_activity = activity_normalized[1:,:]

## do some plotting

#figure_directory = os.environ['PROJECT_DIR'] + 'data/process/figures/'
#plot_name = 'mouse_56165_session_1.png'

figure, axes = plt.subplots(1)
C = activity_normalized.copy()
C[1,:] += C[1,:].min()
for j in range(2, len(C),5):
    axes.plot(np.arange(0,activity.shape[1])/10, C[j,:]+j/5-2)
axes.set_xlabel('t [s]')
axes.set_yticks([])
axes.set_ylabel('activity')
figure.set_size_inches([50., .5 * len(C)/5])
figure.show()
figure.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/mouse_56166_session_3.png')


min_time_step = int(min(np.diff(timeline)))
for j in range(0,activity.shape[0]):
    c_activity_trial = np.zeros((42,min_time_step))
    for i in range(1,len(timeline)-1):
        c_activity_trial[i,:] = activity_normalized[j,int(timeline[i]):int(timeline[i])+min_time_step]

    figure, axes = plt.subplots(1)
    for i in range(1, len(c_activity_trial)):
        c_activity_trial[i] += i
        color = 'r'
        if i%2 == 0:
            color = 'b'
        axes.plot(np.arange(0,min_time_step)/10,c_activity_trial[i], color = color)
    axes.set_xlabel('t [s]')
    axes.set_yticks([])
    axes.set_ylabel('trial')
    figure_name = figure_directory + 'trials/mouse_56165_session_1_cell' + f'{j}'+ '.png'
    figure.savefig(figure_name)


### do analysis corr, PCA and correlate with behaviour

neural_activity = activity[1:,:]
corr_matrix = stats.corr_matrix(neural_activity = neural_activity)
eigenvalues, eigenvectors = stats.compute_PCA(corr_matrix = corr_matrix)
proj2 = stats.PCA_projection(neural_activity=neural_activity, eigenvalues=eigenvalues,
                            eigenvectors=eigenvectors, n_components=6)


PROJECT_DIR = '/home/sebastian/Documents/Melisa/calcium_imaging_behaviour/'
directory = PROJECT_DIR + '/data/scoring_time_vector/'
file_name = 'mouse_56165_events_2.npy'
behaviour = np.load(directory + file_name)

elements2 = []
for i in range(6):
    elements2.append(proj2[:,np.where(behaviour == i)])


elements = elements1
fig1 = plt.figure()
axes = fig1.add_subplot(1, 2, 1, projection='3d')
#axes = fig.add_subplot(111, projection='3d')
axes.scatter(elements[0][0,:,:],elements[0][1,:,:],elements[0][2,:,:])
axes.scatter(elements[1][0,:,:],elements[1][1,:,:],elements[1][2,:,:])
#axes.scatter(elements[2][0,:,:],elements[2][1,:,:],elements[2][2,:,:])
#axes.scatter(elements[3][3,:,:],elements[3][1,:,:],elements[3][2,:,:])
#axes.scatter(elements[4][3,:,:],elements[4][1,:,:],elements[4][2,:,:])
#axes.scatter(elements[5][3,:,:],elements[5][1,:,:],elements[5][2,:,:])
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['Resting', 'Exploration'])
axes.set_title('OVERLAPPING')

axes = fig1.add_subplot(1, 2, 2, projection='3d')
axes.scatter(elements2[0][0,:,:],elements2[0][1,:,:],elements2[0][2,:,:])
axes.scatter(elements2[1][0,:,:],elements2[1][1,:,:],elements2[1][2,:,:])
#axes.scatter(elements2[2][0,:,:],elements2[2][1,:,:],elements2[2][2,:,:])
#axes.scatter(elements2[3][3,:,:],elements2[3][1,:,:],elements2[3][2,:,:])
#axes.scatter(elements2[4][3,:,:],elements2[4][1,:,:],elements2[4][2,:,:])
#axes.scatter(elements2[5][3,:,:],elements2[5][1,:,:],elements2[5][2,:,:])
axes.set_title('RANDOM')
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['Resting', 'Exploration'])
plt.show()
figure_dir = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/resting_vs_exploring.png'
fig1.savefig(figure_dir)


fig2 = plt.figure()
axes = fig2.add_subplot(1, 2, 1, projection='3d')
#axes = fig.add_subplot(111, projection='3d')
#axes.scatter(elements[0][0,:,:],elements[0][1,:,:],elements[0][2,:,:])
#axes.scatter(elements[1][0,:,:],elements[1][1,:,:],elements[1][2,:,:])
axes.scatter(elements[2][0,:,:],elements[2][1,:,:],elements[2][2,:,:])
axes.scatter(elements[3][3,:,:],elements[3][1,:,:],elements[3][2,:,:])
axes.scatter(elements[4][3,:,:],elements[4][1,:,:],elements[4][2,:,:])
axes.scatter(elements[5][3,:,:],elements[5][1,:,:],elements[5][2,:,:])
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['LR', 'LL', 'UR', 'UL'])
axes.set_title('OVERLAPPING')

axes = fig2.add_subplot(1, 2, 2, projection='3d')
#axes.scatter(elements2[0][0,:,:],elements2[0][1,:,:],elements2[0][2,:,:])
#axes.scatter(elements2[1][0,:,:],elements2[1][1,:,:],elements2[1][2,:,:])
axes.scatter(elements2[2][0,:,:],elements2[2][1,:,:],elements2[2][2,:,:])
axes.scatter(elements2[3][3,:,:],elements2[3][1,:,:],elements2[3][2,:,:])
axes.scatter(elements2[4][3,:,:],elements2[4][1,:,:],elements2[4][2,:,:])
axes.scatter(elements2[5][3,:,:],elements2[5][1,:,:],elements2[5][2,:,:])
axes.set_title('RANDOM')
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')
axes.legend(['LR', 'LL', 'UR', 'UL'])
plt.show()
figure_dir = '/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/objects.png'
fig2.savefig(figure_dir)





figure1 , axes = plt.subplots(1,2)
#axes = fig.add_subplot(111, projection='3d')
axes[0].scatter(elements[0][0,:,:],elements[0][1,:,:])
axes[0].scatter(elements[1][0,:,:],elements[1][1,:,:])
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend(['Resting', 'Exploration'])
axes[0].set_title('OVERLAPPING')

axes[1].scatter(elements2[0][0,:,:],elements2[0][1,:,:])
axes[1].scatter(elements2[1][0,:,:],elements2[1][1,:,:])
axes[1].set_title('RANDOM')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend(['Resting', 'Exploration'])
axes[1].legend(['Resting', 'Exploration'])
figure1.show()

figure2 , axes = plt.subplots(1,2)
axes[0].scatter(elements[2][0,:,:],elements[2][1,:,:])
axes[0].scatter(elements[3][3,:,:],elements[3][1,:,:])
axes[0].scatter(elements[4][3,:,:],elements[4][1,:,:])
axes[0].scatter(elements[5][3,:,:],elements[5][1,:,:])
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].legend(['LR', 'LL', 'UR', 'UL'])
axes[0].set_title('OVERLAPPING')

axes[1].scatter(elements2[2][0,:,:],elements2[2][1,:,:])
axes[1].scatter(elements2[3][3,:,:],elements2[3][1,:,:])
axes[1].scatter(elements2[4][3,:,:],elements2[4][1,:,:])
axes[1].scatter(elements2[5][3,:,:],elements2[5][1,:,:])
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend(['LR', 'LL', 'UR', 'UL'])
axes[1].set_title('RANDOM')
figure2.show()


