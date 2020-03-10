'''
Created on Thrus 20 Feb 2020
Author: Melisa

Multidimensional scaling in raw data and in pca data
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
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
cmap = cm.jet

mouse = 56165             ### mouse number id
decoding_v = 1            ## decoding version, normaly equal to one
motion_correction_v = 100 ### 100 means everything was aligned, 200 means it was also equalized
alignment_v = 1           ## alignment version
equalization_v = 0        ## equalization version
source_extraction_v = 1   ## source extraction version
component_evaluation_v = 1 ## component evaluation version
registration_v = 1        ## registration version
sf = 10                   ## sampling frequency of the original signal
re_sf= 50                 ## value of resampling

sessions = [1,2,4]       ## sessions for this particular mouse
task = ['OVERLAPPING','STABLE','RANDOM']
session_now = 1          ## session that will run
n_components = 10         ## number of projected components


file_directory = os.environ['PROJECT_DIR'] + 'neural_analysis/data/process/MDS/'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'neural_analysis/data/timeline/'
behaviour_dir = os.environ['PROJECT_DIR'] + 'calcium_imaging_behaviour/data/scoring_time_vector/'
objects_dir = os.environ['PROJECT_DIR'] + 'calcium_imaging_behaviour/data/object_positions/'
figures_path = os.environ['PROJECT_DIR'] + 'neural_analysis/data/process/figures/MDS/'

#%%
##load mds
neural_activity_msd = []
resample_timeline = []
resample_beh = []
condition_vector = []
condition_vector_trials = []
color = []
counter = 0
for session in sessions:
    file_name = file_directory + 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + \
                '_binsize_'+f'{re_sf}'+'.npy'
    time_file_session = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_1_v' +\
                        f'{decoding_v}' + '.1.' + f'{1}' + \
                        '.' + f'{0}' + '.pkl'
    beh_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_event_' + f'{re_sf}' + '.npy'

    condition_objects_file = objects_dir + 'condition_vector_mouse_' + f'{mouse}' + \
                             '_session_' + f'{session}' + '.npy'
    condition_objects_file_matrix = objects_dir + 'condition_matrix_mouse_' + f'{mouse}' +\
                                    '_session_' + f'{session}' + '.npy'
    condition_vector.append(np.load(condition_objects_file))
    #condition_matrix = np.load(condition_objects_file_matrix)
    neural_activity_msd.append(np.load(file_name))
    timeline_file = open(timeline_file_dir + time_file_session, 'rb')
    timeline_info = pickle.load(timeline_file)
    timeline_session = np.zeros(42 + 1)
    for i in range(42):
        timeline_session[i] = timeline_info [i][1]
    timeline_session = timeline_session / re_sf
    timeline_session[42] = neural_activity_msd[counter].shape[0]
    ## load behavioural file, downsample it and separate different parts of the experiment

    behaviour = np.load(behaviour_dir + beh_file_name)
    # resample neural activity and behavioural vector
    reshape_behaviour = np.reshape(behaviour[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
    resample_behaviour = np.reshape(scipy.stats.mode(reshape_behaviour,axis=1)[0],reshape_behaviour.shape[0])
    resample_timeline.append(timeline_session)
    resample_beh.append(resample_behaviour)
    condition = np.zeros_like(resample_beh[counter])
    time_length = np.diff(resample_timeline[counter])
    trial =0
    for i in range(0,42,2):
        condition[int(resample_timeline[counter][i]):int(resample_timeline[counter][i]) + int(time_length[i])] = \
            np.ones_like(condition[int(resample_timeline[counter][i]):int(resample_timeline[counter][i]) + int(time_length[i])]) * condition_vector[counter][trial]

        condition[int(resample_timeline[counter][i+1]):int(resample_timeline[counter][i+1]) + int(time_length[i+1])] = \
            np.ones_like(condition[int(resample_timeline[counter][i+1]):int(resample_timeline[counter][i+1]) + int(time_length[i+1])]) * condition_vector[counter][trial]

        trial = trial+1

    condition_vector_trials.append(condition)
    counter = counter + 1

#%%
##plotting
mds_figure = figures_path + 'msd_mouse_'+f'{mouse}' +'_binsize_'+f'{re_sf}'+'.png'
figs.plot_MDS_multisessions(neural_activity_msd = neural_activity_msd, sessions = sessions, task = task, path_save=mds_figure)


#%% separate conditions with behavioural conditions
figure_mds_behaviour =  figures_path + 'msd_mouse_'+f'{mouse}' +'_binsize_'+f'{re_sf}'+'_'
figs.plot_MDS_multisession_behaviour(neural_activity_msd = neural_activity_msd , resample_timeline= resample_timeline,
                                    resample_beh=resample_beh, task = task,  save_path = figure_mds_behaviour)


#%% separate conditions with objects configurations by separating trials
color = ['k', 'b', 'r', 'g', 'm', 'c']

for session in [0,1,2]:
    neural_activity_days = []
    time_length = np.diff(resample_timeline[session])
    for i in range(0,42,2):
        trial_matrix = neural_activity_msd[session][int(resample_timeline[session][i]):int(resample_timeline[session][i]) + int(time_length[i]),:]
        neural_activity_days.append(trial_matrix)

    neural_activity_resting_days = []
    for i in range(1,43,2):
        trial_matrix = neural_activity_msd[session][int(resample_timeline[session][i]):int(resample_timeline[session][i]) + int(time_length[i]),:]
        neural_activity_resting_days.append(trial_matrix)

    mds_condifguration = []
    mds_condifguration_rest = []
    for i in range(6):
        mds_condifguration.append([])
        mds_condifguration_rest.append([])

    for i in range(1,7):
        trials = np.where(condition_vector[session]==i)[0]
        for trial in trials:
            mds_condifguration[i-1].append(neural_activity_days[trial])
            mds_condifguration_rest[i-1].append(neural_activity_resting_days[trial])

    fig1 = plt.figure()
    axes1 = fig1.add_subplot(1, 2, 1, projection='3d')
    axes2 = fig1.add_subplot(1, 2, 2, projection='3d')
    for i in range(6):
        for j in range(len(mds_condifguration[i])):
            axes1.scatter(mds_condifguration[i][j][:,2], mds_condifguration[i][j][:,0], mds_condifguration[i][j][:, 1], c=color[i])
            axes2.scatter(mds_condifguration_rest[i][j][:,2], mds_condifguration_rest[i][j][:,0], mds_condifguration_rest[i][j][:, 1], c=color[i])
    fig1.show()

    fig1 = plt.figure()
    fig2 = plt.figure()
    for i in range(6):
        for j in range(len(mds_condifguration[i])):
            axes1 = fig1.add_subplot(3, 2, i + 1, projection='3d')
            axes1.scatter(mds_condifguration[i][j][:, 0], mds_condifguration[i][j][:, 1],
                          mds_condifguration[i][j][:, 2], c=color[i])
            axes1.set_xlim([-1, 1])
            axes1.set_ylim([-1, 1])
            axes1.set_zlim([-1, 1])

            axes2 = fig2.add_subplot(3, 2, i + 1, projection='3d')
            axes2.scatter(mds_condifguration_rest[i][j][:, 0], mds_condifguration_rest[i][j][:, 1],
                          mds_condifguration_rest[i][j][:, 2], c=color[i])
            axes2.set_xlim([-1, 1])
            axes2.set_ylim([-1, 1])
            axes2.set_zlim([-1, 1])

    fig1.show()
    fig2.show()

# %% separate conditions with objects configurations by separating trials
color = ['k', 'b', 'r', 'g', 'm', 'c']

for session in [0, 1, 2]:
    neural_activity_days = []
    time_length = np.diff(resample_timeline[session])
    for i in range(0, 42, 2):
        trial_matrix = neural_activity_msd[session][
                       int(resample_timeline[session][i]):int(resample_timeline[session][i]) + int(time_length[i]), :]
        neural_activity_days.append(trial_matrix)

    neural_activity_resting_days = []
    for i in range(1, 43, 2):
        trial_matrix = neural_activity_msd[session][
                       int(resample_timeline[session][i]):int(resample_timeline[session][i]) + int(time_length[i]), :]
        neural_activity_resting_days.append(trial_matrix)

    mds_condifguration = []
    mds_condifguration_rest = []
    for i in range(6):
        mds_condifguration.append([])
        mds_condifguration_rest.append([])

    for i in range(1, 7):
        trials = np.where(condition_vector[session] == i)[0]
        for trial in trials:
            if trial != 20:
                mds_condifguration[i - 1].append(neural_activity_days[trial])
                mds_condifguration_rest[i - 1].append(neural_activity_resting_days[trial])

    fig1 = plt.figure(constrained_layout=True)
    #fig2 = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(3, 3)
    axes1 = fig1.add_subplot(gs[0:3, 0], projection='3d')
    #axes2 = fig2.add_subplot(gs[0:3, 0], projection='3d')
    for i in range(6):
        for j in range(len(mds_condifguration[i])):
            axes1.scatter(mds_condifguration[i][j][:, 0], mds_condifguration[i][j][:, 1],
                          mds_condifguration[i][j][:, 2], c=color[i])
            axes1.set_xlim([-2, 2])
            axes1.set_ylim([-2, 2])
            axes1.set_zlim([-2, 2])
            #axes2.scatter(mds_condifguration_rest[i][j][:, 0], mds_condifguration_rest[i][j][:, 1],
            #              mds_condifguration_rest[i][j][:, 2], c=color[i])

            axes3 = fig1.add_subplot(gs[int(i/2), i%2+1] , projection='3d')
            axes3.scatter(mds_condifguration[i][j][:, 0], mds_condifguration[i][j][:, 1],
                          mds_condifguration[i][j][:, 2], c=color[i])
            axes3.set_xlim([-2, 2])
            axes3.set_ylim([-2, 2])
            axes3.set_zlim([-2, 2])
            axes3.set_title('Configuration:' + f'{i+1}')

            #axes4 = fig2.add_subplot(gs[int(i/2), i%2+1], projection='3d')
            #axes4.scatter(mds_condifguration_rest[i][j][:, 0], mds_condifguration_rest[i][j][:, 1],
            #              mds_condifguration_rest[i][j][:, 2], c=color[i])
            #axes4.set_xlim([-1, 1])
            #axes4.set_ylim([-1, 1])
            #axes4.set_zlim([-1, 1])

    fig1.set_size_inches(10, 10)
    fig1.show()
    #fig2.show()

#%%
conditions_name = [
    'LR, LL',
    'LR, UR',
    'LR, UL',
    'LL, UR',
    'LL, UL',
    'UR, UL'
]
color = ['b', 'r', 'g', 'm', 'c','y']

for session in [0, 1, 2]:

    distance_matrix = scipy.spatial.distance.cdist(neural_activity_msd[session], neural_activity_msd[session], metric='euclidean')

    distance_1 = []
    mean_distance = np.zeros((6,6))
    std_distance = np.zeros((6,6))
    for i in range(1,7):
        distance_2=[]
        for j in range(1,7):
            X = distance_matrix[np.where(condition_vector_trials[session]==i),:]
            Y = X[0,:,np.where(condition_vector_trials[session]==j)]
            distance_2.append(Y[0,:,:])
            mean_distance[i-1,j-1] = np.mean(Y[0,:,:])
            std_distance[i-1,j-1] = np.std(Y[0,:,:])
        distance_1.append(distance_2)

    fig1 = plt.figure(constrained_layout=True)
    #fig2 = plt.figure(constrained_layout=True)
    #gs = plt.GridSpec(3, 2)
    for i in range(6):
        # fig2 = plt.figure(constrained_layout=True)
        gs = plt.GridSpec(3, 3)
        axes1 = fig1.add_subplot(3, 2, i + 1)
        for j in range(6):
            [hist_val, bins] = np.histogram(distance_1[i][j].flatten(), bins = np.arange(0,3,2/50))
            if i == j :
                axes1.scatter(bins[:-1], hist_val / np.sum(hist_val), marker = '*', c= 'k')
            axes1.plot(bins[:-1], hist_val/np.sum(hist_val), c = color[j])
            axes1.set_title('Condition '+ conditions_name[i])
            axes1.set_ylim([0,0.05])
            axes1.set_ylabel('#')
            axes1.set_xlabel('Distance [a.u.]')

    fig1.suptitle('MDS ' + task[session])
    #fig1.set_size_inches(12, 10)
    mds_figure = figures_path + 'mds_distance_configuration_mouse_' + f'{mouse}'+'_session_' +f'{session}'+ '_binsize_' + f'{re_sf}' + '.png'
    fig1.savefig(mds_figure)
    #fig1.show()






#%%
fig1 = plt.figure()
c = np.linspace(0, 20, neural_activity_transformed.shape[0])
axes = fig1.add_subplot(1, 1, 1, projection='3d')
#axes = fig.add_subplot(111, projection='3d')
#axes.plot3D(proj1_transformed[:,0],proj1_transformed[:,1],proj1_transformed[:,2], color = 'k')
axes.scatter(neural_activity_transformed[:,0],neural_activity_transformed[:,1],neural_activity_transformed[:,2], c=c, cmap=cmap)
axes.set_xlabel('MDS1')
axes.set_ylabel('MDS2')
axes.set_zlabel('MDS3')
fig1.suptitle(task,fontsize = 15)
fig1.show()

fig1.savefig('/home/sebastian/Documents/Melisa/neural_analysis/data/process/figures/'
             'MDS_mouse_'+f'{mouse}'+'_session_'+f'{session}'+'.png')


c = np.linspace(0, 20, len(resample_beh))
elements1 = []
elements1_transformed = []
testing_1 = []
color1=[]
vector = neural_activity_transformed[:,:int(resample_timeline[40])]
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



