

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import configuration

## load source extracted calcium traces
file_directory = os.environ['PROJECT_DIR'] + 'data/calcium_activity/'
file_name = 'mouse_56165_session_1_trial_1_v1.4.100.1.0.1.1.1.npy'
timeline_file_dir = os.environ['PROJECT_DIR'] + 'data/timeline/'
timeline_file_path = timeline_file_dir +  'mouse_56165_session_1_trial_1_v1.1.1.0.pkl'
figure_directory = os.environ['PROJECT_DIR'] + 'data/process/figures/'
plot_name = 'mouse_56165_session_1.png'


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

figure, axes = plt.subplots(1)
C = activity_normalized.copy()
C[1,:] += C[1,:].min()
for j in range(2, len(C)):
    axes.plot(np.arange(0,activity.shape[1])/10, C[j,:]+j-2)
axes.set_xlabel('t [s]')
axes.set_yticks([])
axes.set_ylabel('activity')
figure.set_size_inches([50., .5 * len(C)])
figure.show()
figure.savefig(figure_directory + plot_name)



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
