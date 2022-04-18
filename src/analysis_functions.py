#import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import configuration
import general_statistics as stats
import figures as figs
import matplotlib.cm as cm
from matplotlib import colors
from scipy import signal
from scipy import stats as sstats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy
cmap = cm.jet
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import cross_val_score
import random
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from collections import namedtuple

def speed_selection(tracking = None, speed_limit = 3):
    
    x = tracking[0,:]
    y = tracking[1,:]
    vx = np.diff(x)
    vy = np.diff(y)
    speed = np.sqrt(vx*vx+vy*vy)
    index = np.where(speed > speed_limit)[0]
    return index


def mouse_properties(mouse = None, session_now = None):
    fixed = 'None'
    object_fixed = 'None'
    if mouse == 411857:
        sessions = [1,2,3]
        if session_now == 1:
            task = 'STABLE'
            colapse_behaviour = 2
            labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]
        if session_now == 2:
            task = 'RANDOM'
            colapse_behaviour = 0
            labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]   
        if session_now == 3:
            task = 'OVERLAPPING'
            fixed = 'LL'
            object_fixed = 3
            colapse_behaviour = 1
            labels =['Unlabel','Rest1', 'Navigation', 'Overlap_object' , 'Moving_object','RunOO' , 'RunMO' ]
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]   


    if mouse == 56165 or mouse == 32364:
        if mouse == 56165:
            sessions = [1,2,4]       ## sessions for this particular mouse
        if mouse == 32364:
            sessions = [1,2]
        if session_now == 1:
            task = 'OVERLAPPING'
            colapse_behaviour = 1
            labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]
            if mouse == 32364:
                fixed = 'LR'
                object_fixed = 4
                colapse_behaviour = 1
                labels =['Unlabel','Rest1', 'Navigation', 'Overlap_object' , 'Moving_object','RunOO' , 'RunMO' ]
                colornames=['k',"r","deepskyblue","g","blue","g","blue"]   
            if mouse == 56165:
                fixed = 'UR'
                object_fixed = 5
                colapse_behaviour = 1
                labels =['Unlabel','Rest1', 'Navigation', 'Overlap_object' , 'Moving_object','RunOO' , 'RunMO' ]
                colornames=['k',"r","deepskyblue","g","blue","g","blue"]   

        else:
            if session_now == 2:
                task = 'STABLE'
                colapse_behaviour = 2
                labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
                colornames=['k',"r","deepskyblue","g","blue","g","blue"]
            else:
                task = 'RANDOM'
                colapse_behaviour = 0
                labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
                colornames=['k',"r","deepskyblue","g","blue","g","blue"]

    if mouse == 32365:
        sessions = [2,3] ## sessions for this particular mouse
        if session_now == 2:
            task = 'RANDOM'
            colapse_behaviour = 0
            labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]
        else:
            if session_now == 3:
                task = 'OVERLAPPING'
                fixed = 'LR'
                object_fixed = 4
                colapse_behaviour = 1
                labels =['Unlabel','Rest1', 'Navigation', 'Overlap_object' , 'Moving_object','RunOO' , 'RunMO' ]
                colornames=['k',"r","deepskyblue","g","blue","g","blue"]           

    if mouse == 56166:
        sessions = [1,2] ## sessions for this particular mouse
        if session_now == 1:
            task = 'RANDOM'
            colapse_behaviour = 0
            labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]

        else:
            if session_now == 2:
                task = 'OVERLAPPING'
                fixed = 'UR'
                object_fixed = 5
                colapse_behaviour = 1
                labels =['Unlabel','Rest1', 'Navigation', 'Overlap_object' , 'Moving_object','RunOO' , 'RunMO' ]
                colornames=['k',"r","deepskyblue","g","blue","g","blue"]

    if mouse == 32366:
        sessions = [2,3] ## sessions for this particular mouse
        if session_now == 3:
            task = 'RANDOM'
            colapse_behaviour = 0
            labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]

    if mouse == 32363:
        sessions = [1,2] ## sessions for this particular mouse
        if session_now == 1:
            task = 'RANDOM'
            colapse_behaviour = 0
            labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]
        else:
            if session_now == 2:
                task = 'OVERLAPPING'
                fixed = 'UL'
                object_fixed = 6
                colapse_behaviour = 1
                labels =['Unlabel','Rest1', 'Navigation', 'Overlap_object' , 'Moving_object','RunOO' , 'RunMO' ]
                colornames=['k',"r","deepskyblue","g","blue","g","blue"]


    return task,colapse_behaviour,object_fixed,fixed,labels,colornames


def load_data(mouse = None, session = None, decoding_v = None, motion_correction_v = None, alignment_v = None, equalization_v = None, source_extraction_v = None, component_evaluation_v = None, re_sf = None,file_directory = None, timeline_file_dir = None, behaviour_dir = None, behaviour_dir_parameters=None, tracking_dir = None, objects_dir = None):
    
    activity_list = []
    timeline_list = []
    behaviour_list = []
    corners_list = []
    speed_list = []

    parameters_list = []
    parameters_list2 = []
    parameters_time = []
    tracking_list = []
    total_time = 0
    day = 0

    print('LOADING TRIALS ACTIVITY AND CREATING LIST OF ACTIVITY, TRACKING AND BEHAVIOUR')
    for trial in [1,6,11,16]:

        beh_file_name_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75_ethogram.npy'
        ## LOAD BEHAVIOUR
        behaviour = np.load(behaviour_dir + beh_file_name_1)
        reshape_behaviour = np.reshape(behaviour[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
        resample_beh1 = np.reshape(scipy.stats.mode(reshape_behaviour,axis=1)[0],reshape_behaviour.shape[0])

        beh_file_name_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75_object_corners.npy'
        ## LOAD CORNERS EXPLORATION
        behaviour = np.load(behaviour_dir + beh_file_name_1)
        reshape_behaviour = np.reshape(behaviour[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
        corners = np.reshape(scipy.stats.mode(reshape_behaviour,axis=1)[0],reshape_behaviour.shape[0])

        speed_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75_speed.npy'
        ## LOAD INSTANTANEOUS SPEED
        speed = np.load(behaviour_dir + speed_file_name)
        reshape_speed = np.reshape(speed[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
        resample_speed = np.reshape(scipy.stats.mode(reshape_speed,axis=1)[0],reshape_speed.shape[0])


        beh_file_name_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + f'{day+1}' + '_likelihood_0.75_ethogram_parameters.npy'
        ## LOAD PARAMETRS FOR BEHAVIOUR CLASSIFICATION
        parameters = np.load(behaviour_dir_parameters + beh_file_name_1)

        params0 = []
        params = []
        for param in range(0,2): ## take only ALLOCENTRIC REPRESENTATION
            r1_params = np.reshape(parameters[param,:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
            r2_params = np.reshape(scipy.stats.mode(r1_params,axis=1)[0],reshape_behaviour.shape[0])
            r_params = parameters[param,:resample_speed.shape[0]]
            r_params[:r2_params.shape[0]] = r2_params
            params.append(r_params)
        resample_params0 = np.array(params)

        params = []
        for param in range(2,7): ## take only ALLOCENTRIC REPRESENTATION
            r1_params = np.reshape(parameters[param,:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
            r2_params = np.reshape(scipy.stats.mode(r1_params,axis=1)[0],reshape_behaviour.shape[0])
            r_params = parameters[param,:resample_speed.shape[0]]
            r_params[:r2_params.shape[0]] = r2_params
            params.append(r_params)
        resample_params = np.array(params)

        params2 = []
        for param in range(7,11): ## take only ALLOCENTRIC REPRESENTATION
            r1_params = np.reshape(parameters[param,:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
            r2_params = np.reshape(scipy.stats.mode(r1_params,axis=1)[0],reshape_behaviour.shape[0])
            r_params = parameters[param,:resample_speed.shape[0]]
            r_params[:r2_params.shape[0]] = r2_params
            params.append(r_params)
        resample_params2 = np.array(params)

        ## LOAD TRACKING
        tracking_file_name_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75.npy'
        position = np.load(tracking_dir + tracking_file_name_1)
        resample_position, resample_position_stc = stats.resample_matrix(neural_activity=position.T,
                                                                                                re_sf=re_sf)
        ## LOAD TIMELINE
        time_file_session_1 =  'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_'+ f'{trial}'+'_v1.3.1.0_10.pkl'

        timeline_file= open(timeline_file_dir + time_file_session_1,'rb')
        timeline_info = pickle.load(timeline_file)
        timeline_1 = np.zeros(len(timeline_info) + 1)
        for i in range(len(timeline_info)):
            timeline_1[i] = timeline_info[i][1]
        timeline_1[len(timeline_info)] = behaviour.shape[0]
        timeline = timeline_1/re_sf
        time_lenght = 10
        resample_timeline = timeline_1/re_sf
        timeline_list.append(resample_timeline)

        behaviour_list.append(resample_beh1)
        corners_list.append(corners)
        speed_list.append(resample_speed)

        parameters_list.append(resample_params)
        parameters_list2.append(resample_params2)
        parameters_time.append(resample_params0)
        tracking_list.append(resample_position)
        total_time = total_time + behaviour.shape[0]


        file_name_session_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_'+ f'{trial}'+'_v' + f'{decoding_v}' + '.4.' + f'{motion_correction_v}' + '.' + f'{alignment_v}' + '.' + f'{equalization_v}' + '.' + f'{source_extraction_v}' + '.' + f'{component_evaluation_v}' +  '.0.npy'

         ##load activity and timeline
        activity = np.load(file_directory + file_name_session_1)
        neural_activity1 = activity[1:,:]
        ## z-score neural activity
        neural_activity = neural_activity1[:,:int(int(behaviour.shape[0]/re_sf)*re_sf)]
        ##downsample neural activity
        resample_neural_activity_mean, resample_neural_activity_std = stats.resample_matrix(neural_activity=neural_activity,
                                                                                                re_sf=re_sf)

        activity_list.append(resample_neural_activity_mean)
        print(resample_neural_activity_mean.shape)

        day = day + 1
    
    return activity_list,timeline_list,behaviour_list,corners_list,parameters_time,parameters_list,parameters_list2,speed_list

def transform_neural_data(activity_list, behaviour_list,parameters_time,parameters_list,parameters_list2):
    
    day = 0
    pca = PCA()
    activity_list_pca = []
    variance_list = []
    variance_ratio_list = []

    cca_components = min(5,activity_list[day].shape[0])
    cca = CCA(n_components=cca_components)
    cca_components2 = 4
    cca2 = CCA(n_components=cca_components2)
    cca_components0 = 2
    cca0 = CCA(n_components=cca_components0)

    activity_list_cca = []
    activity_list_cca2 = []
    activity_list_cca0 = []

    clf = LDA()
    activity_list_lda = []

    #embedding = MDS(n_components=3)

    for day in range(len(behaviour_list)):

        ### run pca on the entire dataset
        pca.fit(activity_list[day].T)
        transformed_activity = pca.fit(activity_list[day].T).transform(activity_list[day].T)
        #X_pc_transformed = embedding.fit_transform(transformed_activity.T)
        activity_list_pca.append(transformed_activity.T)
        variance_list.append(pca.explained_variance_/(1+np.sqrt(activity_list[day].shape[0]/activity_list[day].shape[1]))**2)
        normalized_variance = pca.explained_variance_/(1+np.sqrt(activity_list[day].shape[0]/activity_list[day].shape[1]))**2
        variance_ratio_list.append(np.cumsum(normalized_variance/sum(normalized_variance)))

        X = activity_list[day].T
        y = behaviour_list[day]
        X_transformed = clf.fit(X, y).transform(X)
        #X_lda_transformed = embedding.fit_transform(X_transformed.T)
        activity_list_lda.append(X_transformed.T)

        cca_transformed = cca0.fit(activity_list[day].T, parameters_time[day].T).transform(activity_list[day].T)
        #X_cc_transformed = embedding.fit_transform(cca_transformed.T)    
        activity_list_cca0.append(cca_transformed.T)

        cca_transformed = cca.fit(activity_list[day].T, parameters_list[day].T).transform(activity_list[day].T)
        #X_cc_transformed = embedding.fit_transform(cca_transformed.T)    
        activity_list_cca.append(cca_transformed.T)


        cca_transformed = cca2.fit(activity_list[day].T, parameters_list2[day].T).transform(activity_list[day].T)
        #X_cc_transformed = embedding.fit_transform(cca_transformed.T)    
        activity_list_cca2.append(cca_transformed.T)
    
    data_transformation = namedtuple('data_transformation', ['pca','variance_ratio','cca_time','cca_allo','cca_ego','lda'])    
    return data_transformation(activity_list_pca,variance_ratio_list,activity_list_cca0,activity_list_cca,activity_list_cca2,activity_list_lda)

    
def create_task_behaviour(behaviour_list,colapse_behaviour,object_fixed):
    
    # define targets of behaviour according to protocol (random, overlapping and stable)
    id_target = [0,1,2,3,4] # 0: unlabeled, 1:resting, 2:Navigation, 3: Obj1, 4:Obj2, 5:Run1, 6:Run2
    
    print('REDIFINING BEHAVIOUR FOR DIFFERENT SESSIONS')
    if colapse_behaviour == 0 : # RANDOM
        for day in range(len(behaviour_list)):
            for trial in range(5):
                behaviour_trial = behaviour_list[day][int(timeline_list[day][2*trial]):int(timeline_list[day][2*trial+1])]
                objects = np.unique(behaviour_trial)
                selected_object = np.random.randint(len(objects)-4,len(objects)-2,1)
                index0 = np.where(behaviour_trial==objects[selected_object])[0]
                index1 = np.where(np.logical_and(behaviour_trial==objects[len(objects)-4], behaviour_trial!=objects[selected_object]))[0]
                index2 = np.where(np.logical_and(behaviour_trial==objects[len(objects)-3], behaviour_trial!=objects[selected_object]))[0]
                behaviour_trial[index0] = 3
                behaviour_trial[index1] = 4

                behaviour_trial[index2] = 4            

                index0 = np.where(behaviour_trial==objects[selected_object]+4)[0]
                index1 = np.where(np.logical_and(behaviour_trial==objects[len(objects)-2], behaviour_trial!=objects[selected_object]+4))[0]
                index2 = np.where(np.logical_and(behaviour_trial==objects[len(objects)-1], behaviour_trial!=objects[selected_object]+4))[0]
                behaviour_trial[index0] = 0
                behaviour_trial[index1] = 0
                behaviour_trial[index2] = 0 

                behaviour_list[day][int(timeline_list[day][2*trial]):int(timeline_list[day][2*trial+1])] = behaviour_trial


    if colapse_behaviour == 1 : #OVERLAPPING
        for day in range(len(behaviour_list)):
            behaviour_list[day][np.where(behaviour_list[day] == object_fixed)[0]] = 100
            behaviour_list[day][np.where(np.logical_and(behaviour_list[day]>=3, behaviour_list[day]<=6))[0]] = 4
            behaviour_list[day][np.where(behaviour_list[day] == 100)[0]] = 3        
            behaviour_list[day][np.where(behaviour_list[day] == object_fixed +4)[0]] = 0        
            behaviour_list[day][np.where(np.logical_and(behaviour_list[day]>=7, behaviour_list[day]<=10))[0]] = 0
            behaviour_list[day][np.where(behaviour_list[day] == 200)[0]] = 0


    if colapse_behaviour == 2: #STABLE
        for day in range(len(behaviour_list)):
            objects = np.unique(behaviour_list[day])
            selected_object = np.random.randint(len(objects)-4,len(objects)-2,1)
            index0 = np.where(behaviour_list[day]==objects[selected_object])[0]
            index1 = np.where(np.logical_and(behaviour_list[day]==objects[len(objects)-4], behaviour_list[day]!=objects[selected_object]))
            index2 = np.where(np.logical_and(behaviour_list[day]==objects[len(objects)-3], behaviour_list[day]!=objects[selected_object]))
            behaviour_list[day][index0] = 3
            behaviour_list[day][index1] = 4
            behaviour_list[day][index2] = 4      

            index0 = np.where(behaviour_list[day]==objects[selected_object]+4)[0]
            index1 = np.where(np.logical_and(behaviour_list[day]==objects[len(objects)-2], behaviour_list[day]!=objects[selected_object]+4))
            index2 = np.where(np.logical_and(behaviour_list[day]==objects[len(objects)-1], behaviour_list[day]!=objects[selected_object]+4))
            behaviour_list[day][index0] = 0
            behaviour_list[day][index1] = 0
            behaviour_list[day][index2] = 0  
            
    return



SPEED_LIMIT = 5
def create_corners_occupation(behaviour_list, corners_list, speed_list):
    navigation_list = [] 
    exploration_list = []

    for day in range(len(behaviour_list)):
        speed = signal.medfilt(speed_list[day],9)
        # select the corners that are being explored when the animal is doing an exploratory task
        explorating_object = np.zeros_like(corners_list[day])
        explorating_object[np.where(behaviour_list[day]==4)[0]] = corners_list[day][np.where(behaviour_list[day]==4)[0]]
        explorating_object[np.where(behaviour_list[day]==3)[0]] = corners_list[day][np.where(behaviour_list[day]==3)[0]]
        #exploration_corner = np.zeros_like(explorating_object)
        # create a vector that contains zeros everywhere but object ID number when the animal is exploring an object a particular corner with low speed
        #exploration_corner[np.where(speed<SPEED_LIMIT)[0]] = navigation[np.where(speed<SPEED_LIMIT)[0]]
        #exploration_corner = explorating_object

        navigation_corner = np.zeros_like(corners_list[day])
        for corner in [1,2,3,4]:
            # create a vector that contains zeros everywhere but corner ID when the animal is navigation at that position with out an object
            navigations_at_corner = np.logical_and(explorating_object==0,corners_list[day]==corner)
            #print(len(np.where(navigations_at_corner)[0]))
            #navigation_corner[np.logical_and(navigations_at_corner, speed < SPEED_LIMIT)] = corner
            navigation_corner[navigations_at_corner] = corner

        navigation_list.append(navigation_corner)
        exploration_list.append(explorating_object)
        
    return navigation_list, exploration_list

def create_events_list(behaviour, N_SHUFFLINGS):    
    # for each day creates a list that counts and saves times of different events.
    events_day_list_1 = []
    events_day_list_shuffle_1 = []
    events_counter_day_list = []
    events_time_starts_day = []

    for day in range(len(behaviour)):
        events_list = []
        events_counter_list = []
        events_time_starts = []
        random_events = []
        start_counter = 100
        counter = 0
        for i in range(behaviour[day].shape[0]):
            if behaviour[day][i] != start_counter:
                events_list.append(start_counter) # contains a sequence of events ID
                events_counter_list.append(counter) # conteins the duration of each event
                events_time_starts.append(i) # contains the time at which event starts
                start_counter = behaviour[day][i]
                counter = 1
            else:
                counter = counter + 1    


        events_day_list_1.append(events_list)
        shufflings = []
        for j in range(N_SHUFFLINGS):
            shufflings.append(events_list.copy())
        events_day_list_shuffle_1.append(shufflings)
        events_counter_day_list.append(events_counter_list)
        events_time_starts_day.append(events_time_starts)
        
    return events_day_list_1, events_day_list_shuffle_1,events_counter_day_list,events_time_starts_day

def create_id_events(events_day_list_1, events_counter_day_list,events_time_starts_day,id_target):
    
    events_duration_list = []
    total_duration_list = []
    number_of_events_list = []
    events_id = []

    for day in range(len(events_day_list_1)):
        events_duration_day = []
        total_duration_day = []
        number_of_events_day = []
        events_id_day = []

        events = np.array(events_day_list_1[day])  # all events in a day (THIS IS EVENT ID)
        events_counter = np.array(events_counter_day_list[day]) #duration of all day events
        events_time = np.array(events_time_starts_day[day]) # start time of events in day

        for target in id_target:
            position_events = np.where(events == target)[0] # select events related to one specific ID
            events_duration_target = events_counter[position_events]   # take the duration of the events for that ID
            if(len(events_duration_target)):
                events_duration_day.append(events_duration_target)
                total_duration_day.append(np.sum(events_duration_target))
                number_of_events_day.append(len(events_duration_target ))
                events_id_day.append(target)

        events_duration_list.append(events_duration_day)
        total_duration_list.append(total_duration_day)
        number_of_events_list.append(number_of_events_day)
        events_id.append(events_id_day)
        
    return events_duration_list, total_duration_list,number_of_events_list,events_id

def balancing_visits(number_of_events_list,events_duration_list,events_id):

    ### Balancing the number of events for selected targets 
    events_day_list= []         # create a list with the selected index of the ID-list to make a balanced selection
    events_number_list = []
    events_day_list_shuffle = []
    for day in range(len(number_of_events_list)):

        arg_min_target_time = np.argmin(number_of_events_list[day])
        n_events = number_of_events_list[day][arg_min_target_time]
        events_number_list.append(n_events)
        events_list = []
        events_list_copy = []
        print('Number of events per day after balancing: ',n_events)
        #print(n_events)
        for target in range(len(events_id[day])):
            sorted_events = np.sort(events_duration_list[day][target]) #sort of events
            arg_sorted_events = np.argsort(events_duration_list[day][target]) #take the index sorted by duration of events
            selected_events = arg_sorted_events[0:n_events]   # take only the first (sorter duration) events
            events_list.append(selected_events)                           # save the position of long and balanced events
            events_list_copy.append(selected_events.copy())               # make a copy of this events to create a shuffle list

        events_day_list.append(events_list)                              #this list contains index that are selected from the list of index of a specific target
        events_day_list_shuffle.append(events_list_copy)

    return events_number_list,events_day_list, events_day_list_shuffle


def create_shuffling(events_day_list_1,events_day_list_shuffle_1,events_counter_day_list,events_time_starts_day,number_of_events_list,events_id,events_day_list,N_SHUFFLINGS):
### create shuffled behavioural labels that preserve balance and temporal structure
### the shuffling will be in visits or events (so we shuffle the labels of the complete event)

    for day in range(len(events_day_list_1)):
        events_shuffle = []
        events = np.array(events_day_list_1[day])
        events_counter = np.array(events_counter_day_list[day])
        events_time = np.array(events_time_starts_day[day])
        arg_min_target_time = np.argmin(number_of_events_list[day])
        n_events = number_of_events_list[day][arg_min_target_time]

        for target in range(len(events_id[day])):
            #print(target)
            #print(len(events_day_list[day]))
            #print(events_day_list[day][target])
            all_events = np.where(events == events_id[day][target])[0]
            #print(len(all_events))
            position_events = all_events[events_day_list[day][target]] # select the balanced data  
            events_shuffle.append(position_events)

        for j in range(N_SHUFFLINGS):
            counter_permutations = 0
            for i in range(n_events):
                permutation = np.random.permutation(len(events_id[day]))
                #print(permutation)
                counter_permutations +=1
                for index in range(len(events_id[day])):
                    events_day_list_shuffle_1[day][j][events_shuffle[index][i]] = events_id[day][permutation[index]]

    return events_day_list_shuffle_1

def create_activity_events(activity_list,period,events_day_list_1,events_counter_day_list,events_time_starts_day,events_id,events_day_list):
    
    ## put all events together and take neural activity from each event
    events_activity_pre_norm= []
    events_duration_list = []

    for day in range(len(activity_list)):
        target_activity = []
        events_duration_day = []

        events = np.array(events_day_list_1[day])
        events_counter = np.array(events_counter_day_list[day])
        events_time = np.array(events_time_starts_day[day])

        for target in range(len(events_id[day])):
            all_events = np.where(events == events_id[day][target])[0]
            #print(all_events)
            #print(events_day_list[day][target])
            position_events = all_events[events_day_list[day][target]] ### this contains the balanced events

            events_duration = events_counter[position_events]   # convert to seconds
            time = events_time[position_events]
            i = 0
            event_target = []

            #events_duration_target = np.zeros(len(events_duration),)
            events_duration_target = []
            for event in events_duration:
                if event and time[i]-period >0 and time[i]+period < activity_list[day].shape[1]:
                    local_activity = activity_list[day][:,time[i]-period:time[i]+period]

                    event_target.append(local_activity)
                    events_duration_target.append(events_duration[i])
                i = i + 1
            target_activity.append(event_target)
            events_duration_day.append(events_duration_target)

        events_activity_pre_norm.append(target_activity)
        events_duration_list.append(events_duration_day)

    return events_activity_pre_norm, events_duration_list



def create_activity_events_shuffle(activity_list,period,events_day_list_shuffle_1,events_counter_day_list,events_time_starts_day,events_id,events_day_list_shuffle, N_SHUFFLINGS):

    events_activity_pre_norm_shuffle= []

    for day in range(len(activity_list)):

        shufflings = []

        for j in range(N_SHUFFLINGS):
            target_activity_shuffle = []

            events_duration_day_shuffle = []
            for target in range(len(events_id[day])):
                events = np.array(events_day_list_shuffle_1[day][j])
                events_counter = np.array(events_counter_day_list[day])
                events_time = np.array(events_time_starts_day[day])

                all_events = np.where(events == events_id[day][target])[0]

                #print(all_events)
                #print(events_day_list[day][target])
                #print(events_day_list_shuffle[day][target])
                position_events = all_events[events_day_list_shuffle[day][target]]

                events_duration = events_counter[position_events]   # convert to seconds
                time = events_time[position_events]
                i = 0
                event_target = []
                #events_duration_target = np.zeros(len(events_duration),)
                events_duration_target = []
                for event in events_duration:
                    if event and time[i]-period >0 and time[i]+period < activity_list[day].shape[1]:
                        local_activity = activity_list[day][:,time[i]-period:time[i]+period]

                        event_target.append(local_activity)
                        #events_duration_target[i]=1
                        events_duration_target.append(events_duration[i])
                    i = i + 1
                target_activity_shuffle.append(event_target)

            shufflings.append(target_activity_shuffle)

        events_activity_pre_norm_shuffle.append(shufflings)

    return events_activity_pre_norm_shuffle


def create_visits_matrix(events_activity,events_id):
    
    for day in range(len(events_id)): 
        #print('Computing Trial matrices for DAY = ' + str(day) )

        trial_activity_vectors = np.zeros((len(events_id[day]),events_activity[day][0][0].shape[0],events_activity[day][0][0].shape[1]))
        j= 0    
        for target in range(len(events_id[day])):
            # real data! 
            #print('Target:' + str(target))
            #print('NUMBER OF VISITS:' + str(len(events_activity[day][target])))

            if events_activity[day][target] != []:

                trial_activity = np.zeros((events_activity[day][target][0].shape[0],events_activity[day][target][0].shape[1]))
                ### generate matrix with mean activity and entire trial repetitions activity
                for neuron in range(events_activity[day][target][0].shape[0]):
                    neuron_trial_activity = np.zeros((events_activity[day][target][1].shape[1],))
                    for trial in range(len(events_activity[day][target])):
                        if len(events_activity[day][target][trial][neuron,:]):
                            neuron_trial_activity += events_activity[day][target][trial][neuron,:]#/(np.max(events_activity[day][target][trial][neuron,:])-np.min(events_activity[day][target][trial][neuron,:]))   
                    neuron_trial_activity = neuron_trial_activity / len(events_activity[day][target])
                    trial_activity[neuron,:] = neuron_trial_activity
            trial_activity_vectors[j,:,:] = trial_activity
            j= j+1            

    return trial_activity_vectors

def create_visits_matrix_shufflings(events_activity_pre_norm_shuffle,events_id,N_SHUFFLINGS):
    for day in range(len(events_id)): 
        #print('Computing Trial matrices in SHUFFLINGS')                
        trial_activity_vectors_shuffle = np.zeros((N_SHUFFLINGS,len(events_id[day]),events_activity_pre_norm_shuffle[day][0][0][0].shape[0],events_activity_pre_norm_shuffle[day][0][0][0].shape[1]))

        for shuffle in range(N_SHUFFLINGS):
            j=0
            for target in range(len(events_id[day])):
                trial_activity = np.zeros((events_activity_pre_norm_shuffle[day][shuffle][target][0].shape[0],events_activity_pre_norm_shuffle[day][shuffle][target][0].shape[1]))
                ### generate matrix with mean activity and entire trial repetitions activity
                for neuron in range(events_activity_pre_norm_shuffle[day][shuffle][target][0].shape[0]):
                    neuron_trial_activity = np.zeros((events_activity_pre_norm_shuffle[day][shuffle][target][0].shape[1],))
                    for trial in range(len(events_activity_pre_norm_shuffle[day][shuffle][target])):
                        if len(events_activity_pre_norm_shuffle[day][shuffle][target][trial][neuron,:]):
                            neuron_trial_activity += events_activity_pre_norm_shuffle[day][shuffle][target][trial][neuron,:]#/(np.max(events_activity[day][target][trial][neuron,:])-np.min(events_activity[day][target][trial][neuron,:]))
                    trial_activity[neuron,:] = neuron_trial_activity / len(events_activity_pre_norm_shuffle[day][shuffle][target])
                trial_activity_vectors_shuffle[shuffle,j,:,:] = trial_activity
                j=j+1

    return trial_activity_vectors_shuffle

    
def compute_representational_distance(trial_activity_vectors, trial_activity_vectors_shuffle, n_components, N_SHUFFLINGS):
        
    distance_neural = np.zeros((trial_activity_vectors.shape[2],))

    for time in range(trial_activity_vectors.shape[2]):
        counter = 0
        for i in range(trial_activity_vectors.shape[0]):
            for j in range(i+1,trial_activity_vectors.shape[0]):
                distance_neural[time] += np.linalg.norm(trial_activity_vectors[i,0:n_components,time] - trial_activity_vectors[j,0:n_components,time])
                counter+=1
        distance_neural[time] = distance_neural[time]/counter

    distance_neural_shuffle = np.zeros((N_SHUFFLINGS,trial_activity_vectors.shape[2],))
    for shuffle in range(N_SHUFFLINGS):
        for time in range(trial_activity_vectors.shape[2]):
            counter = 0
            for i in range(trial_activity_vectors_shuffle.shape[1]):
                for j in range(i+1,trial_activity_vectors_shuffle.shape[1]):
                    distance_neural_shuffle[shuffle,time] += np.linalg.norm(trial_activity_vectors_shuffle[shuffle,i,0:n_components,time] - trial_activity_vectors_shuffle[shuffle,j,0:n_components,time])
                    counter = counter +1
                    #print(distance_neural_shuffle[shuffle,time])
            distance_neural_shuffle[shuffle,time] = distance_neural_shuffle[shuffle,time]/counter

    print(distance_neural_shuffle)
    distance_mean = np.nanmean(distance_neural_shuffle,axis=0)
    distance_std = np.nanstd(distance_neural_shuffle,axis=0)
    z_scored_distance = (distance_neural - distance_mean)/distance_std

    return distance_neural, z_scored_distance

def create_events_activity_data_transformation(activity_list,data_transformation,period,events,events_counter,events_onset,events_id,events_b):
    
    events_activity, events_duration = create_activity_events(activity_list,period,events,events_counter,events_onset,events_id,events_b)
    events_activity_pca, events_duration = create_activity_events(data_transformation.pca,period,events,events_counter,events_onset,events_id,events_b)
    events_activity_cca_time, events_duration = create_activity_events(data_transformation.cca_time,period,events,events_counter,events_onset,events_id,events_b)
    events_activity_cca_allo, events_duration = create_activity_events(data_transformation.cca_allo,period,events,events_counter,events_onset,events_id,events_b)
    events_activity_cca_ego, events_duration = create_activity_events(data_transformation.cca_ego,period,events,events_counter,events_onset,events_id,events_b)
    events_activity_lda, events_duration = create_activity_events(data_transformation.lda,period,events,events_counter,events_onset,events_id,events_b)
    
    events = namedtuple('events', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])    
    return events(events_activity,events_activity_pca,events_activity_cca_time,events_activity_cca_allo,events_activity_cca_ego,events_activity_lda)



def create_events_activity_data_transformation_shuffling(activity_list,data_transformation,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS):
    
    events_activity_shuffle = create_activity_events_shuffle(activity_list,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS)
    events_activity_shuffle_pca = create_activity_events_shuffle(data_transformation.pca,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS)
    events_activity_shuffle_cca_time = create_activity_events_shuffle(data_transformation.cca_time,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS)
    events_activity_shuffle_cca_allo = create_activity_events_shuffle(data_transformation.cca_allo,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS)
    events_activity_shuffle_cca_ego= create_activity_events_shuffle(data_transformation.cca_ego,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS)
    events_activity_shuffle_lda = create_activity_events_shuffle(data_transformation.lda,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS)

    events = namedtuple('events', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])    
    return events(events_activity_shuffle,events_activity_shuffle_pca,events_activity_shuffle_cca_time,events_activity_shuffle_cca_allo,events_activity_shuffle_cca_ego,events_activity_shuffle_lda)



def create_trial_activity_list(activity_events,events_id):
    
    trial_activity_vectors = create_visits_matrix(activity_events.neural,events_id)
    trial_activity_pca = create_visits_matrix(activity_events.pca,events_id)
    trial_activity_cca_time = create_visits_matrix(activity_events.cca_time,events_id)
    trial_activity_cca_allo = create_visits_matrix(activity_events.cca_allo,events_id)
    trial_activity_cca_ego = create_visits_matrix(activity_events.cca_ego,events_id)
    trial_activity_lda = create_visits_matrix(activity_events.lda,events_id)

    trial_activity = namedtuple('trial_activity', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])    
    return trial_activity(trial_activity_vectors,trial_activity_pca, trial_activity_cca_time,trial_activity_cca_allo,trial_activity_cca_ego,trial_activity_lda)

def create_trial_activity_list_shuffle(activity_events_shuffle,events_id, N_SHUFFLINGS):
    
    trial_activity_vectors = create_visits_matrix_shufflings(activity_events_shuffle.neural,events_id,N_SHUFFLINGS)
    trial_activity_pca = create_visits_matrix_shufflings(activity_events_shuffle.pca,events_id,N_SHUFFLINGS)
    trial_activity_cca_time = create_visits_matrix_shufflings(activity_events_shuffle.cca_time,events_id,N_SHUFFLINGS)
    trial_activity_cca_allo = create_visits_matrix_shufflings(activity_events_shuffle.cca_allo,events_id,N_SHUFFLINGS)
    trial_activity_cca_ego = create_visits_matrix_shufflings(activity_events_shuffle.cca_ego,events_id,N_SHUFFLINGS)
    trial_activity_lda= create_visits_matrix_shufflings(activity_events_shuffle.lda,events_id,N_SHUFFLINGS)

    trial_activity_shuffle = namedtuple('trial_activity_shuffle', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])
    return trial_activity_shuffle(trial_activity_vectors,trial_activity_pca, trial_activity_cca_time,trial_activity_cca_allo,trial_activity_cca_ego,trial_activity_lda)

def compute_distance_list(trial_activity, trial_activity_shuffle, N_SHUFFLINGS):

    distance_neural, z_scored_neural = compute_representational_distance(trial_activity.neural,trial_activity_shuffle.neural,trial_activity_etho.neural.shape[1], N_SHUFFLINGS)
    pca_components = np.where(data_transformation.variance_ratio[0]>0.7)[0][0]
    distance_pca, z_scored_pca = compute_representational_distance(trial_activity.pca,trial_activity_shuffle.pca,pca_components, N_SHUFFLINGS)
    distance_cca_time, z_scored_cca_time = compute_representational_distance(trial_activity.cca_time,trial_activity_shuffle.cca_time,trial_activity.cca_time.shape[1], N_SHUFFLINGS)
    distance_cca_allo, z_scored_cca_allo = compute_representational_distance(trial_activity.cca_allo,trial_activity_shuffle.cca_allo,trial_activity.cca_allo.shape[1], N_SHUFFLINGS)
    distance_cca_ego, z_scored_cca_ego = compute_representational_distance(trial_activity.cca_ego,trial_activity_shuffle.cca_ego,trial_activity.cca_ego.shape[1], N_SHUFFLINGS)
    distance_lda, z_scored_cca_lda = compute_representational_distance(trial_activity.lda,trial_activity_shuffle.lda,trial_activity.lda.shape[1], N_SHUFFLINGS)

    distance = namedtuple('distance', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])
    return distance(z_scored_neural, z_scored_pca, z_scored_cca_time, z_scored_cca_allo, z_scored_cca_ego, z_scored_cca_lda)

