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
        if session_now == 1 or session_now == 4:
            task = 'STABLE'
            colapse_behaviour = 2
            labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]
        if session_now == 2 or session_now == 6:
            task = 'RANDOM'
            colapse_behaviour = 0
            labels =['Unlabel','Rest1', 'Navigation', 'Obj1' , 'Obj2', 'Run1', 'Run2']
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]   
        if session_now == 3 or session_now == 5:
            task = 'OVERLAPPING'
            fixed = 'LL'
            if session_now == 3:
                object_fixed = 3
            if session_now == 5:
                object_fixed = 6
            colapse_behaviour = 1
            labels =['Unlabel','Rest1', 'Navigation', 'Overlap_object' , 'Moving_object','RunOO' , 'RunMO' ]
            colornames=['k',"r","deepskyblue","g","blue","g","blue"]   

    if mouse == 401714:
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
    
    trial_list = []

    print('LOADING TRIALS ACTIVITY AND CREATING LIST OF ACTIVITY, TRACKING AND BEHAVIOUR')
    for trial in [1,6,11,16]:

        beh_file_name_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75_ethogram.npy'
        beh_file_name_2 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75_object_corners.npy'
        speed_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75_speed.npy'
        beh_file_name_3= 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + f'{day+1}' + '_likelihood_0.75_ethogram_parameters.npy'
        tracking_file_name_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75.npy'
        time_file_session_1 =  'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_'+ f'{trial}'+'_v1.3.1.0_10.pkl'
        calcium_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_'+ f'{trial}'+'_v' + f'{decoding_v}' + '.4.' + f'{motion_correction_v}' + '.' + f'{alignment_v}' + '.' + f'{equalization_v}' + '.' + f'{source_extraction_v}' + '.' + f'{component_evaluation_v}' +  '.0.npy'
    
        if os.path.isfile(behaviour_dir + beh_file_name_1) and os.path.isfile(behaviour_dir + beh_file_name_2) and os.path.isfile(behaviour_dir + speed_file_name) and os.path.isfile(behaviour_dir_parameters + beh_file_name_3) and os.path.isfile(tracking_dir + tracking_file_name_1) and os.path.isfile(timeline_file_dir + time_file_session_1) and os.path.isfile(file_directory + calcium_file_name):
            
            ## LOAD BEHAVIOUR
            behaviour = np.load(behaviour_dir + beh_file_name_1)
            reshape_behaviour = np.reshape(behaviour[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
            resample_beh1 = np.reshape(scipy.stats.mode(reshape_behaviour,axis=1)[0],reshape_behaviour.shape[0])
            ## LOAD CORNERS EXPLORATION
            behaviour = np.load(behaviour_dir + beh_file_name_2)
            reshape_behaviour = np.reshape(behaviour[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
            corners = np.reshape(scipy.stats.mode(reshape_behaviour,axis=1)[0],reshape_behaviour.shape[0])
            ## LOAD INSTANTANEOUS SPEED
            speed = np.load(behaviour_dir + speed_file_name)
            reshape_speed = np.reshape(speed[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
            resample_speed = np.reshape(scipy.stats.mode(reshape_speed,axis=1)[0],reshape_speed.shape[0])

            parameters = np.load(behaviour_dir_parameters + beh_file_name_3)

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
            position = np.load(tracking_dir + tracking_file_name_1)
            resample_position, resample_position_stc = stats.resample_matrix(neural_activity=position.T,
                                                                                                    re_sf=re_sf)    
            
            ## LOAD TIMELINE
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
            
            
            activity = np.load(file_directory + calcium_file_name)
            neural_activity1 = activity[1:,:]
            ## z-score neural activity
            neural_activity = neural_activity1[:,:int(int(behaviour.shape[0]/re_sf)*re_sf)]
            ##downsample neural activity
            resample_neural_activity_mean, resample_neural_activity_std = stats.resample_matrix(neural_activity=neural_activity,
                                                                                                    re_sf=re_sf)
            
            
            behaviour_list.append(resample_beh1)
            corners_list.append(corners)
            speed_list.append(resample_speed)
            parameters_list.append(resample_params)
            parameters_list2.append(resample_params2)
            parameters_time.append(resample_params0)
            tracking_list.append(resample_position)
            total_time = total_time + behaviour.shape[0]
            activity_list.append(resample_neural_activity_mean)
            
            
            print('neural shape =  ' +  str(resample_neural_activity_mean.shape) + ' beh shape' + str(resample_beh1.shape))

            day = day + 1
            
            trial_list.append(day)
            
        else:
            #return activity_list,timeline_list,behaviour_list,corners_list,parameters_time,parameters_list,parameters_list2,speed_list
            day = day+1
            print(day)
            trial_list.append(0)
            continue


    
    return activity_list,timeline_list,behaviour_list,corners_list,parameters_time,parameters_list,parameters_list2,speed_list,trial_list



def load_data_trial(mouse = None, session = None, decoding_v = None, motion_correction_v = None, alignment_v = None, equalization_v = None, source_extraction_v = None, component_evaluation_v = None, re_sf = None,file_directory = None, timeline_file_dir = None, behaviour_dir = None, behaviour_dir_parameters=None, tracking_dir = None, objects_dir = None):


    activity_list_trial = []
    behaviour_list_trial = []
    parameters_list_trial = []
    corners_list_trial = []
    speed_list_trial = []

    behaviour_list_unsup = []
    parameters_list = []
    parameters_list2 = []
    parameters_time = []
    tracking_list = []
    total_time = 0
    day = 0
    activity_list = []
    timeline_list = []
    behaviour_list = []
    corners_list = []
    speed_list = []
    
    trial_list = np.zeros((20,))


    print('LOADING TRIALS ACTIVITY AND CREATING LIST OF ACTIVITY, TRACKING AND BEHAVIOUR')
    for trial in [1,6,11,16]:

        beh_file_name_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75_ethogram.npy'
        beh_file_name_2 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75_object_corners.npy'
        speed_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75_speed.npy'
        beh_file_name_3= 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_' + f'{day+1}' + '_likelihood_0.75_ethogram_parameters.npy'
        tracking_file_name_1 = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_day_' + f'{day+1}' + '_likelihood_0.75.npy'
        time_file_session_1 =  'mouse_'+ f'{mouse}'+'_session_'+ f'{session}' +'_trial_'+ f'{trial}'+'_v1.3.1.0_10.pkl'
  
        calcium_file_name = 'mouse_' + f'{mouse}' + '_session_' + f'{session}' + '_trial_'+ f'{trial}'+'_v' + f'{decoding_v}' + '.4.' + f'{motion_correction_v}' + '.' + f'{alignment_v}' + '.' + f'{equalization_v}' + '.' + f'{source_extraction_v}' + '.' + f'{component_evaluation_v}' +  '.0.npy'
    
        if os.path.isfile(behaviour_dir + beh_file_name_1) and os.path.isfile(behaviour_dir + beh_file_name_2) and os.path.isfile(behaviour_dir + speed_file_name) and os.path.isfile(behaviour_dir_parameters + beh_file_name_3) and os.path.isfile(tracking_dir + tracking_file_name_1) and os.path.isfile(timeline_file_dir + time_file_session_1) and os.path.isfile(file_directory + calcium_file_name):
           
            ## LOAD BEHAVIOUR
            behaviour = np.load(behaviour_dir + beh_file_name_1)
            reshape_behaviour = np.reshape(behaviour[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
            resample_beh1 = np.reshape(scipy.stats.mode(reshape_behaviour,axis=1)[0],reshape_behaviour.shape[0])
            ## LOAD CORNERS EXPLORATION
            behaviour = np.load(behaviour_dir + beh_file_name_2)
            reshape_behaviour = np.reshape(behaviour[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
            corners = np.reshape(scipy.stats.mode(reshape_behaviour,axis=1)[0],reshape_behaviour.shape[0])
            ## LOAD INSTANTANEOUS SPEED
            speed = np.load(behaviour_dir + speed_file_name)
            reshape_speed = np.reshape(speed[:int(int(behaviour.shape[0]/re_sf)*re_sf)],(int(behaviour.shape[0]/re_sf),re_sf))
            resample_speed = np.reshape(scipy.stats.mode(reshape_speed,axis=1)[0],reshape_speed.shape[0])

            parameters = np.load(behaviour_dir_parameters + beh_file_name_3)

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
            position = np.load(tracking_dir + tracking_file_name_1)
            resample_position, resample_position_stc = stats.resample_matrix(neural_activity=position.T,
                                                                                                    re_sf=re_sf)    
            
            ## LOAD TIMELINE
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
            
            
            activity = np.load(file_directory + calcium_file_name)
            neural_activity1 = activity[1:,:]
            ## z-score neural activity
            neural_activity = neural_activity1[:,:int(int(behaviour.shape[0]/re_sf)*re_sf)]
            ##downsample neural activity
            resample_neural_activity_mean, resample_neural_activity_std = stats.resample_matrix(neural_activity=neural_activity,
                                                                                                    re_sf=re_sf)
            

            #activity_list.append(resample_neural_activity_mean)
            for i in range(5):
                time1 = int(resample_timeline[2*i])
                time2 = int(resample_timeline[2*i+1])
                activity_list.append(resample_neural_activity_mean[:,time1:time2])
                behaviour_list.append(resample_beh1[time1:time2])
                speed_list.append(resample_speed[time1:time2])
                parameters_list.append(resample_params[:,time1:time2])
                corners_list.append(corners[time1:time2])
                parameters_list2.append(resample_params2[:,time1:time2])
                parameters_time.append(resample_params0[:,time1:time2])
                print('neural shape =  ' +  str(resample_neural_activity_mean.shape) + ' beh shape' + str(resample_beh1.shape))
                trial_list[day*5+i] = day*5+i+1
            day = day + 1
        else:
            #return activity_list,timeline_list,behaviour_list,corners_list,parameters_time,parameters_list,parameters_list2,speed_list
            trial_list[day*5 : day*5+5] = np.zeros((5,))
            day = day+1
            continue
            
    return activity_list,timeline_list,behaviour_list,corners_list,parameters_time,parameters_list,parameters_list2,speed_list,trial_list


def transform_neural_data(activity_list, behaviour_list,parameters_time,parameters_list,parameters_list2,trial_list):
    
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
    trial_list_new = np.zeros_like(trial_list)
    for day in range(len(trial_list)):
        if trial_list[day]:
            if activity_list[day].shape[1] == behaviour_list[day].shape[0]:
                trial_list_new[day] = trial_list[day]
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
            else:
                trial_list_new[day] = 0
                activity_list_pca.append([])
                variance_list.append([])
                variance_ratio_list.append([])
                activity_list_lda.append([])
                activity_list_cca0.append([])
                activity_list_cca.append([])
                activity_list_cca2.append([])   
        else:
            trial_list_new[day] = 0
            activity_list_pca.append([])
            variance_list.append([])
            variance_ratio_list.append([])
            activity_list_lda.append([])
            activity_list_cca0.append([])
            activity_list_cca.append([])
            activity_list_cca2.append([])
            continue
    
    data_transformation = namedtuple('data_transformation', ['pca','variance_ratio','cca_time','cca_allo','cca_ego','lda','trials'])    
    return data_transformation(activity_list_pca,variance_ratio_list,activity_list_cca0,activity_list_cca,activity_list_cca2,activity_list_lda,trial_list_new)
    

    
def create_task_behaviour(behaviour_list,colapse_behaviour,object_fixed,timeline_list):
    
    # define targets of behaviour according to protocol (random, overlapping and stable)
    id_target = [0,1,2,3,4] # 0: unlabeled, 1:resting, 2:Navigation, 3: Obj1, 4:Obj2, 5:Run1, 6:Run2
    
    print('REDIFINING BEHAVIOUR FOR DIFFERENT SESSIONS')
    if colapse_behaviour == 0 : # RANDOM
        for day in range(len(behaviour_list)):
            for trial in range(5):
                behaviour_trial = behaviour_list[day][int(timeline_list[day][2*trial]):int(timeline_list[day][2*trial+1])]
                objects = np.unique(behaviour_trial)
                if len(objects)>4:
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
            if len(objects)>4:
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

def create_task_behaviour_trial(behaviour_list,colapse_behaviour,object_fixed,timeline_list):
    
    # define targets of behaviour according to protocol (random, overlapping and stable)
    id_target = [0,1,2,3,4] # 0: unlabeled, 1:resting, 2:Navigation, 3: Obj1, 4:Obj2, 5:Run1, 6:Run2
    
    if colapse_behaviour == 0 : # RANDOM
        for trial in range(len(behaviour_list)):
            behaviour_trial = behaviour_list[trial]
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

            behaviour_list[trial] = behaviour_trial

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
            if len(objects)>4:
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

def balancing_visits(number_of_events_list,events_duration_list,events_id, balance_flag = False):

    if balance_flag == True:
        ### Balancing the number of events for selected targets 
        events_day_list= []         # create a list with the selected index of the ID-list to make a balanced selection
        events_number_list = []
        events_day_list_shuffle = []
        for day in range(len(number_of_events_list)):
            if len(number_of_events_list[day]):
                arg_min_target_time = np.argmin(number_of_events_list[day])
                n_events = number_of_events_list[day][arg_min_target_time]
            else:
                n_events = 0
            events_number_list.append(n_events)
            events_list = []
            events_list_copy = []
            #print('Number of events per day after balancing: ',n_events)
            #print(n_events)
            for target in range(len(events_id[day])):
                sorted_events = np.sort(events_duration_list[day][target]) #sort of events
                arg_sorted_events = np.argsort(events_duration_list[day][target]) #take the index sorted by duration of events
                selected_events = arg_sorted_events[0:n_events]   # take only the first (sorter duration) events
                events_list.append(selected_events)                           # save the position of long and balanced events
                events_list_copy.append(selected_events.copy())               # make a copy of this events to create a shuffle list

            events_day_list.append(events_list)                              #this list contains index that are selected from the list of index of a specific target
            events_day_list_shuffle.append(events_list_copy)

    else: 
        events_day_list= []         # create a list with the selected index of the ID-list to make a balanced selection
        events_number_list = []
        events_day_list_shuffle = []
        for day in range(len(number_of_events_list)):
            if len(number_of_events_list[day]):
                arg_min_target_time = np.argmin(number_of_events_list[day])
                n_events = number_of_events_list[day][arg_min_target_time]
            else:
                n_events = 0
            events_number_list.append(n_events)
            events_list = []
            events_list_copy = []
            #print('Number of events per day after balancing: ',n_events)
            #print(n_events)
            for target in range(len(events_id[day])):
                sorted_events = np.sort(events_duration_list[day][target]) #sort of events
                arg_sorted_events = np.argsort(events_duration_list[day][target]) #take the index sorted by duration of events
                selected_events = arg_sorted_events#[0:n_events]   # take only the first (sorter duration) events
                events_list.append(selected_events)                           # save the position of long and balanced events
                events_list_copy.append(selected_events.copy())               # make a copy of this events to create a shuffle list

            events_day_list.append(events_list)                              #this list contains index that are selected from the list of index of a specific target
            events_day_list_shuffle.append(events_list_copy)
        events_number_list = number_of_events_list
        
    return events_number_list,events_day_list, events_day_list_shuffle


def create_shuffling(events_day_list_1,events_day_list_shuffle_1,events_counter_day_list,events_time_starts_day,number_of_events_list,events_id,events_day_list,N_SHUFFLINGS):
### create shuffled behavioural labels that preserve balance and temporal structure
### the shuffling will be in visits or events (so we shuffle the labels of the complete event)

    for day in range(len(events_day_list_1)):
        events_shuffle = []
        events = np.array(events_day_list_1[day])
        events_counter = np.array(events_counter_day_list[day])
        events_time = np.array(events_time_starts_day[day])
        if len(number_of_events_list[day]):
            arg_min_target_time = np.argmin(number_of_events_list[day])
            n_events = number_of_events_list[day][arg_min_target_time]
        else:
            n_events = 0 
        #print(n_events)

        for target in range(len(events_id[day])):
            #print(target)
            #print(len(events_day_list[day]))
            #print(events_day_list[day][target])
            all_events = np.where(events == events_id[day][target])[0]
            #print(len(all_events))
            position_events = all_events[events_day_list[day][target]] # select the balanced data  
            #print(position_events)
            events_shuffle.append(position_events)
            
        #print(len(events_day_list_shuffle_1[day][0]))
        for j in range(N_SHUFFLINGS):
            #counter_permutations = 0
            for i in range(n_events):
                permutation = np.random.permutation(len(events_id[day]))
                #print(permutation)
                #counter_permutations +=1
                for index in range(len(events_id[day])):
                    #print(events_day_list_shuffle_1[day][j][events_shuffle[index][i]])
                    events_day_list_shuffle_1[day][j][events_shuffle[index][i]] = events_id[day][permutation[index]]
                    #print(events_day_list_shuffle_1[day][j][events_shuffle[index][i]])

    return events_day_list_shuffle_1

def create_activity_events(activity_list,period,events_day_list_1,events_counter_day_list,events_time_starts_day,events_id,events_day_list,trial_list):
    
    ## put all events together and take neural activity from each event
    events_activity_pre_norm= []
    events_duration_list = []
    
    day = 0
    for all_counter in range(len(trial_list)):
        if trial_list[all_counter]:
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
            day = day +1
        else:
            events_activity_pre_norm.append([])
            events_duration_list.append([])
            day = day +1

    return events_activity_pre_norm, events_duration_list


def create_activity_events_shuffle(activity_list,period,events_day_list_shuffle_1,events_counter_day_list,events_time_starts_day,events_id,events_day_list_shuffle, N_SHUFFLINGS,trial_list):

    events_activity_pre_norm_shuffle= []
    day = 0
    for all_counter in range(len(trial_list)):

        if trial_list[all_counter]:
            shufflings = []
            for j in range(N_SHUFFLINGS):
                target_activity_shuffle = []

                events_duration_day_shuffle = []
                for target in range(len(events_id[day])):

                    events = np.array(events_day_list_shuffle_1[day][j])

                    events_counter = np.array(events_counter_day_list[day])
                    events_time = np.array(events_time_starts_day[day])

                    all_events = np.where(events == events_id[day][target])[0]

                    #print(len(all_events))
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
            day = day +1
        else:
            events_activity_pre_norm_shuffle.append([])
            day = day +1
            
    return events_activity_pre_norm_shuffle


def create_visits_matrix(events_activity,events_id,trials_list):
    
    trial_activity_vectors_list = []
    day = 0
    new_trial_list = np.zeros_like(trials_list)
    for all_counter in range(len(trials_list)): 
        if trials_list[all_counter] and events_activity[day][0] != []:
            new_trial_list[all_counter] = trials_list[all_counter]
            trial_activity_vectors = np.zeros((len(events_id[day]),events_activity[day][0][0].shape[0],events_activity[day][0][0].shape[1]))
            j= 0    
            for target in range(len(events_activity[day])):
                if events_activity[day][target] != []:
                    trial_activity = np.zeros((events_activity[day][target][0].shape[0],events_activity[day][target][0].shape[1]))
                    ### generate matrix with mean activity and entire trial repetitions activity
                    for neuron in range(events_activity[day][target][0].shape[0]):
                        neuron_trial_activity = np.zeros((events_activity[day][target][0].shape[1],))
                        for trial in range(len(events_activity[day][target])):
                            if len(events_activity[day][target][trial][neuron,:]):
                                neuron_trial_activity += events_activity[day][target][trial][neuron,:]#/(np.max(events_activity[day][target][trial][neuron,:])-np.min(events_activity[day][target][trial][neuron,:]))   
                        neuron_trial_activity = neuron_trial_activity / len(events_activity[day][target])
                        trial_activity[neuron,:] = neuron_trial_activity
                trial_activity_vectors[j,:,:] = trial_activity
                j= j+1
            trial_activity_vectors_list.append(trial_activity_vectors)
            day = day +1
            
        else:
            trial_activity_vectors_list.append([])
            day = day+1

    return trial_activity_vectors_list, new_trial_list

def create_visits_matrix_shufflings(events_activity_pre_norm_shuffle,events_id,N_SHUFFLINGS,trials_list):
    
    trial_activity_vectors_shuffle_list = []
    day = 0
    for all_counter in range(len(trials_list)): 
        if trials_list[all_counter] and events_activity_pre_norm_shuffle[day][0][0] != []:
            #print('Computing Trial matrices in SHUFFLINGS')                
            trial_activity_vectors_shuffle = np.zeros((N_SHUFFLINGS,len(events_id[day]),events_activity_pre_norm_shuffle[day][0][0][0].shape[0],events_activity_pre_norm_shuffle[day][0][0][0].shape[1]))
            for shuffle in range(N_SHUFFLINGS):
                j=0
                for target in range(len(events_activity_pre_norm_shuffle[day][shuffle])):
                    if events_activity_pre_norm_shuffle[day][shuffle][target] != []:
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

            trial_activity_vectors_shuffle_list.append(trial_activity_vectors_shuffle)
            day = day+1
        else:
            trial_activity_vectors_shuffle_list.append([])
            day=day+1
    return trial_activity_vectors_shuffle_list

    
def compute_representational_distance(trial_activity_vectors, trial_activity_vectors_shuffle, n_components, N_SHUFFLINGS):

    distance = []
    distance_zs = []
    for day in range(len(trial_activity_vectors)):
        distance_neural = np.zeros((trial_activity_vectors[day].shape[2],))

        for time in range(trial_activity_vectors[day].shape[2]):
            counter = 0
            for i in range(trial_activity_vectors[day].shape[0]):
                for j in range(i+1,trial_activity_vectors[day].shape[0]):
                    distance_neural[time] += np.linalg.norm(trial_activity_vectors[day][i,0:n_components[day],time] - trial_activity_vectors[day][j,0:n_components[day],time])
                    counter+=1
            distance_neural[time] = distance_neural[time]/counter

        distance_neural_shuffle = np.zeros((N_SHUFFLINGS,trial_activity_vectors[day].shape[2],))
        for shuffle in range(N_SHUFFLINGS):
            for time in range(trial_activity_vectors[day].shape[2]):
                counter = 0
                for i in range(trial_activity_vectors_shuffle[day].shape[1]):
                    for j in range(i+1,trial_activity_vectors_shuffle[day].shape[1]):
                        distance_neural_shuffle[shuffle,time] += np.linalg.norm(trial_activity_vectors_shuffle[day][shuffle][i,0:n_components[day],time] - trial_activity_vectors_shuffle[day][shuffle][j,0:n_components[day],time])
                        counter = counter +1
                        #print(distance_neural_shuffle[shuffle,time])
                distance_neural_shuffle[shuffle,time] = distance_neural_shuffle[shuffle,time]/counter

        #print(distance_neural_shuffle)
        distance_mean = np.nanmean(distance_neural_shuffle,axis=0)
        distance_std = np.nanstd(distance_neural_shuffle,axis=0)
        z_scored_distance = (distance_neural - distance_mean)/distance_std
        distance.append(distance_neural)
        distance_zs.append(z_scored_distance)

    return distance, distance_zs


def create_events_activity_data_transformation(activity_list,data_transformation,period,events,events_counter,events_onset,events_id,events_b,trial_list):
    
    events_activity, events_duration = create_activity_events(activity_list,period,events,events_counter,events_onset,events_id,events_b,trial_list)
    events_activity_pca, events_duration = create_activity_events(data_transformation.pca,period,events,events_counter,events_onset,events_id,events_b,trial_list)
    events_activity_cca_time, events_duration = create_activity_events(data_transformation.cca_time,period,events,events_counter,events_onset,events_id,events_b,trial_list)
    events_activity_cca_allo, events_duration = create_activity_events(data_transformation.cca_allo,period,events,events_counter,events_onset,events_id,events_b,trial_list)
    events_activity_cca_ego, events_duration = create_activity_events(data_transformation.cca_ego,period,events,events_counter,events_onset,events_id,events_b,trial_list)
    events_activity_lda, events_duration = create_activity_events(data_transformation.lda,period,events,events_counter,events_onset,events_id,events_b,trial_list)
    
    events = namedtuple('events', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])    
    return events(events_activity,events_activity_pca,events_activity_cca_time,events_activity_cca_allo,events_activity_cca_ego,events_activity_lda)



def create_events_activity_data_transformation_shuffling(activity_list,data_transformation,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS,trial_list):
    
    events_activity_shuffle = create_activity_events_shuffle(activity_list,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS,trial_list)
    events_activity_shuffle_pca = create_activity_events_shuffle(data_transformation.pca,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS,trial_list)
    events_activity_shuffle_cca_time = create_activity_events_shuffle(data_transformation.cca_time,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS,trial_list)
    events_activity_shuffle_cca_allo = create_activity_events_shuffle(data_transformation.cca_allo,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS,trial_list)
    events_activity_shuffle_cca_ego= create_activity_events_shuffle(data_transformation.cca_ego,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS,trial_list)
    events_activity_shuffle_lda = create_activity_events_shuffle(data_transformation.lda,period,events_shuffle_b,events_counter,events_onset,events_id,events_s_b, N_SHUFFLINGS,trial_list)

    events = namedtuple('events', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])    
    return events(events_activity_shuffle,events_activity_shuffle_pca,events_activity_shuffle_cca_time,events_activity_shuffle_cca_allo,events_activity_shuffle_cca_ego,events_activity_shuffle_lda)



def create_trial_activity_list(activity_events,events_id,trial_list):
    
    trial_activity_vectors,new_list = create_visits_matrix(activity_events.neural,events_id,trial_list)
    trial_activity_pca,new_list = create_visits_matrix(activity_events.pca,events_id,trial_list)
    trial_activity_cca_time,new_list = create_visits_matrix(activity_events.cca_time,events_id,trial_list)
    trial_activity_cca_allo,new_list = create_visits_matrix(activity_events.cca_allo,events_id,trial_list)
    trial_activity_cca_ego,new_list = create_visits_matrix(activity_events.cca_ego,events_id,trial_list)
    trial_activity_lda,new_list = create_visits_matrix(activity_events.lda,events_id,trial_list)

    trial_activity = namedtuple('trial_activity', ['neural','pca','cca_time','cca_allo','cca_ego','lda','trials'])    
    return trial_activity(trial_activity_vectors,trial_activity_pca, trial_activity_cca_time,trial_activity_cca_allo,trial_activity_cca_ego,trial_activity_lda,new_list)

def create_trial_activity_list_shuffle(activity_events_shuffle,events_id, N_SHUFFLINGS,trial_list):
    
    trial_activity_vectors = create_visits_matrix_shufflings(activity_events_shuffle.neural,events_id,N_SHUFFLINGS,trial_list)
    trial_activity_pca = create_visits_matrix_shufflings(activity_events_shuffle.pca,events_id,N_SHUFFLINGS,trial_list)
    trial_activity_cca_time = create_visits_matrix_shufflings(activity_events_shuffle.cca_time,events_id,N_SHUFFLINGS,trial_list)
    trial_activity_cca_allo = create_visits_matrix_shufflings(activity_events_shuffle.cca_allo,events_id,N_SHUFFLINGS,trial_list)
    trial_activity_cca_ego = create_visits_matrix_shufflings(activity_events_shuffle.cca_ego,events_id,N_SHUFFLINGS,trial_list)
    trial_activity_lda= create_visits_matrix_shufflings(activity_events_shuffle.lda,events_id,N_SHUFFLINGS,trial_list)

    trial_activity_shuffle = namedtuple('trial_activity_shuffle', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])
    return trial_activity_shuffle(trial_activity_vectors,trial_activity_pca, trial_activity_cca_time,trial_activity_cca_allo,trial_activity_cca_ego,trial_activity_lda)

def compute_distance_list(trial_activity, trial_activity_shuffle, data_transformation, N_SHUFFLINGS):
    
    neural_components = []
    pca_components = []
    cca_time_components = []
    cca_allo_components = []
    cca_ego_components = []
    lda_components = []
    for day in range(len(trial_activity.neural)):
        neural_components.append(trial_activity.neural[day].shape[1])
        pca_components.append(np.where(data_transformation.variance_ratio[0]>0.7)[0][0])
        cca_time_components.append(trial_activity.cca_time[day].shape[1])
        cca_allo_components.append(trial_activity.cca_allo[day].shape[1])
        cca_ego_components.append(trial_activity.cca_ego[day].shape[1])
        lda_components.append(trial_activity.lda[day].shape[1])
                         
    distance_neural, z_scored_neural = compute_representational_distance(trial_activity.neural,trial_activity_shuffle.neural,neural_components, N_SHUFFLINGS)
    distance_pca, z_scored_pca = compute_representational_distance(trial_activity.pca,trial_activity_shuffle.pca,pca_components, N_SHUFFLINGS)
    distance_cca_time, z_scored_cca_time = compute_representational_distance(trial_activity.cca_time,trial_activity_shuffle.cca_time,cca_time_components, N_SHUFFLINGS)
    distance_cca_allo, z_scored_cca_allo = compute_representational_distance(trial_activity.cca_allo,trial_activity_shuffle.cca_allo,cca_allo_components, N_SHUFFLINGS)
    distance_cca_ego, z_scored_cca_ego = compute_representational_distance(trial_activity.cca_ego,trial_activity_shuffle.cca_ego,cca_ego_components, N_SHUFFLINGS)
    distance_lda, z_scored_cca_lda = compute_representational_distance(trial_activity.lda,trial_activity_shuffle.lda,lda_components, N_SHUFFLINGS)

    distance = namedtuple('distance', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])
    return distance(z_scored_neural, z_scored_pca, z_scored_cca_time, z_scored_cca_allo, z_scored_cca_ego, z_scored_cca_lda)


def compute_representational_distance_measures(activity_list,data_transformation,period,behaviour_list,id_target,N_SHUFFLINGS,trials_list):
    
    print('CREATE LIST THAT SAVES ALL THE EVENTS IN A DAY AND CONTAINS ONSET OF VISITS')
    events_etho, events_shuffle_etho,events_counter_etho,events_onset_etho = create_events_list(behaviour_list, N_SHUFFLINGS)
    print('NOW WE SEPARATE EVENTS TYPES ACCORDING TO CORNER/OBJECT VISIT')
    events_duration_etho, total_duration_etho,number_of_events_etho,events_id_etho = create_id_events(events_etho, events_counter_etho,events_onset_etho,id_target)
    print('BALANCING TO THE LOWER NUMBER OF VISITS')
    events_number_etho, events_etho_b, events_etho_s_b = balancing_visits(number_of_events_etho,events_duration_etho,events_id_etho)
    print('CREATE SHUFFLE LABELS THAT PRESERVE BALANCE')
    events_etho_shuffle_b = create_shuffling(events_etho,events_shuffle_etho,events_counter_etho,events_onset_etho,number_of_events_etho,events_id_etho,events_etho_b,N_SHUFFLINGS)
    
    print('TAKING NEURAL OR TRANSFORMED ACTIVITY FOR EACH EVENT ... and create list with that')
    activity_events_etho = create_events_activity_data_transformation(activity_list,data_transformation,period,events_etho,events_counter_etho,events_onset_etho,events_id_etho,events_etho_b,trials_list)
    activity_events_etho_shuffling = create_events_activity_data_transformation_shuffling(activity_list,data_transformation,period,events_etho_shuffle_b,events_counter_etho,events_onset_etho,events_id_etho,events_etho_s_b, N_SHUFFLINGS,trials_list)
    print('CREATING VISITIS ACTIVITY MATRICES')
    trial_activity_etho = create_trial_activity_list(activity_events_etho,events_id_etho,trials_list)
    trial_activity_shuffle_etho = create_trial_activity_list_shuffle(activity_events_etho_shuffling,events_id_etho, N_SHUFFLINGS,trials_list)
    print('CREATING DISTANCE tuple')
    distance = compute_distance_list(trial_activity_etho,trial_activity_shuffle_etho, data_transformation, N_SHUFFLINGS)

    return distance



def save_distance(mouse,session,distance,task,day_flag = True):
    
    for day in range(len(distance.neural)):
        final_distance = np.zeros((6, 200))
        final_distance[0,:] = distance.neural[day]
        final_distance[1,:] = distance.pca[day]
        final_distance[2,:] = distance.cca_time[day]
        final_distance[3,:] = distance.cca_allo[day]
        final_distance[4,:] = distance.cca_ego[day]
        final_distance[5,:] = distance.lda[day]

        print('Saving')
        if day_flag:
            output_path =  os.environ['PROJECT_DIR'] +'neural_analysis/data/mean_representational_distance/'
            file_name = output_path + task +'_distance_day_mouse_'+f'{mouse}'+'_session_'+f'{session}'+ '_day_'+f'{day}'
            np.save(file_name, final_distance)

        else:
            output_path =  os.environ['PROJECT_DIR'] +'neural_analysis/data/mean_representational_distance/'
            file_name = output_path+ task +'_distance_trial_mouse_'+f'{mouse}'+'_session_'+f'{session}'+ '_trial_'+f'{day}'
            np.save(file_name, final_distance)           

    return


def compute_representational_distance_all_to_all(trial_activity_vectors, trial_activity_vectors_shuffle, n_components, N_SHUFFLINGS, trial_list, trial_flag = False):

    distance = []
    distance_zs = []
        
    distance_matrix = np.zeros((32,32,trial_activity_vectors[0].shape[2]))
    distance_matrix_shuffle = np.zeros((N_SHUFFLINGS,32,32,trial_activity_vectors[0].shape[2]))
    day_conditions = 8
    if trial_flag :
        distance_matrix = np.zeros((80,80,trial_activity_vectors[0].shape[2]))
        distance_matrix_shuffle = np.zeros((N_SHUFFLINGS,80,80,trial_activity_vectors[0].shape[2]))
        day_conditions = 4
        trial_vector = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
    
    for day1 in range(len(trial_list)):
        
        # if trial_list[day1]:
            # ### compute distances within trial
            # distance_neural = np.zeros((day_conditions,day_conditions,trial_activity_vectors[day1].shape[2]))
            # if trial_activity_vectors[day1].shape[0] == day_conditions:
            #     for time in range(trial_activity_vectors[day1].shape[2]):
            #         for i in range(trial_activity_vectors[day1].shape[0]):
            #             for j in range(trial_activity_vectors[day1].shape[0]):
            #                 distance_neural[i,j,time] = np.linalg.norm(trial_activity_vectors[day1][i,0:n_components[day1],time] - trial_activity_vectors[day1][j,0:n_components[day1],time])
            # distance_matrix[day1*day_conditions:(day1+1)*day_conditions,day1*day_conditions:(day1+1)*day_conditions,:] = distance_neural

        for day2 in range(day1,len(trial_list)):
            if trial_list[day1] and trial_list[day2]:
                if trial_activity_vectors[day1].shape[0] == day_conditions and trial_activity_vectors[day2].shape[0] == day_conditions:
                    if trial_vector[day1] == trial_vector[day2]:
                        for time in range(trial_activity_vectors[day1].shape[2]):
                            for i in range(trial_activity_vectors[day1].shape[0]):
                                for j in range(trial_activity_vectors[day2].shape[0]):
                                    x1 = trial_activity_vectors[day1][i,0:n_components[day1],time]
                                    x2 =  trial_activity_vectors[day2][j,0:n_components[day2],time]
                                    x = np.linalg.norm(x1 - x2)
                                    distance_matrix[day1*day_conditions+i,day2*day_conditions+j,time] = x
                                    for shuffle in range(N_SHUFFLINGS):
                                        x1 = trial_activity_vectors_shuffle[day1][shuffle][i,0:n_components[day1],time]
                                        x2 = trial_activity_vectors_shuffle[day2][shuffle][j,0:n_components[day2],time]
                                        x = np.linalg.norm(x1-x2)
                                        distance_matrix_shuffle[shuffle,day1*day_conditions+i,day2*day_conditions+j,time] = x

            # distance_neural_shuffle = np.zeros((N_SHUFFLINGS,day_conditions,day_conditions,trial_activity_vectors[day1].shape[2]))
            # for shuffle in range(N_SHUFFLINGS):
            #     if trial_activity_vectors_shuffle[day1][shuffle].shape[0] == day_conditions:            
            #         for time in range(trial_activity_vectors_shuffle[day1][shuffle].shape[2]):
            #             for i in range(trial_activity_vectors_shuffle[day1][shuffle].shape[0]):
            #                 for j in range(trial_activity_vectors_shuffle[day1][shuffle].shape[0]):
            #                     distance_neural_shuffle[shuffle,i,j,time] = np.linalg.norm(trial_activity_vectors_shuffle[day1][shuffle][i,0:n_components[day1],time] - trial_activity_vectors_shuffle[day1][shuffle][j,0:n_components[day1],time])
            # distance_matrix_shuffle[:,day1*day_conditions:(day1+1)*day_conditions,day1*day_conditions:(day1+1)*day_conditions,:] = distance_neural_shuffle

#             #for day2 in range(day1,len(trial_list)):
#                 # if trial_list[day2]:
#                     for shuffle in range(N_SHUFFLINGS):
#                         if trial_activity_vectors_shuffle[day1][shuffle].shape[0] == day_conditions and trial_activity_vectors_shuffle[day2][shuffle].shape[0] == day_conditions :
#                             if trial_vector[day1] == trial_vector[day2]:
#                                 for time in range(trial_activity_vectors_shuffle[day1][shuffle].shape[2]):
#                                     for i in range(trial_activity_vectors_shuffle[day1][shuffle].shape[0]):
#                                         for j in range(trial_activity_vectors_shuffle[day2][shuffle].shape[0]):
#                                             # print(distance_matrix_shuffle[shuffle,day1*day_conditions+i,day2*day_conditions+j,time])
#                                             # print(trial_activity_vectors_shuffle[day1][i,0:n_components[day1],time].shape)
#                                             # print(trial_activity_vectors_shuffle[day2][j,0:n_components[day2],time].shape)
#                                             # print(np.linalg.norm(trial_activity_vectors_shuffle[day1][i,0:n_components[day1],time] - trial_activity_vectors_shuffle[day2][j,0:n_components[day2],time]).shape)
#                                             x1 = trial_activity_vectors_shuffle[day1][shuffle][i,0:n_components[day1],time]
#                                             x2 = trial_activity_vectors_shuffle[day2][shuffle][j,0:n_components[day2],time]
#                                             x = np.linalg.norm()
            
#                                             distance_matrix_shuffle[shuffle,day1*day_conditions+i,day2*day_conditions+j,time] = x
            
    #print(distance_neural_shuffle)
    #distance_mean = np.nanmean(distance_matrix_shuffle,axis=0)
    #distance_std = np.nanstd(distance_matrix_shuffle,axis=0)
    #z_scored_distance = (distance_matrix - distance_mean)/distance_std
        
    mean_distance = np.nanmean(distance_matrix,axis = 2)
    mean_distance_shuffling =  np.nanmean(distance_matrix_shuffle,axis = 3)
    # for i in range(mean_distance.shape[0]):
    #     mean_distance[i,i] = 0
    #     mean_distance_shuffling[:,i,i] = np.zeros_like( mean_distance_shuffling[:,i,i])

                
    mean_distance_zs = (mean_distance - np.nanmean(mean_distance_shuffling,axis = 0)) / np.nanstd(mean_distance_shuffling, axis = 0)
    print(mean_distance.shape)

    return mean_distance,  mean_distance_zs



def compute_distance_list_all_to_all(trial_activity, trial_activity_shuffle, data_transformation, N_SHUFFLINGS,trial_list,trial_flag = False):
    
    neural_components = []
    pca_components = []
    cca_time_components = []
    cca_allo_components = []
    cca_ego_components = []
    lda_components = []
    
    for day in range(len(trial_activity.neural)):
        if trial_list[day]:
            neural_components.append(trial_activity.neural[day].shape[1])
            pca_components.append(np.where(data_transformation.variance_ratio[0]>0.7)[0][0])
            cca_time_components.append(trial_activity.cca_time[day].shape[1])
            cca_allo_components.append(trial_activity.cca_allo[day].shape[1])
            cca_ego_components.append(trial_activity.cca_ego[day].shape[1])
            lda_components.append(trial_activity.lda[day].shape[1])
        else:
            neural_components.append([])
            pca_components.append([])
            cca_time_components.append([])
            cca_allo_components.append([])
            cca_ego_components.append([])
            lda_components.append([])

    distance_neural, z_scored_neural = compute_representational_distance_all_to_all(trial_activity.neural,trial_activity_shuffle.neural,neural_components, N_SHUFFLINGS,trial_list,trial_flag)
    distance_pca, z_scored_pca = compute_representational_distance_all_to_all(trial_activity.pca,trial_activity_shuffle.pca,pca_components, N_SHUFFLINGS,trial_list,trial_flag)
    distance_cca_time, z_scored_cca_time = compute_representational_distance_all_to_all(trial_activity.cca_time,trial_activity_shuffle.cca_time,cca_time_components, N_SHUFFLINGS,trial_list,trial_flag)
    distance_cca_allo, z_scored_cca_allo = compute_representational_distance_all_to_all(trial_activity.cca_allo,trial_activity_shuffle.cca_allo,cca_allo_components, N_SHUFFLINGS,trial_list,trial_flag)
    distance_cca_ego, z_scored_cca_ego = compute_representational_distance_all_to_all(trial_activity.cca_ego,trial_activity_shuffle.cca_ego,cca_ego_components, N_SHUFFLINGS,trial_list,trial_flag)
    distance_lda, z_scored_lda = compute_representational_distance_all_to_all(trial_activity.lda,trial_activity_shuffle.lda,lda_components, N_SHUFFLINGS,trial_list, trial_flag)

    
    non_nan_z_scored_neural = np.nan_to_num( np.nansum([z_scored_neural,z_scored_neural.T],axis = 0),neginf=0)
    non_nan_z_scored_pca = np.nan_to_num( np.nansum([z_scored_pca,z_scored_pca.T],axis = 0),neginf=0)
    non_nan_z_scored_cca_time = np.nan_to_num( np.nansum([z_scored_cca_time,z_scored_cca_time.T],axis = 0),neginf=0)
    non_nan_z_scored_cca_allo = np.nan_to_num( np.nansum([z_scored_cca_allo,z_scored_cca_allo.T],axis = 0),neginf=0)
    non_nan_z_scored_cca_ego = np.nan_to_num( np.nansum([z_scored_cca_ego,z_scored_cca_ego.T],axis = 0),neginf=0)
    non_nan_z_scored_lda = np.nan_to_num( np.nansum([z_scored_lda,z_scored_lda.T],axis = 0),neginf=0)
    
    distance = namedtuple('distance', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])
    return distance(non_nan_z_scored_neural, non_nan_z_scored_pca, non_nan_z_scored_cca_time, non_nan_z_scored_cca_allo, non_nan_z_scored_cca_ego, non_nan_z_scored_lda)




def compute_representational_distance_measures_all_to_all(activity_list,data_transformation,period,behaviour_list,id_target,N_SHUFFLINGS,trial_list, trial_flag = False):
    
    print('CREATE LIST THAT SAVES ALL THE EVENTS IN A DAY AND CONTAINS ONSET OF VISITS')
    events_etho, events_shuffle_etho,events_counter_etho,events_onset_etho = create_events_list(behaviour_list, N_SHUFFLINGS)
    print('NOW WE SEPARATE EVENTS TYPES ACCORDING TO CORNER/OBJECT VISIT')
    events_duration_etho, total_duration_etho,number_of_events_etho,events_id_etho = create_id_events(events_etho, events_counter_etho,events_onset_etho,id_target)
    print('BALANCING TO THE LOWER NUMBER OF VISITS')
    events_number_etho, events_etho_b, events_etho_s_b = balancing_visits(number_of_events_etho,events_duration_etho,events_id_etho)
    print('CREATE SHUFFLE LABELS THAT PRESERVE BALANCE')
    events_etho_shuffle_b = create_shuffling(events_etho,events_shuffle_etho,events_counter_etho,events_onset_etho,number_of_events_etho,events_id_etho,events_etho_b,N_SHUFFLINGS)
    print(events_number_etho)
    
    print('TAKING NEURAL OR TRANSFORMED ACTIVITY FOR EACH EVENT ... and create list with that')
    activity_events_etho = create_events_activity_data_transformation(activity_list,data_transformation,period,events_etho,events_counter_etho,events_onset_etho,events_id_etho,events_etho_b,trial_list)
    activity_events_etho_shuffling = create_events_activity_data_transformation_shuffling(activity_list,data_transformation,period,events_etho_shuffle_b,events_counter_etho,events_onset_etho,events_id_etho,events_etho_s_b, N_SHUFFLINGS,trial_list)
    print('CREATING VISITIS ACTIVITY MATRICES')
    trial_activity_etho = create_trial_activity_list(activity_events_etho,events_id_etho,trial_list)
    trial_activity_shuffle_etho = create_trial_activity_list_shuffle(activity_events_etho_shuffling,events_id_etho, N_SHUFFLINGS,trial_list)
    print('CREATING DISTANCE tuple')
    distance = compute_distance_list_all_to_all(trial_activity_etho,trial_activity_shuffle_etho, data_transformation, N_SHUFFLINGS, trial_list, trial_flag)
    return trial_activity_etho, trial_activity_shuffle_etho, distance




def mean_distance_occupancy(distance_matrix,index_corners_trial,occupancy_mask):
    
    corners_distance = np.zeros((4,20))
    for i in range(4):
        corners_distance[i,:] = distance_matrix[index_corners_trial[i,0,:],index_corners_trial[i,1,:]]

    distance_mean = np.zeros((2,4))

    for i in range(4):
        values = corners_distance[i,occupancy_mask[i,:]]
        nonzero_values = values[np.where(values)]
        distance_mean[0,i] = np.mean(nonzero_values)
        values = corners_distance[i,np.where(occupancy_mask[i,:]==False)]
        nonzero_values = values[np.where(values)]
        distance_mean[1,i] = np.mean(nonzero_values)   
        
    return distance_mean

def mean_distance_occupancy_transformations(distance, index_corners_trial, occupancy_mask):
    
    
    d_neural = mean_distance_occupancy(distance.neural, index_corners_trial, occupancy_mask)
    d_pca = mean_distance_occupancy(distance.pca, index_corners_trial, occupancy_mask)
    d_cca_time = mean_distance_occupancy(distance.cca_time, index_corners_trial, occupancy_mask)
    d_cca_allo = mean_distance_occupancy(distance.cca_allo, index_corners_trial, occupancy_mask)
    d_cca_ego = mean_distance_occupancy(distance.cca_ego, index_corners_trial, occupancy_mask)
    d_lda = mean_distance_occupancy(distance.lda, index_corners_trial, occupancy_mask)
    
    distance = namedtuple('distance', ['neural','pca','cca_time','cca_allo','cca_ego','lda'])
    return distance(d_neural, d_pca, d_cca_time, d_cca_allo, d_cca_ego, d_lda)