import os
import numpy as np
import matplotlib.pyplot as plt
import configuration
import math
from numpy import linalg as LA

def normalize_neural_activity(activity = None, timeline = None):

    '''
    Takes the neural activity and the timeline of concatenated points, and do a normalization
    for ever trial (segment of the entire signal)
    :param activity: np 2-D array with size (number_neurons+1) X (time). Activity consists of many concatenated
    trials
    :param timeline: Has the information about frames where the concatenation was done
    :return: the new normalized matrix and a nd-array with the timeline information
    '''

    timeline_vector = np.zeros(len(timeline) + 1)
    for i in range(len(timeline)):
        timeline_vector[i] = timeline[i][1]
    timeline_vector[len(timeline)] = activity.shape[1]
    ### do analysis corr, PCA
    ## normalize activity within trial and for each neuron
    activity_normalized = np.zeros((activity.shape))
    for j in range(activity.shape[0]):
        for i in range(0, len(timeline_vector) - 1):
            activity_segment = activity[j, int(timeline_vector[i]):int(timeline_vector[i + 1])]
            activity_segment = activity_segment - min(activity_segment)
            if max(activity_segment):
                activity_segment_normalized = activity_segment / max(activity_segment)
                activity_normalized[j, int(timeline_vector[i]):int(timeline_vector[i + 1])] = \
                    activity_segment_normalized
    neural_activity_normalized = activity_normalized[1:, :]

    delete_list = []
    sum_activity = np.sum(neural_activity_normalized, axis=1)
    for i in range(neural_activity_normalized.shape[0]):
        if sum_activity[i] == 0:
            delete_list.append(i)
    for i in delete_list:
        neural_activity_normalized = np.delete(neural_activity_normalized, i, 0)

    return neural_activity_normalized, timeline_vector

def resample_matrix(neural_activity= None, re_sf= 1):

    '''
    Resample the neural activity by the mean, and also gives the variance.
    :param neural_activity: 2D np.array of size (number_neurons+1) X (time)
    :param re_sf: int with resample value
    :return: resample mean activity and std
    '''
    reshape_neural_activity = np.reshape(neural_activity[:, :int(int(neural_activity.shape[1] / re_sf) * re_sf)],
                                         (neural_activity.shape[0], int(neural_activity.shape[1] / re_sf), re_sf))
    resample_neural_activity_mean = np.mean(reshape_neural_activity, axis=2)
    resample_neural_activity_std = np.std(reshape_neural_activity, axis=2)

    return resample_neural_activity_mean, resample_neural_activity_std

def corr_matrix(neural_activity = None):

    '''
    This function creates the correlation between time traces of calcium activity, subtracting the mean
    Input : neural_activity -> numpy array matrix containing n_neurons rows X time columns
    Output: cross_matrix -> numoy array n_neurons X n_neurons matrix containing the correlation matrix
    '''
    n_neurons = neural_activity.shape[0]
    n_time = neural_activity.shape[1]

    mean_activity = np.mean(neural_activity,axis = 1)
    std_activity = np.std(neural_activity,axis = 1)
    corr_matrix = np.dot((neural_activity - mean_activity[:,np.newaxis]), (neural_activity - mean_activity[:,np.newaxis]).T)
    #corr_matrix = corr_matrix / math.sqrt(n_time * (n_time-1))
    corr_matrix = corr_matrix / (n_time-1)
    for i in range(n_neurons):
        for j in range(n_neurons):
            if std_activity[i] > 0 and std_activity[j] > 0 :
                corr_matrix[i,j] = corr_matrix[i,j]/(std_activity[i]*std_activity[j])
            else:
                corr_matrix[i, j] = 0

    return corr_matrix

def cov_matrix(neural_activity = None):

    '''
    This function creates the correlation between time traces of calcium activity, subtracting the mean
    Input : neural_activity -> numpy array matrix containing n_neurons rows X time columns
    Output: cross_matrix -> numoy array n_neurons X n_neurons matrix containing the correlation matrix
    '''
    n_neurons = neural_activity.shape[0]
    n_time = neural_activity.shape[1]

    mean_activity = np.mean(neural_activity,axis = 1)
    cov_matrix = np.dot((neural_activity - mean_activity[:,np.newaxis]), (neural_activity - mean_activity[:,np.newaxis]).T)
    #corr_matrix = corr_matrix / math.sqrt(n_time * (n_time-1))
    cov_matrix = cov_matrix / (n_time-1)

    return cov_matrix

def compute_PCA(corr_matrix = None):

    '''
    Compute eigenvalues and eigenvectors of correlation matrix and returns it sorted
    :param corr_matrix:
    :return:
    '''
    eigenvalues, eigenvectors = LA.eig(corr_matrix)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors

def PCA_projection(neural_activity = None, eigenvalues = None, eigenvectors = None, n_components = None):

    '''
    Projects the neural activity to the PCA basis, and returns the activity of each neuron in that basis, using
    the number of principal components (n_components).
    :param neural_activity:
    :param eigenvalues:
    :param eigenvectectors:
    :param n_components:
    :return:
    '''

    ## construct eigenvalues diagonal matrix
    Eig_Mat =LA.matrix_power(np.sqrt(np.diag(eigenvalues)),-1)
    Eig_Mat = Eig_Mat[0:n_components,0:n_components]
    EigV_PC = eigenvectors[:,0:n_components]

    #Proj_Act = np.dot(Eig_Mat ,np.dot(EigV_PC.T,neural_activity))
    Proj_Act = np.dot(EigV_PC.T,neural_activity)

    return Proj_Act

def compute_DKL(p = None, q = None):
    dkl = np.dot(p + np.finfo(float).eps ,np.log(np.divide(p + np.finfo(float).eps,q + np.finfo(float).eps))+ np.finfo(float).eps)
    return dkl




##
'''
figure, axes = plt.subplots(1)
axes.imshow(np.log(corr_matrix))
figure.show()

figure, axes = plt.subplots(1)
axes.scatter(np.arange(1,eigenvalues.shape[0]+1),eigenvalues)
axes.set_xlabel('Order')
axes.set_ylabel('Eigenvalue')
figure.suptitle('Eigenvalue Spectrum')
figure.show()


n_components = 6
figure,axes = plt.subplots(n_components,1)
for i in range(n_components):
    axes[i].plot(np.arange(0,Proj_Act.shape[1])/10,Proj_Act[i,:])
    axes[i].set_ylabel('PC:' + f'{i}')
axes[5].set_xlabel('Time (s)')
figure.show()
'''
