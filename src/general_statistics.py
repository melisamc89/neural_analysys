import os
import numpy as np
import matplotlib.pyplot as plt
import configuration
import math
from numpy import linalg as LA

def corr_matrix(neural_activity = None):

    '''
    This function creates the correlation between time traces of calcium activity, subtracting the mean
    Input : neural_activity -> numpy array matrix containing n_neurons rows X time columns
    Output: cross_matrix -> numoy array n_neurons X n_neurons matrix containing the correlation matrix
    '''
    n_neurons = neural_activity.shape[0]
    n_time = neural_activity.shape[1]

    mean_activity = np.mean(neural_activity,axis = 1)
    corr_matrix = np.dot((neural_activity - mean_activity[:,np.newaxis]), (neural_activity - mean_activity[:,np.newaxis]).T)
    corr_matrix = corr_matrix / math.sqrt(n_time * (n_time-1))

    return corr_matrix

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

    Proj_Act = np.dot(Eig_Mat ,np.dot(EigV_PC.T,neural_activity))


    return Proj_Act

##

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

