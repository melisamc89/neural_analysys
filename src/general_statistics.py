import os
import numpy as np
import matplotlib.pyplot as plt
import configuration



def cross_correlation_matrix(neural_activity = None):

    '''
    This function creates the cross correlation (PCC) between time traces of calcium activity.
    Input : neural_activity -> numpy array matrix containing n_neurons rows X time columns
    Output: cross_matrix -> numoy array n_neurons X n_neurons matrix containing pearson correlation coefficient.
    '''
    n_neurons = neural_activity.shape[0]
    cross_matrix = np.zeros((neural_activity.shape[0],neural_activity.shape[0]))

    for i in range(n_neurons):
        for j in range(n_neurons):
            cross_matrix[i,j] = np.correlate(neural_activity[i,:],neural_activity[j,:])

    return cross_matrix