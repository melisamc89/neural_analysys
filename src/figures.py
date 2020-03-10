'''
Created on Fri 28 Feb 2020
Author: Melisa

Plotting

'''
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pickle
import configuration
import general_statistics as stats
import matplotlib.cm as cm
from matplotlib import colors
import general_statistics as gstats
from scipy import signal
import scipy
from mpl_toolkits.mplot3d import axes3d
from IPython.display import HTML
import matplotlib.animation as animation


def plot_correlation_matrix(corr_matrix = None, save_path = None, title = None):

    '''
    Plots One correlation matrix for one condition
    :param corr_matrix:
    :param save_path:
    :param title:
    :return:
    '''

    figure, axes = plt.subplots(1)
    axes.imshow(corr_matrix, cmap = 'viridis')
    axes.set_title(title)
    figure.suptitle('Correlation Matrix')
    figure.colorbar(corr_matrix, ax=axes, orientation='vertical')
    # figure.suptitle('Correlation Matrix. Bin size:' + f'{re_sf}' + 'frames' , fontsize = 15)
    figure.savefig(save_path)

    return


def plot_correlation_matrix_conditions(matrix_list = None, save_path = None, title = None , conditions = None):

    '''
    Plots multiple correlation matrix. As long as matrix list and conditions have the same size and proper information,
    can be use to plot all sessions of one mouse or also multiple mouse (use the correct labeling in conditions)
    :param matrix_list:
    :param save_path:
    :param title:
    :param conditions:
    :return:
    '''

    size = int(math.sqrt((len(matrix_list))))+1
    figure, axes = plt.subplots(size, size)
    images = []
    counter = 0
    for i in range(size):
        for j in range(size):
            if counter < len(matrix_list):
                images.append(axes[i, j].imshow(np.log10(matrix_list[counter]), cmap='viridis'))
                counter = counter+1
                axes[i, j].label_outer()

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    counter = 0
    for i in range(size):
        for j in range(size):
            if counter < len(matrix_list):
                axes[i, j].set_title(conditions[counter])
                counter = counter +1
    figure.colorbar(images[0], ax=axes[1, 1], orientation='vertical', fraction=0.1)
    figure.suptitle(title , fontsize = 15)
    figure.savefig(save_path)

    return

def plot_correlation_matrix_behaviour(corr_matrix_list = None , path_save = None, title = None):


    figure, axes = plt.subplots(3, 2)
    images = []
    for i in range(3):
        for j in range(2):
            images.append(axes[i, j].imshow(np.log10(corr_matrix_list[i * 2 + j]), cmap='viridis'))
            axes[i, j].label_outer()
            figure.colorbar(images[i], ax=axes[i, j])

    vmin = min(image.get_array().min() for image in images)
    vmax = 0
    # vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    # figure.colorbar(images[0], ax=axes, orientation='vertical', fraction=0.1)

    # axes.legend(['LR', 'LL', 'UR', 'UL'])
    axes[0, 0].set_title('Resting', fontsize=12)
    axes[0, 1].set_title('Not exploring', fontsize=12)
    axes[1, 0].set_title('LR', fontsize=12)
    axes[1, 1].set_title('LL', fontsize=12)
    axes[2, 0].set_title('UR', fontsize=12)
    axes[2, 1].set_title('UL', fontsize=12)
    figure.suptitle(title)
    figure.savefig(path_save)

    return

def plot_correlation_statistics_behaviour(corr_matrix = None, path_save = None):

    corr_mean = np.zeros(len(corr_matrix))
    corr_error = np.zeros(len(corr_matrix))
    max_corr = 0
    for i in range(len(corr_matrix)):
        corr_mean[i] = np.mean(corr_matrix[i].flatten())
        corr_error[i] = np.std(corr_matrix[i].flatten())/math.sqrt(corr_matrix[i].flatten().shape[0])
        max_value = np.max(corr_matrix[i].flatten())
        if max_value > max_corr:
            max_corr = max_value

    min_corr= 0.0001
    max_corr = 0.002
    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(3, 12)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_title('Resting', fontsize = 15)
    [counter,bin_num] = np.histogram(corr_matrix[0].flatten(),bins=np.arange(0.0, max_corr, max_corr / 20))
    ax1.fill_between(bin_num[:-1],counter / np.sum(counter))
    ax1.set_ylim(0,1)

    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.set_title('Not exploring', fontsize = 15)
    [counter,bin_num] = np.histogram(corr_matrix[1].flatten(),bins=np.arange(0.0, max_corr, max_corr / 20))
    ax2.fill_between(bin_num[:-1],counter / np.sum(counter))
    ax2.set_ylim(0,1)

    ax3 = fig.add_subplot(gs[0, 4:6])
    ax3.set_title('LR', fontsize = 15)
    [counter,bin_num] = np.histogram(corr_matrix[2].flatten(),bins=np.arange(0.0, max_corr, max_corr / 20))
    ax3.fill_between(bin_num[:-1],counter / np.sum(counter))
    ax3.set_ylim(0,1)

    ax4 = fig.add_subplot(gs[0,6:8])
    ax4.set_title('LL', fontsize = 15)
    [counter,bin_num] = np.histogram(corr_matrix[3].flatten(),bins=np.arange(0.0, max_corr, max_corr / 20))
    ax4.fill_between(bin_num[:-1],counter / np.sum(counter))
    ax4.set_ylim(0,1)

    ax5 = fig.add_subplot(gs[0,8:10])
    ax5.set_title('UR', fontsize = 15)
    [counter,bin_num] = np.histogram(corr_matrix[4].flatten(),bins=np.arange(0.0, max_corr, max_corr / 20))
    ax5.fill_between(bin_num[:-1],counter / np.sum(counter))
    ax5.set_ylim(0,1)

    ax6 = fig.add_subplot(gs[0,10:12])
    ax6.set_title('UL', fontsize = 15)
    [counter,bin_num] = np.histogram(corr_matrix[5].flatten(),bins=np.arange(0.0, max_corr, max_corr / 20))
    ax6.fill_between(bin_num[:-1],counter / np.sum(counter))
    ax6.set_ylim(0,1)

    ax7 = fig.add_subplot(gs[1:3, 0:4])
    conditions= ['Rest', 'NotE', 'LR','LL','UR','UL']
    x_pos = np.arange(len(conditions))
    ax7.bar(x_pos, corr_mean, yerr=corr_error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax7.set_ylabel('Mean Correlation')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(conditions)
    ax7.set_title('Correlation statistics', fontsize = 15)
    ax7.yaxis.grid(True)
    ax7.set_ylim(0,np.max(corr_mean)+5*np.max(corr_error))
    fig.tight_layout()


    ax8 = fig.add_subplot(gs[1:3, 4:8])
    ax8.set_title('Correlation', fontsize = 15)
    corr_of_corr = np.zeros((len(corr_matrix),len(corr_matrix)))
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            correlation = np.corrcoef(corr_matrix[i].flatten(), corr_matrix[j].flatten())
            corr_of_corr[i,j] = correlation[0,1]
    x = ax8.imshow(corr_of_corr,cmap = 'gray')
    fig.colorbar(x, ax=ax8)


    ax9 = fig.add_subplot(gs[1:3, 8:12])
    ax9.set_title('KLD', fontsize = 15)
    dkl_matrix = np.zeros((len(corr_matrix), len(corr_matrix)))
    for i in range(len(corr_matrix)):
        [x1, bin_num1] = np.histogram(corr_matrix[i].flatten(),
                                          bins=np.arange(0.0, max_corr, max_corr / 20))
        for j in range(len(corr_matrix)):
            [y1, bin_num2] = np.histogram(corr_matrix[j].flatten(),
                                                bins=np.arange(0.0, max_corr, max_corr / 20))
            # figures.colorbar(x, ax=axes[0, i])
            dkl_matrix[i, j] = gstats.compute_DKL(x1 / np.sum(x1), y1 / np.sum(y1))
    x = ax9.imshow(dkl_matrix,cmap = 'viridis')
    fig.colorbar(x, ax=ax9)

    fig.set_size_inches(20, 9)
    fig.savefig(path_save)

    return


def plot_correlation_statistics_learning(corr_matrix1 = None, corr_matrix2=None, path_save = None,title = None):

    corr_mean1 = np.zeros(len(corr_matrix1))
    corr_error1 = np.zeros(len(corr_matrix1))
    max_corr = 0
    for i in range(len(corr_matrix1)):
        corr_mean1[i] = np.mean(corr_matrix1[i].flatten())
        corr_error1[i] = np.std(corr_matrix1[i].flatten())/math.sqrt(corr_matrix1[i].flatten().shape[0])
        max_value = np.max(corr_matrix1[i].flatten())
        if max_value > max_corr:
            max_corr = max_value

    corr_mean2 = np.zeros(len(corr_matrix2))
    corr_error2 = np.zeros(len(corr_matrix2))
    max_corr = 0
    for i in range(len(corr_matrix2)):
        corr_mean2[i] = np.mean(corr_matrix2[i].flatten())
        corr_error2[i] = np.std(corr_matrix2[i].flatten())/math.sqrt(corr_matrix2[i].flatten().shape[0])
        max_value = np.max(corr_matrix2[i].flatten())
        if max_value > max_corr:
            max_corr = max_value

    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('In trial period', fontsize=12)
    conditions = ['day1', 'day2', 'day3', 'day4', 'Test']
    x_pos = np.arange(len(conditions))
    ax1.bar(x_pos, corr_mean1, yerr=corr_error1, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax1.set_ylabel('Mean Correlation')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(conditions)
    ax1.yaxis.grid(True)
    ax1.set_ylim(0, np.max(corr_mean1) + 5 * np.max(corr_error1))
    fig.tight_layout()

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Resting period', fontsize=12)
    conditions = ['day1', 'day2', 'day3', 'day4', 'Test']
    x_pos = np.arange(len(conditions))
    ax2.bar(x_pos, corr_mean2, yerr=corr_error2, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax2.set_ylabel('Mean Correlation')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(conditions)
    ax2.yaxis.grid(True)
    ax2.set_ylim(0, np.max(corr_mean1) + 5 * np.max(corr_error1))
    fig.tight_layout()

    ax3 = fig.add_subplot(gs[0, 1])
    ax3.set_title('Correlation', fontsize = 12)
    corr_of_corr = np.zeros((len(corr_matrix1),len(corr_matrix1)))
    for i in range(len(corr_matrix1)):
        for j in range(len(corr_matrix1)):
            correlation = np.corrcoef(corr_matrix1[i].flatten(), corr_matrix1[j].flatten())
            corr_of_corr[i,j] = correlation[0,1]
    x = ax3.imshow(np.log10(corr_of_corr),cmap = 'gray')
    fig.colorbar(x, ax=ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Correlation', fontsize = 12)
    corr_of_corr = np.zeros((len(corr_matrix2),len(corr_matrix2)))
    for i in range(len(corr_matrix2)):
        for j in range(len(corr_matrix2)):
            correlation = np.corrcoef(corr_matrix2[i].flatten(), corr_matrix2[j].flatten())
            corr_of_corr[i,j] = correlation[0,1]
    x = ax4.imshow(np.log10(corr_of_corr),cmap = 'gray')
    fig.colorbar(x, ax=ax4)


    ax5 = fig.add_subplot(gs[0, 2])
    ax5.set_title('KLD', fontsize = 12)
    dkl_matrix = np.zeros((len(corr_matrix1), len(corr_matrix1)))
    for i in range(len(corr_matrix1)):
        [x1, bin_num1] = np.histogram(corr_matrix1[i].flatten(),
                                          bins=np.arange(0.0, max_corr, max_corr / 20))
        for j in range(len(corr_matrix1)):
            [y1, bin_num2] = np.histogram(corr_matrix1[j].flatten(),
                                                bins=np.arange(0.0, max_corr, max_corr / 20))
            # figures.colorbar(x, ax=axes[0, i])
            dkl_matrix[i, j] = gstats.compute_DKL(x1 / np.sum(x1), y1 / np.sum(y1))
    x = ax5.imshow(dkl_matrix,cmap = 'viridis')
    fig.colorbar(x, ax=ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('KLD', fontsize = 12)
    dkl_matrix = np.zeros((len(corr_matrix2), len(corr_matrix2)))
    for i in range(len(corr_matrix2)):
        [x1, bin_num1] = np.histogram(corr_matrix2[i].flatten(),
                                          bins=np.arange(0.0, max_corr, max_corr / 20))
        for j in range(len(corr_matrix2)):
            [y1, bin_num2] = np.histogram(corr_matrix2[j].flatten(),
                                                bins=np.arange(0.0, max_corr, max_corr / 20))
            # figures.colorbar(x, ax=axes[0, i])
            dkl_matrix[i, j] = gstats.compute_DKL(x1 / np.sum(x1), y1 / np.sum(y1))
    x = ax6.imshow(dkl_matrix,cmap = 'viridis')
    fig.colorbar(x, ax=ax6)

    fig.set_size_inches(20, 9)
    fig.suptitle(title,fontsize = 15)
    fig.savefig(path_save)

    return


def plot_correlation_statistics_trials(corr_matrix1 = None, corr_matrix2 = None, path_save = None, title = None):

    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Correlation trials', fontsize = 12)
    corr_of_corr = np.zeros((len(corr_matrix1),len(corr_matrix1)))
    for i in range(len(corr_matrix1)):
        for j in range(len(corr_matrix1)):
            correlation = np.corrcoef(corr_matrix1[i].flatten(), corr_matrix1[j].flatten())
            corr_of_corr[i,j] = correlation[0,1]
    x = ax1.imshow(np.log10(corr_of_corr),cmap = 'gray')
    fig.colorbar(x, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Correlation resting', fontsize = 12)
    corr_of_corr = np.zeros((len(corr_matrix2),len(corr_matrix2)))
    for i in range(len(corr_matrix1)):
        for j in range(len(corr_matrix2)):
            correlation = np.corrcoef(corr_matrix2[i].flatten(), corr_matrix2[j].flatten())
            corr_of_corr[i,j] = correlation[0,1]
    x = ax2.imshow(np.log10(corr_of_corr),cmap = 'gray')
    fig.colorbar(x, ax=ax2)

    dkl_matrix1 = np.zeros((len(corr_matrix1),len(corr_matrix2)))
    dkl_matrix2= np.zeros((len(corr_matrix1),len(corr_matrix2)))
    for i in range(len(corr_matrix1)):
        x1 = np.histogram(corr_matrix1[i].flatten()[np.where(corr_matrix1[i].flatten() > 0.01)],
                          bins=np.arange(0.01, 0.05, 0.04 / 15))
        x2 = np.histogram(corr_matrix2[i].flatten()[np.where(corr_matrix2[i].flatten() > 0.01)],
                          bins=np.arange(0.01, 0.05, 0.04 / 15))
        for j in range(len(corr_matrix1)):
            y1 = np.histogram(corr_matrix1[j].flatten()[np.where(corr_matrix1[j].flatten() > 0.01)],
                              bins=np.arange(0.01, 0.05, 0.04 / 15))
            y2 = np.histogram(
                corr_matrix2[j].flatten()[np.where(corr_matrix2[j].flatten() > 0.01)],
                bins=np.arange(0.01, 0.05, 0.04 / 15))
            # figures.colorbar(x, ax=axes[0, i])
            dkl_matrix1[i, j] = stats.compute_DKL(x1[0] / np.sum(x1[0]), y1[0] / np.sum(y1[0]))
            dkl_matrix2[i, j] = stats.compute_DKL(x2[0] / np.sum(x2[0]), y2[0] / np.sum(y2[0]))


    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('KLD trials', fontsize = 12)
    x = ax3.imshow(dkl_matrix1,cmap = 'viridis')
    fig.colorbar(x, ax=ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('KLD resting', fontsize=12)
    x = ax4.imshow(dkl_matrix2, cmap='viridis')
    fig.colorbar(x, ax=ax4)

    fig.set_size_inches(20, 9)
    fig.suptitle(title,fontsize = 15)
    fig.savefig(path_save)

    return

def plot_correlation_statistics_objects(corr_matrix1 = None, corr_matrix2 = None, overlapping_matrix = None,path_save = None, title = None):

    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Correlation trials', fontsize=12)
    corr_of_corr1 = np.zeros((len(corr_matrix1), len(corr_matrix1)))
    for i in range(len(corr_matrix1)):
        for j in range(len(corr_matrix1)):
            correlation = np.corrcoef(corr_matrix1[i].flatten(), corr_matrix1[j].flatten())
            corr_of_corr1[i, j] = correlation[0, 1]
    x = ax1.imshow(np.log10(corr_of_corr1), cmap='gray')
    fig.colorbar(x, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Correlation resting', fontsize=12)
    corr_of_corr2 = np.zeros((len(corr_matrix2), len(corr_matrix2)))
    for i in range(len(corr_matrix1)):
        for j in range(len(corr_matrix2)):
            correlation = np.corrcoef(corr_matrix2[i].flatten(), corr_matrix2[j].flatten())
            corr_of_corr2[i, j] = correlation[0, 1]
    x = ax2.imshow(np.log10(corr_of_corr2), cmap='gray')
    fig.colorbar(x, ax=ax2)

    aux1 = []
    aux2 = []
    aux3 = []
    for i in range(21):
        for j in range(i + 1, 21):
            aux1.append(corr_of_corr1[i, j])
            aux2.append(corr_of_corr2[i, j])
            aux3.append(overlapping_matrix[i, j])

    correlation1 = np.corrcoef(np.array(aux1), np.array(aux3))
    corr_value1 = round(correlation1[0, 1], 2)
    ax1.set_title('Correlation trials, C:'+ f'{corr_value1}', fontsize=15)

    correlation2 = np.corrcoef(np.array(aux2), np.array(aux3))
    corr_value2 = round(correlation2[0, 1], 2)
    ax2.set_title('Correlation trials, C:'+ f'{corr_value2}', fontsize=15)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('Objects Overlapping', fontsize=15)
    x = ax3.imshow(overlapping_matrix, cmap='viridis')
    fig.colorbar(x, ax=ax3)

    fig.set_size_inches(10, 10)
    fig.suptitle('OBJECTS POSITIONS: ' + title, fontsize=15)
    fig.savefig(path_save)

    return

def plot_correlation_with_resting_evolution(corr_matrix1 = None, corr_matrix2 = None,path_save = None):

    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(2,1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Correlation with post resting activity', fontsize=15)
    corr_evolution = np.zeros((len(corr_matrix1),1))
    for i in range(len(corr_matrix1)):
        correlation = np.corrcoef(corr_matrix1[i].flatten(), corr_matrix2[i].flatten())
        corr_evolution[i] = correlation[0, 1]

    ax1.plot(np.arange(1,len(corr_matrix1)+1), corr_evolution)
    ax1.set_ylabel('Correlation')
    ax1.set_ylim([0,1])
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Correlation with pre resting activity', fontsize=15)
    corr_evolution = np.zeros((len(corr_matrix1),1))
    for i in range(1,len(corr_matrix1)-1):
        correlation = np.corrcoef(corr_matrix1[i+1].flatten(), corr_matrix2[i].flatten())
        corr_evolution[i] = correlation[0, 1]

    ax2.plot(np.arange(1,len(corr_matrix1)+1), corr_evolution)
    ax2.set_ylabel('Correlation')
    ax2.set_ylim([0,1])
    ax2.set_xlabel('Trials')
    fig.savefig(path_save)

    return


def plot_pca_decomposition(eigenvalues = None, eigenvectors = None, n_components = None, title = None , path_save = None):

    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Eigenvalue Spectrum', fontsize=13)
    ax1.scatter(np.arange(eigenvalues.shape[0]), eigenvalues)
    ax1.vlines(n_components, ymin=0, ymax=np.max(eigenvalues), color='k', linestyle='--')
    ax1.legend(['EV= ' + f'{round(np.sum(eigenvalues[:n_components] / np.sum(eigenvalues)),2)}'])
    ax1.set_xlabel('Order', fontsize = 12)
    ax1.set_ylabel('Eigenvalue')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('Eigenvectors', fontsize=13)
    ax2.plot(np.arange(eigenvalues.shape[0]), eigenvectors[:,:n_components] )
    ax2.set_xlabel('Dimension', fontsize = 12)
    ax2.set_ylabel('Projection')

    fig.set_size_inches(9, 5)
    fig.suptitle('PCA analysis: ' + title)
    fig.savefig(path_save)
    return

def plot_pca_projection(projection = None, title = None, path_save = None):

    cmap = cm.jet
    n_components = projection.shape[0]
    color = np.linspace(0, 20, projection.shape[1])

    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(int(n_components/4), 2)

    for i in range(int(n_components/4)):
        ax1 = fig.add_subplot(gs[i, 0],projection='3d')
        ax1.scatter(projection[0,:], projection[i*4+1,:], projection[i*4+2,:], c=color, cmap=cmap)
        ax1.set_xlabel('PC1', fontsize = 12)
        ax1.set_ylabel('PC'+f'{i*4+2}')
        ax1.set_zlabel('PC'+f'{i*4+3}')

        ax2 = fig.add_subplot(gs[i, 1],projection='3d')
        ax2.scatter(projection[0,:], projection[i*4+3,:], projection[i*4+4,:], c=color, cmap=cmap)
        ax2.set_xlabel('PC1', fontsize = 12)
        ax2.set_ylabel('PC'+f'{i*4+4}')
        ax2.set_zlabel('PC'+f'{i*4+5}')

    fig.suptitle('PC projection: ' + title, fontsize = 15)
    fig.set_size_inches(10, 9)
    fig.savefig(path_save)

    return

def plot_pca_behavioral_representation(components_list = None, color = None, title = None, path_save = None):

    cmap = cm.jet

    max_projection = np.zeros((components_list[0].shape[0],1))
    min_projection = np.ones((components_list[0].shape[0],1)) * 100000
    for i in range(len(components_list)):
        components_max = np.max(components_list[i],axis = 2)
        components_min = np.min(components_list[i],axis = 2)
        for j in range(components_list[i].shape[0]):
            if components_max[j] > max_projection[j]:
                max_projection[j] = components_max[j]
            if components_min[j] < min_projection[j]:
                min_projection[j] = components_min[j]

    fig1 = plt.figure()
    axes = fig1.add_subplot(3, 2, 1, projection='3d')
    # axes = fig.add_subplot(111, projection='3d')
    axes.scatter(components_list[0][0, :, :],components_list[0][1, :, :], components_list[0][2, :, :], c=color[0], cmap=cmap)
    axes.set_xlabel('PC1')
    axes.set_ylabel('PC2')
    axes.set_zlabel('PC3')
    axes.set_xlim([min_projection[0], max_projection[0]])
    axes.set_ylim([min_projection[1], max_projection[1]])
    axes.set_zlim([min_projection[2], max_projection[2]])
    axes.set_title('Resting', fontsize = 12)

    axes = fig1.add_subplot(3, 2, 2, projection='3d')
    # axes = fig.add_subplot(111, projection='3d')
    axes.scatter(components_list[1][0, :, :],components_list[1][1, :, :], components_list[1][2, :, :], c=color[1], cmap=cmap)
    axes.set_xlabel('PC1')
    axes.set_ylabel('PC2')
    axes.set_zlabel('PC3')
    axes.set_xlim([min_projection[0], max_projection[0]])
    axes.set_ylim([min_projection[1], max_projection[1]])
    axes.set_zlim([min_projection[2], max_projection[2]])
    axes.set_title('Not Exploring', fontsize = 12)

    axes = fig1.add_subplot(3, 2, 3, projection='3d')
    axes.scatter(components_list[2][0, :, :],components_list[2][1, :, :], components_list[2][2, :, :], c=color[2], cmap=cmap)
    axes.set_xlabel('PC1')
    axes.set_ylabel('PC2')
    axes.set_zlabel('PC3')
    axes.set_xlim([min_projection[0], max_projection[0]])
    axes.set_ylim([min_projection[1], max_projection[1]])
    axes.set_zlim([min_projection[2], max_projection[2]])
    axes.set_title('LR', fontsize = 12)

    axes = fig1.add_subplot(3, 2, 4, projection='3d')
    axes.scatter(components_list[3][0, :, :], components_list[3][1, :, :], components_list[3][2, :, :], c=color[3], cmap=cmap)
    axes.set_xlabel('PC1')
    axes.set_ylabel('PC2')
    axes.set_zlabel('PC3')
    axes.set_xlim([min_projection[0], max_projection[0]])
    axes.set_ylim([min_projection[1], max_projection[1]])
    axes.set_zlim([min_projection[2], max_projection[2]])
    axes.set_title('LL', fontsize = 12)

    axes = fig1.add_subplot(3, 2, 5, projection='3d')
    axes.scatter(components_list[4][0, :, :], components_list[4][1, :, :], components_list[4][2, :, :], c=color[4], cmap=cmap)
    axes.set_xlabel('PC1')
    axes.set_ylabel('PC2')
    axes.set_zlabel('PC3')
    axes.set_xlim([min_projection[0], max_projection[0]])
    axes.set_ylim([min_projection[1], max_projection[1]])
    axes.set_zlim([min_projection[2], max_projection[2]])
    axes.set_title('UR', fontsize = 12)

    axes = fig1.add_subplot(3, 2, 6, projection='3d')
    axes.scatter(components_list[5][0, :, :], components_list[5][1, :, :], components_list[5][2, :, :], c=color[5], cmap=cmap)
    axes.set_xlabel('PC1')
    axes.set_ylabel('PC2')
    axes.set_zlabel('PC3')
    axes.set_xlim([min_projection[0], max_projection[0]])
    axes.set_ylim([min_projection[1], max_projection[1]])
    axes.set_zlim([min_projection[2], max_projection[2]])
    axes.set_title('UL', fontsize = 12)

    fig1.set_size_inches(15, 9)
    fig1.suptitle('Behavioural Projections: ' + title)
    fig1.savefig(path_save)

    return


def plot_pca_spectrum_behaviour(eigenvalues = None, n_components = None , title = None, path_save = None):

    fig = plt.figure(constrained_layout=True)
    gs = plt.GridSpec(len(eigenvalues), 1)

    for i in range(len(eigenvalues)):
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.scatter(np.arange(eigenvalues[i].shape[0]), eigenvalues[i])
        ax1.vlines(n_components, ymin=0, ymax=np.max(eigenvalues[i]), color='k', linestyle='--')
        ax1.legend(['EV= ' + f'{round(np.sum(eigenvalues[i][:n_components] / np.sum(eigenvalues[i])), 2)}'])
        ax1.set_ylabel('Eigenvalue')

    ax1.set_xlabel('Order', fontsize=12)
    fig.suptitle('Eigenvalue Spectrum: ' + title, fontsize = 15)
    fig.set_size_inches(5, 8)
    fig.savefig(path_save)

    return

def plot_pca_eigenvector_distance_behaviour(eigenvectors = None, n_components = None , title = None, path_save = None):

    size = len(eigenvectors)
    fig, axes = plt.subplots(size, size)

    distance_list = []
    for i in range(len(eigenvectors)):
        for j in range(i,len(eigenvectors)):
            distance = np.zeros((n_components,n_components))
            for eig1 in range(n_components):
                for eig2 in range(n_components):
                    vec1 = eigenvectors[i][:,eig1]
                    vec2 = eigenvectors[j][:,eig2]
                    dist = np.linalg.norm(vec1 - vec2)
                    distance[eig1,eig2] = dist
            distance_list.append(distance)

    images = []
    counter = 0
    for i in range(len(eigenvectors)):
        for j in range(i,len(eigenvectors)):
            if counter < len(distance_list):
                images.append(axes[i, j].imshow(distance_list[counter], cmap='viridis'))
                counter = counter+1
                axes[i, j].label_outer()
                axes[i, j].axis('off')

    vmin = min(dist.get_array().min() for dist in images)
    vmax = max(dist.get_array().max() for dist in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    for i in range(len(eigenvectors)):
        for j in range(i):
            axes[i, j].axis('off')

    axes[0,0].set_title('Rest')
    axes[0,1].set_title('NotE')
    axes[0,2].set_title('LR')
    axes[0,3].set_title('LL')
    axes[0,4].set_title('UR')
    axes[0,5].set_title('UL')


    for i in range(len(eigenvectors)):
        for j in range(len(eigenvectors)):
            axes[i, j].axis('off')
    fig.colorbar(images[0], ax=axes[1:size-1, 0], orientation='vertical', fraction=0.9)

    fig.suptitle('Distance between eigenvectors: ' + title, fontsize=15)
    fig.set_size_inches(6, 9)
    fig.savefig(path_save)

    return


def plot_pca_eigenvector_distance_distribution_behaviour(eigenvectors = None, n_components = None , title = None, path_save = None):

    size = len(eigenvectors)
    fig, axes = plt.subplots(2, 2)

    distance_list = []
    for i in range(len(eigenvectors)):
        for j in range(i,len(eigenvectors)):
            distance = np.zeros((n_components,n_components))
            for eig1 in range(n_components):
                for eig2 in range(n_components):
                    vec1 = eigenvectors[i][:,eig1]
                    vec2 = eigenvectors[j][:,eig2]
                    dist = np.linalg.norm(vec1 - vec2)
                    distance[eig1,eig2] = dist
            distance_list.append(distance)

    for i in range(len(eigenvectors)):
        [hist_val, bins_val] = np.histogram(distance_list[i].flatten(),bins = np.arange(0,2,2/15))
        axes[0,0].plot(bins_val[:-1],hist_val/np.sum(hist_val))
    for i in range(len(eigenvectors),len(eigenvectors)+5):
        [hist_val, bins_val] = np.histogram(distance_list[i].flatten(),bins = np.arange(0,2,2/15))
        axes[0,1].plot(bins_val[:-1],hist_val/np.sum(hist_val))

    for i in range(len(eigenvectors)+6,len(distance_list)):
        [hist_val, bins_val] = np.histogram(distance_list[i].flatten(),bins = np.arange(0,2,2/15))
        axes[1,1].plot(bins_val[:-1],hist_val/np.sum(hist_val))

    for i in range(2):
        for j in range(2):
            axes[i,j].set_ylim([0,1])
            axes[i,j].set_ylabel('Count Normalized')
            axes[i,j].set_ylabel('Distance')

    axes[0,0].set_title('Distance with resting')
    axes[0,1].set_title('Distance with not exploting')
    axes[1,1].set_title('Distance between objects')
    axes[1, 0].axis('off')

    fig.suptitle('Distance between eigenvectors distributions: ' + title, fontsize=15)
    fig.set_size_inches(6, 9)
    fig.savefig(path_save)

    return

def plot_eigenvalues_spectrum_learning(eigenvalues = None, n_components = None , title = None, path_save = None):

    fig = plt.figure()
    for i in range(len(eigenvalues)):
        axes = fig.add_subplot(2, 5 ,i+1)
        axes.scatter(np.arange(len(eigenvalues[i])),eigenvalues[i])
        axes.vlines(n_components, ymin=0, ymax=np.max(eigenvalues[i]), color='k', linestyle='--')
        axes.legend(['EV= ' + f'{round(np.sum(eigenvalues[i][:n_components] / np.sum(eigenvalues[i])), 2)}'])
        axes.set_ylabel('Eigenvalue')
        axes.set_xlabel('Order')
        if i+1 < 6:
            axes.set_title('Day '+ f'{i+1}')
        else:
            axes.set_title('Resting')
    fig.set_size_inches(15, 6)
    fig.suptitle('Eigenvalue Spectrum in days: ' + title, fontsize=15)
    fig.savefig(path_save)

    return


def plot_pca_eigenvector_distance_learning(eigenvectors = None, n_components = None , title = None, path_save = None):

    size = len(eigenvectors)
    fig, axes = plt.subplots(size, size)

    distance_list = []
    for i in range(len(eigenvectors)):
        for j in range(i,len(eigenvectors)):
            distance = np.zeros((n_components,n_components))
            for eig1 in range(n_components):
                for eig2 in range(n_components):
                    vec1 = eigenvectors[i][:,eig1]
                    vec2 = eigenvectors[j][:,eig2]
                    dist = np.linalg.norm(vec1 - vec2)
                    distance[eig1,eig2] = dist
            distance_list.append(distance)

    images = []
    counter = 0
    for i in range(len(eigenvectors)):
        for j in range(i,len(eigenvectors)):
            if counter < len(distance_list):
                images.append(axes[i, j].imshow(distance_list[counter], cmap='viridis'))
                counter = counter+1
                axes[i, j].label_outer()
                axes[i, j].axis('off')

    vmin = min(dist.get_array().min() for dist in images)
    vmax = max(dist.get_array().max() for dist in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    for i in range(len(eigenvectors)):
        for j in range(i):
            axes[i, j].axis('off')

    axes[0,0].set_title('Day1')
    axes[0,1].set_title('Day2')
    axes[0,2].set_title('Day3')
    axes[0,3].set_title('Day4')
    axes[0,4].set_title('Test')
    axes[0,5].set_title('Rest1')
    axes[0,6].set_title('Rest2')
    axes[0,7].set_title('Rest3')
    axes[0,8].set_title('Rest4')
    axes[0,9].set_title('RestTest')

    for i in range(len(eigenvectors)):
        for j in range(len(eigenvectors)):
            axes[i, j].axis('off')
    fig.colorbar(images[0], ax=axes[1:size-1, 0], orientation='vertical', fraction=0.9)

    fig.suptitle('Distance between eigenvectors: ' + title, fontsize=15)
    fig.set_size_inches(8, 10)
    fig.savefig(path_save)

    return

def plot_pca_EV_learning(eigenvalues1= None, eigenvalues2 = None, n_components = None, title = None, path_save = None):
    '''
    Plots explained variance (EV) for evolving trials, using different number of components.

    :param eigenvalues1:
    :param eigenvalues2:
    :param n_components:
    :param title:
    :param path_save:
    :return:
    '''
    EV_trials = np.zeros((len(eigenvalues1), 1))
    EV_rest = np.zeros((len(eigenvalues2), 1))

    fig, axes = plt.subplots(1)
    for n in range(n_components-5,n_components+5):
        for i in range(len(eigenvalues1)):
            EV_trials[i] = np.sum(eigenvalues1[i][:n] / np.sum(eigenvalues1[i]))
            EV_rest[i] = np.sum(eigenvalues2[i][:n] / np.sum(eigenvalues2[i]))
        axes.plot(np.arange(1, EV_trials.shape[0] + 1), EV_trials, c='b')
        axes.plot(np.arange(1, EV_rest.shape[0] + 1), EV_rest, c='r')

    axes.set_ylim([0, 1])
    axes.set_xlabel('Trials', fontsize = 12)
    axes.set_ylabel('EV', fontsize = 12)
    fig.suptitle('Explained Variance: '+ title)
    fig.savefig(path_save)

    return


def plot_MDS_multisessions(neural_activity_msd = [], sessions = [], task = [], path_save=None):
    '''
    Scatter plot MDS for all sessions in the sessions list
    :param neural_activity_msd:
    :param sessions:
    :param task:
    :param path_save:
    :return: None
    '''

    cmap = cm.jet
    fig1 = plt.figure()
    for session in range(len(sessions)):
        axes = fig1.add_subplot(1, len(sessions), session + 1, projection='3d')
        color = np.linspace(0, 20, neural_activity_msd[session].shape[0])
        axes.scatter(neural_activity_msd[session][:, 0], neural_activity_msd[session][:, 1],
                     neural_activity_msd[session][:, 2], c=color, cmap=cmap)
        axes.set_xlabel('MDS1')
        axes.set_ylabel('MDS2')
        axes.set_zlabel('MDS3')
        axes.set_title(task[session], fontsize=12)
        #for angle in range(0, 360):
        #    axes.view_init(30, angle)
    fig1.suptitle('MDS', fontsize=15)
    fig1.set_size_inches(15, 9)
    #plt.draw()
    #plt.pause(.001)
    fig1.savefig(path_save)

    return


def plot_MDS_multisession_behaviour(neural_activity_msd = None, resample_timeline= None, resample_beh=None, task = None,  save_path = None):

    color = ['k', 'b', 'r', 'g', 'm', 'c']
    ## separate different behavioural parts of the experiment in the mds
    for session in range(len(task)):
        # define a variable with only training data (remove testing)
        training_data = neural_activity_msd[session][:int(resample_timeline[session][40]),
                        :]  ## neural activity in training
        training_data_beh = resample_beh[session][
                            :int(resample_timeline[session][40])]  ## bahavioural vector in training
        testing_data = neural_activity_msd[session][int(resample_timeline[session][40]):,
                       :]  ## neural activity mds in testing
        testing_data_beh = resample_beh[session][
                           int(resample_timeline[session][40]):]  ## neural activity mds in testing

        mds_training = []  ## list containing neural activity mds for different bahavioural parts
        mds_testing = []  ## list containing different behaviours in the testing trial

        # color_training=[]           ## arrange of colors. Will be necesary to have a common color criteria when plotting
        # color_testing = []
        for i in range(6):  ## 6 different behaviours defined here
            mds_training.append(training_data[np.where(training_data_beh == i), :])
            mds_testing.append(testing_data[np.where(testing_data_beh == i), :])
            # color_training.append(color[np.where(training_data_beh ==i)])
            # color_testing.append(color[np.where(testing_data_beh==i)])

        fig1 = plt.figure()
        axes = fig1.add_subplot(1, 2, 1, projection='3d')
        for i in range(0, 2):
            axes.scatter(mds_training[i][0, :, 0], mds_training[i][0, :, 1], mds_training[i][0, :, 2], c=color[i])
        axes.set_xlabel('MDS1')
        axes.set_ylabel('MDS2')
        axes.set_zlabel('MDS3')
        axes.legend(['Resting', 'Non Exploring'])
        axes = fig1.add_subplot(1, 2, 2, projection='3d')
        for i in range(2, 6):
            axes.scatter(mds_training[i][0, :, 0], mds_training[i][0, :, 1], mds_training[i][0, :, 2], c=color[i])
        axes.set_xlabel('MDS1')
        axes.set_ylabel('MDS2')
        axes.set_zlabel('MDS3')
        axes.legend(['LR','LL', 'UR','UL'])
        axes.set_xlim([-2, 2])
        axes.set_ylim([-2, 2])
        axes.set_zlim([-2, 2])
        fig1.suptitle('Behavioural Conditions: ' + task[session])
        fig1.savefig(save_path + task[session] + '.png')

    return