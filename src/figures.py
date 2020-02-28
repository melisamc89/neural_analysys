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
cmap = cm.jet



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
    gs = fig.add_gridspec(3, 12)
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
    gs = fig.add_gridspec(2, 3)
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
