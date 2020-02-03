"""
Created on Fri Jan 17 13:23:00 2020

@author: Melisa
"""

import os
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import configuration


source_extracted_files_dir = os.environ['PROJECT_DIR'] + 'data/calcium_traces_concatenation/'
output_dir = os.environ['PROJECT_DIR']  + 'data/calcium_activity/'
cropping_number = [1,2,3,4]
mouse = 56165
decoding_v = 1
motion_correction_v = 1
alignment_v = 2
source_extraction_v = 1
time_sf = 10

sessions = [1,2,4]

for session in sessions:
    calcium_trace = []
    calcium_trace_shape = []
    for cropping_v in cropping_number:
        input_file_name = 'mouse_'+ f'{mouse}'+ '_session_' + f'{session}' + '_trial_1_v' + f'{decoding_v}' '.' + f'{cropping_v}'\
                    + '.' + f'{motion_correction_v}' + '.' + f'{alignment_v}' + '.' + f'{source_extraction_v}' + '.hdf5'
        cnm = load_CNMF(source_extracted_files_dir + input_file_name)
        calcium_trace.append(cnm.estimates.C - cnm.estimates.bl[:, np.newaxis])
        calcium_trace_shape.append(cnm.estimates.C.shape)
    time = np.arange(0,(calcium_trace_shape[0])[1])/time_sf
    n_neurons = 0
    for i in range(len(cropping_number)):
        n_neurons = n_neurons + (calcium_trace_shape[i])[0]
    activity_matrix = np.zeros((n_neurons+1,len(time)))
    activity_matrix[0,:] = time
    init = 1
    finish = (calcium_trace_shape[0])[0]+1
    for i in range(len(cropping_number)-1):
        activity_matrix[init:finish,:] = calcium_trace[i]
        init = init + (calcium_trace_shape[i])[0]
        finish = finish + (calcium_trace_shape[i+1])[0]
    activity_matrix[init:finish,:] = calcium_trace[len(cropping_number)-1]
    output_file_name = 'mouse_'+ f'{mouse}'+ '_session_' + f'{session}' + '_v' + f'{decoding_v}' '.' + f'{cropping_v}'\
                    + '.' + f'{motion_correction_v}' + '.' + f'{alignment_v}' + '.' + f'{source_extraction_v}'
    np.save(output_dir + output_file_name, activity_matrix)