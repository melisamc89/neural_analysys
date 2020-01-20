import os

#%% ENVIRONMENT VARIABLES
os.environ['PROJECT_DIR_LOCAL'] = '/home/sebastian/Documents/Melisa/neural_analysis/'
os.environ['PROJECT_DIR_SERVER'] = '/home/mmaidana/src/neural_analysis/'

os.environ['DATA_DIR_LOCAL'] = '/mnt/Data01/data/neural_analysis/'
os.environ['DATA_DIR_SERVER'] ='/scratch/melisa/neural_analsysis/'

os.environ['CAIMAN_ENV_SERVER'] = '/memdym/melisa/caiman/bin/python'

os.environ['LOCAL_USER'] = 'sebastian'
os.environ['SERVER_USER'] = 'mmaidana'
os.environ['SERVER_HOSTNAME'] = 'cn43'
os.environ['ANALYST'] = 'Meli'

#%% PROCESSING
os.environ['LOCAL'] = str((os.getlogin() == os.environ['LOCAL_USER']))
os.environ['SERVER'] = str(not(eval(os.environ['LOCAL'])))

os.environ['PROJECT_DIR'] = os.environ['PROJECT_DIR_LOCAL'] if eval(os.environ['LOCAL']) else os.environ['PROJECT_DIR_SERVER']
os.environ['DATA_DIR'] = os.environ['DATA_DIR_LOCAL'] if eval(os.environ['LOCAL']) else os.environ['DATA_DIR_SERVER']
