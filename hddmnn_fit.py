"""
fit HDDM model, with history terms, to data from IBL mice
Anne Urai, 2019, CSHL

"""

# ============================================ #
# GETTING STARTED
# ============================================ #

from optparse import OptionParser
import pandas as pd
import os
import numpy as np

# # check if we have a GPU available
# print ('Checking if there is a GPU...')
# import torch
# print(torch.cuda.is_available())

# import HDDM functions, defined in a separate file
import utils_hddmnn
import hddm

# define path to save the data and figures
datapath = 'data'
figpath = 'figures'

# ============================================ #
# READ INPUT ARGUMENTS
# ============================================ #

# read inputs
parser = OptionParser("HDDM_run.py [options]")
parser.add_option("-m", "--model",
                  default=0,
                  type="int",
                  help="model number")
parser.add_option("-d", "--dataset",
                  default=0,
                  type="int",
                  help="dataset nr")
opts, args = parser.parse_args()

# find path depending on location and dataset
usr = os.environ['USER']
if 'uraiae' in usr: # ALICE
    modelpath = '/home/uraiae/data1/HDDMnn' # on ALICE

# HDDMnn folder in /home/uraiae/data1
datasets = ['ibl_trainingchoiceworld_clean']

# select only this dataset
if isinstance(opts.dataset, str):
    opts.dataset = [opts.dataset]
dataset = datasets[opts.dataset]

# ============================================ #
# READ INPUT ARGUMENTS; model
# ============================================ #

models = ['ddm_nohist_stimcat', 
          'ddm_nohist', # 1
          'ddm_prevresp_z', 
          'ddm_prevresp_v', 
          'ddm_prevresp_zv',
          'angle_nohist', # 5
          'angle_prevresp_z', 
          'angle_prevresp_v', 
          'angle_prevresp_zv',
          'angle_stimcat_prevresp_zv',
          'weibull_nohist', # 10
          'weibull_prevresp_z', 
          'weibull_prevresp_v', 
          'weibull_prevresp_zv',
          ]

if isinstance(opts.model, str):
    opts.model = [opts.model]
# select only this model
m = models[opts.model]

# ============================================ #
# NOW RUN THE FITS ON THIS DATASET

data = pd.read_csv(os.path.join(datapath, '%s.csv' % dataset))
data = data.dropna(subset=['rt', 'response', 'prevresp', 'prevfb', 'stimulus'])

# FIT THE ACTUAL MODEL
m_fitted = utils_hddmnn.run_model(data, m, os.path.join(modelpath, dataset, m), n_samples=5000)

# PLOT SEVERAL THINGS AFTERWARDS
utils_hddmnn.plot_model(m_fitted, os.path.join(modelpath, dataset, m))
