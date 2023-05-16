"""
fit HDDM model, with history terms, to data from IBL mice
Anne Urai, 2019, CSHL

"""

# ============================================ #
# GETTING STARTED
# ============================================ #

# import matplotlib as mpl
# mpl.use('Agg')  # to still plot even when no display is defined
from optparse import OptionParser
import pandas as pd
import os

# check if we have a GPU available
print ('Checking if there is a GPU...')
import torch
print(torch.cuda.is_available())

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

models = ['ddm_nohist', 
          'ddm_nohist_stimcat', 
          'ddm_prevresp_dcz', 
          'ddm_prevresp_dc', 
          'ddm_prevresp_z'
          ]

if isinstance(opts.model, str):
    opts.model = [opts.model]
# select only this model
m = models[opts.model]

# ============================================ #
# NOW RUN THE FITS ON THIS DATASET

data = pd.read_csv(os.path.join(datapath, '%s.csv' % dataset))
data = data.dropna(subset=['rt', 'response', 'prevresp', 'prevfb', 'signed_contrast'])

# FIT THE ACTUAL MODEL
utils_hddmnn.run_model(data, m, os.path.join(modelpath, dataset, m), n_samples=2000)
