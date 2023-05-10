"""
fit HDDM model, with history terms, to data from IBL mice
Anne Urai, 2019, CSHL

"""

# ============================================ #
# GETTING STARTED
# ============================================ #

import matplotlib as mpl
mpl.use('Agg')  # to still plot even when no display is defined
from optparse import OptionParser
import pandas as pd
import os

# import HDDM functions, defined in a separate file
import hddmnn_funcs

# more handy imports
import hddm
import seaborn as sns
sns.set()

# define path to save the data and figures
datapath = 'data'
figpath = 'figures'
modelpath = '/home/uraiae/data1/HDDMnn' # on ALICE

# ============================================ #
# READ INPUT ARGUMENTS
# ============================================ #

usage = "HDDM_run.py [options]"
parser = OptionParser(usage)
parser.add_option("-m", "--model",
                  default=['ddm_nohist', 'ddm_nohist_stimcat', 
                           'ddm_prevresp_dcz', 'ddm_prevresp_dc', 'ddm_prevresp_z'],
                  type="string",
                  help="name of the model to run")
parser.add_option("-d", "--dataset",
                  default=["trainingchoiceworld_clean"],
                  type="string",
                  help="which data to fit on")
opts, args = parser.parse_args()
if isinstance(opts.model, str):
    opts.model = [opts.model]
print(opts)

# ============================================ #

for d in opts.dataset:

    # GET DATA
    data = pd.read_csv(os.path.join(datapath, 'ibl_%s.csv' % d))
    data['stimulus'] = data['signed_contrast']
    data = data.dropna(subset=['rt', 'response', 'previous_choice', 'previous_outcome', 'stimulus'])

    # FIT THE ACTUAL MODEL
    for m in opts.model:
        hddmnn_funcs.run_model(data, m, os.path.join(modelpath, d, m), n_samples=1000, force=True)

        # also sample posterior predictives (will only do if doesn't already exists)
        # hddm_funcs.posterior_predictive(os.path.join(datapath, d, m), n_samples=100)