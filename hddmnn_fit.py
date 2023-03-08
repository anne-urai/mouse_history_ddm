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
import hddm_funcs

# more handy imports
import hddm
import seaborn as sns
sns.set()

# define path to save the data and figures
datapath = 'data'
modelpath = '/home/uraiae/data1/HDDMnn' # on ALICE
figpath = 'figures'

# ============================================ #
# READ INPUT ARGUMENTS
# ============================================ #

usage = "HDDM_run.py [options]"
parser = OptionParser(usage)
parser.add_option("-m", "--model",
                  default=['nohist', 'prevchoice_dcz', 'prevchoice_dc', 'prevchoice_z',
                           'prevchoiceoutcome_dcz', 'prevchoiceoutcome_dc',
                           'prevchoiceoutcome_z'],
                  type="string",
                  help="name of the model to run")
parser.add_option("-d", "--dataset",
                  default=["trainingchoiceworld"],
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
    # MAKE A PLOT OF THE RT DISTRIBUTIONS PER ANIMAL
    g = sns.FacetGrid(data, col='subj_idx', col_wrap=8)
    g.map(sns.distplot, "rt", kde=False, rug=True)
    g.savefig(os.path.join(figpath, 'rtdist_%s.png' % d))
    print(data.rt.describe())

    # FIT THE ACTUAL MODEL
    for m in opts.model:

        # skip fitting blocked models in trainingChoiceWorld
        if 'blocks' in m and not 'biased' in d:
            continue

        # md = run_model_gsq(data, m, os.path.join(datapath, d))
        hddm_funcs.run_model(data, m, os.path.join(modelpath, d, m), n_samples=10000, force=True)

        # also sample posterior predictives (will only do if doesn't already exists)
        # hddm_funcs.posterior_predictive(os.path.join(datapath, d, m), n_samples=100)