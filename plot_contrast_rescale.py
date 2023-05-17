"""
check if drift rate scales linearly with signed_contrast
if not, apply a transform to approximate linearity
Anne Urai, Leiden University, 2023

"""

#%% ============================================ #
# GETTING STARTED
# ============================================ #

import pandas as pd
import scipy as sp
import os
from scipy.optimize import curve_fit
import numpy as np

import matplotlib
matplotlib.use('Agg') # to still plot even when no display is defined
import matplotlib.pyplot as plt
import seaborn as sns

# more handy imports
from utils_plot import results_long2wide_hddmnn, seaborn_style
seaborn_style()

# find path depending on location and dataset
usr = os.environ['USER']
if 'uraiae' in usr: # ALICE
    modelpath = '/home/uraiae/data1/HDDMnn/ibl_trainingchoiceworld_clean' # on ALICE
figpath = 'figures'
datapath = 'data'

#%% ============================================ #
# sanity check, drift rate by contrast
# ============================================ #

md = results_long2wide_hddmnn(pd.read_csv(os.path.join(modelpath, 'ddm_nohist_stimcat', 'results_combined.csv')))
md.columns = md.columns.map(''.join)
md.rename(columns={'v-100.0':-100., 'v-50.0':-50., 'v-25.0':-25., 'v-12.5':-12.5, 'v-6.25':-6.25, 'v0.0':0.,
                                            'v6.25':6.25, 'v12.5':12.5, 'v25.0':25., 'v50.0':50., 'v100.0':100.},
                                            inplace=True)

plt.close('all')
fig, ax = plt.subplots(ncols=2, nrows=1)
order = [-100., -50., -25., -12.5, -6.25, 0., 6.25, 12.5, 25., 50., 100]
md_long = pd.melt(md, id_vars=['subj_idx'], value_vars=order)
#sns.stripplot(data=md_long, x='variable', y='value', ax=ax, color='grey', zorder=0)
sns.lineplot(data=md_long, x='variable', y='value', ax=ax[0], color='darkblue',
            err_style='bars', marker='o')
ax[0].set(xlabel='Signed contrast (%)', ylabel='Drift rate (v)')

# the drift rate does *not* scale linearly, it levels off at higher contrast levels. can we correct for those?
driftrates = md_long.groupby('variable')['value'].mean().reset_index()
xdata = driftrates['variable']
ydata = driftrates['value']

def sigmoid(x, a, b):
    return a * np.tanh( b * x )

p0 = [3, 0.4]
popt, pcov = curve_fit(sigmoid, xdata, ydata)
# popt: array([2.13731484, 0.05322221])
xModel = np.linspace(min(xdata), max(xdata))
yModel = sigmoid(xModel, *popt)
ax[0].plot(xModel, yModel, 'darkred')

# now rescale the drift rate values with tanh
contrast_rescaled = sigmoid(xdata.values, popt[0], popt[1])
driftrates['new_contrast'] = contrast_rescaled
md_long['new_contrast'] = md_long['variable'].apply(lambda x: sigmoid(x, popt[0], popt[1]))

sns.lineplot(data=md_long, x='new_contrast', y='value', ax=ax[1], 
             units='subj_idx', color='lightgrey', estimator=None,
             marker='.')
sns.lineplot(data=md_long, x='new_contrast', y='value', ax=ax[1], color='darkblue',
            err_style='bars', marker='o')
ax[1].set(xlabel='Rescaled contrast', ylabel='Drift rate (v)',
          title='Rescaling: \nnew_contrast = %.3f * tanh(%.3fx)'%(popt[0], popt[1]))

sns.despine(trim=True)
fig.tight_layout()
fig.savefig(os.path.join(figpath, 'driftrate_nohist.png'))