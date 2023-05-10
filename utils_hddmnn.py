import pandas as pd
import numpy as np
import scipy as sp
import sys, os, glob, time
import pickle
import math

import matplotlib

matplotlib.use('Agg')  # to still plot even when no display is defined
import matplotlib.pyplot as plt
import seaborn as sns

# more handy imports
from IPython import embed as shell
import hddm, kabuki

# ============================================ #
# define some functions
# ============================================ #

def run_model(data, modelname, mypath, n_samples=1000, trace_id=0):

    from hddmnn_modelspec import make_model # specifically for HDDMnn models

    print('HDDM version: ', hddm.__version__)
    print('Kabuki version: ', kabuki.__version__)

    # get the model
    m = make_model(data, modelname)
    time.sleep(trace_id) # to avoid different jobs trying to make the same folder

    # make a new folder if it doesn't exist yet
    if not os.path.exists(mypath):
        try:
            os.makedirs(mypath)
            print('creating directory %s' % mypath)
        except:
            pass

    print("begin sampling") # this is the core of the fitting
    m.sample(n_samples, burn = np.max([n_samples/10, 100]))

    print("save model comparison indices")
    df = dict()
    df['dic'] = [m.dic]
    df['aic'] = [aic(m)]
    df['bic'] = [bic(m)]
    df2 = pd.DataFrame(df)
    df2.to_csv(os.path.join(mypath, 'model_comparison.csv'))

    # save useful output
    print("saving summary stats")
    results = m.gen_stats().reset_index()  # point estimate for each parameter and subject
    results.to_csv(os.path.join(mypath, 'results_combined.csv'))

    print("saving traces")
    # get the names for all nodes that are available here
    group_traces = m.get_group_traces()
    group_traces.to_csv(os.path.join(mypath, 'group_traces.csv'))

# ============================================ #
# MODEL COMPARISON INDICES
# ============================================ #

def aic(self):
    
    k = len(self.get_stochastics())
    try:
        logp = sum([x.logp for x in self.get_observeds()['node']])
        aic = 2 * k - 2 * logp
    except:
        aic = np.nan
    return aic


def bic(self):

    k = len(self.get_stochastics())
    n = len(self.data)
    try:
        logp = sum([x.logp for x in self.get_observeds()['node']])
        bic = -2 * logp + k * np.log(n)
    except:
        bic = np.nan    
    return bic

# ============================================ #
# ANNOTATE THE CORRELATION PLOT
# ============================================ #

def corrfunc(x, y, **kws):

    # compute spearmans correlation across age groups
    r, pval = sp.stats.spearmanr(x, y, nan_policy='omit')
    print('%s, %s, %.2f, %.3f'%(x.name, y.name, r, pval))

    if 'ax' in kws.keys():
        ax = kws['ax']
    else:
        ax = plt.gca()

    # if this correlates, draw a regression line across groups
    if pval < 0.05/4:
        sns.regplot(x, y, truncate=True, color='gray',
        scatter=False, ci=None, robust=True, ax=ax)

    # now plot the datapoint, with age groups
    if 'yerr' in kws.keys():
        ax.errorbar(x, y, yerr=kws['yerr'].values, fmt='none', zorder=0, ecolor='silver', elinewidth=0.5)
        kws.pop('yerr', None)
    sns.scatterplot(x=x, y=y, legend=False, **kws)

    # annotate with the correlation coefficient + n-2 degrees of freedom
    txt = r"$\rho$({}) = {:.3f}".format(len(x)-2, r) + "\n" + "p = {:.4f}".format(pval)
    if pval < 0.0001:
        txt = r"$\rho$({}) = {:.3f}".format(len(x)-2, r) + "\n" + "p < 0.0001"
    ax.annotate(txt, xy=(.7, .1), xycoords='axes fraction', fontsize='small')


def results_long2wide(md):

    # recode to something more useful
    # 0. replace x_subj(yy).ZZZZ with x(yy)_subj.ZZZZ
    md["colname_tmp"] = md["Unnamed: 0"].str.replace('.+\_subj\(.+\)\..+', '.+\(.+\)\_subj\..+', regex=True)

    # 1. separate the subject from the parameter
    new = md["Unnamed: 0"].str.split("_subj.", n=1, expand=True)
    md["parameter"] = new[0]
    md["subj_idx"] = new[1]
    new = md["subj_idx"].str.split("\)\.", n=1, expand=True)

    # separate out subject idx and parameter value
    for index, row in new.iterrows():
        if row[1] == None:
            row[1] = row[0]
            row[0] = None

    md["parameter_condition"] = new[0]
    md["subj_idx"] = new[1]

    # pivot to put parameters as column names and subjects as row names
    md = md.drop('Unnamed: 0', axis=1)
    md_wide = md.pivot_table(index=['subj_idx'], values='mean',
                             columns=['parameter', 'parameter_condition']).reset_index()
    return md_wide
