import pandas as pd
import numpy as np
import scipy as sp
import sys, os, glob
import pickle
import math

import matplotlib

matplotlib.use('Agg')  # to still plot even when no display is defined
import matplotlib.pyplot as plt
import seaborn as sns

# more handy imports
from IPython import embed as shell
import hddm, kabuki
from hddm_modelspec_gsq import make_model

# ============================================ #
# define some functions
# ============================================ #

# get around a problem with saving regression outputs in Python 3
# https://groups.google.com/forum/#!topic/hddm-users/lKNAFbFk-F8
def savePatch(self, fname):
    with open(fname, 'wb') as f:
        pickle.dump(self, f)


def run_model(data, modelname, mypath, n_samples=10000, trace_id=0, force=False):

    # skip if already exists
    if os.path.exists(os.path.join(mypath, 'model_comparison.csv')) and not force:
        print('skipping, already exists')

    else:

        # get the model
        m = make_model(data, modelname)

        # make a new folder if it doesn't exist yet
        if not os.path.exists(mypath):
            print('creating directory %s' % mypath)
            os.makedirs(mypath)

        print("finding starting values")
        m.find_starting_values()  # this should help the sampling

        print("begin sampling")
        m.sample(n_samples, burn=n_samples / 10, thin=2, db='pickle',
                 dbname=os.path.join(mypath, 'modelfit_md%d.db' % trace_id))

        m.save(os.path.join(mypath, 'modelfit-md%d.model' % trace_id))  # save the model to disk
        # hddm.HDDMRegressor.savePatch = savePatch
        # m.savePatch(os.path.join(mypath, 'modelfit_md%d.model' % trace_id))

        print("saving summary stats")
        results = m.gen_stats()  # point estimate for each parameter and subject
        results.to_csv(os.path.join(mypath, 'results_combined.csv'))

        print("saving traces")
        # get the names for all nodes that are available here
        group_traces = m.get_group_traces()
        group_traces.to_csv(os.path.join(mypath, 'group_traces.csv'))

        all_traces = m.get_traces()
        all_traces.to_csv(os.path.join(mypath, 'all_traces.csv'))

        print("save model comparison indices")
        # save model comparison indices
        df = dict()
        df['dic'] = [m.dic]
        df['aic'] = [aic(m)]
        df['bic'] = [bic(m)]
        df2 = pd.DataFrame(df)
        df2.to_csv(os.path.join(mypath, 'model_comparison.csv'))


def posterior_predictive(mypath, n_samples=10):

    # # STANDARD POSTERIOR PREDICTIVES DON'T WORK FOR REGRESSION MODELS!
    # sample posterior predictive
    if not os.path.exists(os.path.join(mypath, 'posterior_predictive.csv')):
        # load the model
        m = hddm.load(os.path.join(mypath, 'modelfit-md0.model'))
        print('sampling posterior predictives...')
        ppc_data = hddm.utils.post_pred_gen(m, samples=n_samples, append_data=True, progress_bar=True)
        ppc_data.to_csv(os.path.join(mypath, 'posterior_predictive.csv'))
    else:
        ppc_data = pd.DataFrame.from_csv(os.path.join(mypath, 'posterior_predictive.csv'))

    # now plot posterior predictives with some custom code
    g = sns.FacetGrid(ppc_data, col="subj_idx", col_wrap=4, xlim=[-5, 5])
    g = g.map(plt.hist, "rt_sampled", color="firebrick", density=True,
              histtype='step', bins=np.linspace(-5, 5, 1000))
    g = g.map(plt.hist, "rt", color=".5", histtype='step', density=True, bins=np.linspace(-5, 5, 1000))
    g.savefig(os.path.join(mypath, 'posterior_predictive.pdf'))

    # then plot posteriors
    print('plotting posteriors')
    hddm.utils.plot_posteriors(m, path=mypath, suffix='_posterior.pdf')


# ============================================ #
# SAME THING BUT FOR QUICK AND DIRTY GSQ FITS
# ============================================ #

def run_model_gsq(data, modelname, mypath):
    subj_params = []
    for subj_idx, subj_data in data.groupby('subj_idx'):
        print(subj_data.subj_idx.unique()[0])

        # GET THE MODEL SPECIFICATION
        m_subj = make_model(subj_data, modelname)

        thismodel = m_subj.optimize('gsquare')
        thismodel.update({'subj_idx': subj_data.subj_idx.unique()[0],
                          'lab_name': subj_data.lab_name.unique()[0],
                          'bic': m_subj.bic,
                          'aic': m_subj.aic})  # keep original subject number

        # already recode for history shifts
        subj_params.append(thismodel)

    # SAVE THIS
    params = pd.DataFrame(subj_params)
    if not os.path.exists(mypath):
        os.makedirs(mypath)
    params.to_csv(os.path.join(mypath, '%s_gsquare.csv' % modelname))


# ============================================ #
# MODEL COMPARISON INDICES
# ============================================ #

def aic(self):
    k = len(self.get_stochastics())
    logp = sum([x.logp for x in self.get_observeds()['node']])
    return 2 * k - 2 * logp


def bic(self):
    k = len(self.get_stochastics())
    n = len(self.data)
    logp = sum([x.logp for x in self.get_observeds()['node']])
    return -2 * logp + k * np.log(n)


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
