import pandas as pd
import numpy as np
import scipy as sp
import sys, os, glob, time

import matplotlib
matplotlib.use('Agg') # to still plot even when no display is defined
import matplotlib.pyplot as plt

# more handy imports
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
        os.makedirs(mypath)
        print('creating directory %s' % mypath)

    print("begin sampling") # this is the core of the fitting
    m.sample(n_samples, burn = np.max([n_samples/10, 100]),
             dbname='traces.db', db='pickle')

    print('saving model itself')
    m.save(os.path.join(mypath, 'model'))
    
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

    return m

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


def plot_model(m, savepath):

    # MAKE SOME PLOTS
    # for testing parameter trade-offs
    hddm.plotting.plot_posterior_pair(m, samples=50,
                                      save=True, save_path=savepath)

    # to see the overall model behave
    hddm.plotting.plot_posterior_predictive(model = m,
                                            save=True, save_path=savepath,
                                            columns = 4, # groupby = ['subj_idx'],
                                            value_range = np.arange(0, 3, 0.1),
                                            plot_func = hddm.plotting._plot_func_model,
                                            parameter_recovery_mode = True,
                                            **{'add_legend': False,
                                            'alpha': 0.01,
                                            'ylim': 6.0,
                                            'bin_size': 0.025,
                                            'add_posterior_mean_model': True,
                                            'add_posterior_mean_rts': True,
                                            'add_posterior_uncertainty_model': True,
                                            'add_posterior_uncertainty_rts': False,
                                            'samples': 200,
                                            'legend_fontsize': 7,
                                            'legend_loc': 'upper left',
                                            'linewidth_histogram': 1.0,
                                            'subplots_adjust': {'top': 0.9, 'hspace': 0.35, 'wspace': 0.3}})
    plt.savefig(os.path.join(savepath, 'posterior_predictive_model_plot.png'))

    hddm.plotting.plot_posterior_predictive(model = m,
                                            save=True, save_path=savepath,
                                            columns = 4, # groupby = ['subj_idx'],
                                            value_range = np.arange(-4, 4, 0.01),
                                            plot_func = hddm.plotting._plot_func_posterior_pdf_node_nn,
                                            parameter_recovery_mode = True,
                                            **{'alpha': 0.01,
                                            'ylim': 3,
                                            'bin_size': 0.05,
                                            'add_posterior_mean_rts': True,
                                            'add_posterior_uncertainty_rts': True,
                                            'samples': 200,
                                            'legend_fontsize': 7,
                                            'subplots_adjust': {'top': 0.9, 'hspace': 0.3, 'wspace': 0.3}})
    plt.savefig(os.path.join(savepath, 'posterior_predictive_plot.png'))
