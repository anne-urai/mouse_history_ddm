import pandas as pd
import numpy as np
import scipy as sp
import sys, os, glob, time

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
