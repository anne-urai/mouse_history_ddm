import pandas as pd
import numpy as np
import scipy as sp
import sys, os, glob
import pickle
import math

import matplotlib
matplotlib.use('Agg') # to still plot even when no display is defined
import matplotlib.pyplot as plt
import seaborn as sns

# more handy imports
from IPython import embed as shell
import hddm, kabuki

# LINK FUNCTION FOR Z

def z_link_func(x):
	# logistic link function for starting point
    return 1 / (1 + np.exp(-(x.values.ravel())))

# ============================================ #
# MODEL SPECIFICATION
# ============================================ #

def make_model(data, mname):
 
    print('making HDDM model')

    # NEW SET WITH LOG-TRANSFORMED CONTRAST VALUES
    if mname == 'nohist':
        v_reg = {'model': 'v ~ 1 + logSignedContrast', 'link_func': lambda x:x}
        m = hddm.HDDMRegressor(data, [v_reg], include=['z', 'sv'], group_only_nodes=['sv'],
            group_only_regressors=False, keep_regressor_trace=False, p_outlier=0.05)

    # 20/80 AND 80/20 BLOCKS
    elif mname == 'blocks_dcz':
        v_reg = {'model': 'v ~ 1 + logSignedContrast + C(probabilityLeft, Treatment(0.5))', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1 + C(probabilityLeft, Treatment(0.5))', 'link_func': z_link_func}
        m = hddm.HDDMRegressor(data, [v_reg, z_reg], include=['z', 'sv'], group_only_nodes=['sv'],
            group_only_regressors=False, keep_regressor_trace=False, p_outlier=0.05)

    elif mname == 'blocks_dc':
        v_reg = {'model': 'v ~ 1 + logSignedContrast + C(probabilityLeft, Treatment(0.5))', 'link_func': lambda x:x}
        # z_reg = {'model': 'z ~ 1 + C(probabilityLeft, Treatment(0.5))', 'link_func': z_link_func}
        m = hddm.HDDMRegressor(data, [v_reg], include=['z', 'sv'], group_only_nodes=['sv'],
            group_only_regressors=False, keep_regressor_trace=False, p_outlier=0.05)

    elif mname == 'blocks_z':
        v_reg = {'model': 'v ~ 1 + logSignedContrast', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1 + C(probabilityLeft, Treatment(0.5))', 'link_func': z_link_func}
        m = hddm.HDDMRegressor(data, [v_reg, z_reg], include=['z', 'sv'], group_only_nodes=['sv'],
            group_only_regressors=False, keep_regressor_trace=False, p_outlier=0.05)

    # PREVIOUS CHOICE, TAKE ONLY THOSE BLOCKS WITH 50/50
    elif mname == 'prevchoice_dcz':

        # take only data in unbiased blocks, drop the rest
        data = data.drop(data[data.probabilityLeft != 0.5].index)
        data = data.dropna(subset=['prevchoice_correct', 'prevchoice_error']) 

        # now use previous outcome (coded as in Busse)
        v_reg = {'model': 'v ~ 1 + logSignedContrast + prevchoice_correct + prevchoice_error', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1 + prevchoice_correct + prevchoice_error', 'link_func': z_link_func}
        m = hddm.HDDMRegressor(data, [v_reg, z_reg], include=['z', 'sv'], group_only_nodes=['sv'],
            group_only_regressors=False, keep_regressor_trace=False, p_outlier=0.05)

    elif mname == 'prevchoice_dc':

        # take only data in unbiased blocks, drop the rest
        data = data.drop(data[data.probabilityLeft != 0.5].index)
        data = data.dropna(subset=['prevchoice_correct', 'prevchoice_error']) 

        # now use previous outcome (coded as in Busse)
        v_reg = {'model': 'v ~ 1 + logSignedContrast + prevchoice_correct + prevchoice_error', 'link_func': lambda x:x}
        # z_reg = {'model': 'z ~ 1 + prevchoice_correct + prevchoice_error', 'link_func': z_link_func}
        m = hddm.HDDMRegressor(data, [v_reg], include=['z', 'sv'], group_only_nodes=['sv'],
            group_only_regressors=False, keep_regressor_trace=False, p_outlier=0.05)

    elif mname == 'prevchoice_z':

        # take only data in unbiased blocks, drop the rest
        data = data.drop(data[data.probabilityLeft != 0.5].index)
        data = data.dropna(subset=['prevchoice_correct', 'prevchoice_error']) 

        # now use previous outcome (coded as in Busse)
        v_reg = {'model': 'v ~ 1 + logSignedContrast', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1 + prevchoice_correct + prevchoice_error', 'link_func': z_link_func}
        m = hddm.HDDMRegressor(data, [v_reg, z_reg], include=['z', 'sv'], group_only_nodes=['sv'],
            group_only_regressors=False, keep_regressor_trace=False, p_outlier=0.05)

    # BOTH BLOCKS AND PREVIOUS CHOICE
    elif mname == 'blocks_prevchoice_dcz':

        # avoid nans after trials without a choice
        data = data.dropna(subset=['prevchoice_correct', 'prevchoice_error']) 

        # now use previous outcome (coded as in Busse)
        v_reg = {'model': 'v ~ 1 + logSignedContrast + C(probabilityLeft, Treatment(0.5)) + prevchoice_correct + prevchoice_error', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1 + C(probabilityLeft, Treatment(0.5)) + prevchoice_correct + prevchoice_error', 'link_func': z_link_func}
        m = hddm.HDDMRegressor(data, [v_reg, z_reg], include=['z', 'sv'], group_only_nodes=['sv'],
            group_only_regressors=False, keep_regressor_trace=False, p_outlier=0.05)

    return m