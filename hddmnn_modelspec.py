# ============================================ #
# MODEL SPECIFICATION
# ============================================ #

import pandas as pd
import hddm, kabuki

def make_model(data, mname_full):
 
    print('making HDDMnn model')

    mname_split = mname_full.split('_')
    base_model = mname_split[0]
    mname = "_".join(mname_split[1:])

    print('base model: %s'%base_model)
    print('model name: %s'%mname)

    if mname == 'nohist_stimcat':
        # confirm that the drift rate scales linearly with signed contrast
        # use the regular HDDMnn, not the regressor model, for this

        hddmnn_model = hddm.HDDMnn(data,
                                   model = base_model,
                                   depends_on = {'v': 'signed_contrast'},
                                   include = hddm.simulators.model_config[base_model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True)

    elif mname == 'nohist_stimcat_dummycode':

        # full-rank coding: each contrast level gets its own predictor term
        v_reg = {'model': 'v ~ 0 + C(signed_contrast, Treatment)', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1', 'link_func': lambda x:x}
        
        hddmnn_model = hddm.HDDMnnRegressor(data,
                                   [v_reg, z_reg],
                                   model = base_model,
                                   include = hddm.simulators.model_config[base_model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True, # hierarchical model, parameters per subject
                                   group_only_regressors = False,
                                   informative = False)

    elif mname == 'nohist_stimcat_reducedrankcode':

        # reduced rank coding: reference term for the 0-contrast (effectively the intercept)
        v_reg = {'model': 'v ~ C(signed_contrast, Treatment(0))', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1', 'link_func': lambda x:x}
        
        hddmnn_model = hddm.HDDMnnRegressor(data,
                                   [v_reg, z_reg],
                                   model = base_model,
                                   include = hddm.simulators.model_config[base_model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True, # hierarchical model, parameters per subject
                                   group_only_regressors = False,
                                   informative = False)
        
    elif mname == 'nohist':
                
        v_reg = {'model': 'v ~ 1 + stimulus', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1', 'link_func': lambda x:x}
        
        hddmnn_model = hddm.HDDMnnRegressor(data,
                                   [v_reg, z_reg],
                                   model = base_model,
                                   include = hddm.simulators.model_config[base_model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True, # hierarchical model, parameters per subject
                                   group_only_regressors = False,
                                   informative = False)

    elif mname == 'prevresp_v':
        
        v_reg = {'model': 'v ~ 1 + stimulus + prevresp', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1', 'link_func': lambda x: x}
        
        hddmnn_model = hddm.HDDMnnRegressor(data,
                                   [v_reg, z_reg],
                                   model = base_model,
                                   include = hddm.simulators.model_config[base_model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True,
                                   group_only_regressors=False,
                                   informative = False)
        
    elif mname == 'prevresp_z':
        
        v_reg = {'model': 'v ~ 1 + stimulus', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1 + prevresp', 'link_func': lambda x: x}
        
        hddmnn_model = hddm.HDDMnnRegressor(data,
                                   [v_reg, z_reg],
                                   model = base_model,
                                   include = hddm.simulators.model_config[base_model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True,
                                   group_only_regressors=False,
                                   informative = False)

    elif mname == 'stimcat_prevresp_zv':
        
        v_reg = {'model': 'v ~ C:(signed_contrast) + prevresp', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1 + prevresp', 'link_func': lambda x: x}
        
        hddmnn_model = hddm.HDDMnnRegressor(data,
                                   [v_reg, z_reg],
                                   model = base_model,
                                   include = hddm.simulators.model_config[base_model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True,
                                   group_only_regressors=False,
                                   informative = False)       
    elif mname == 'prevresp_zv':
        
        v_reg = {'model': 'v ~ 1 + stimulus + prevresp', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1 + prevresp', 'link_func': lambda x: x}
        
        hddmnn_model = hddm.HDDMnnRegressor(data,
                                   [v_reg, z_reg],
                                   model = base_model,
                                   include = hddm.simulators.model_config[base_model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True,
                                   group_only_regressors=False,
                                   informative = False)

    else:
        print('model name not recognized!')

    return hddmnn_model


#%% try out different patsy configurations for categorical stimulus coding
# see also https://github.com/anne-urai/MEG/blob/8f78e28a77e67fb8be7a87edb392008b1435b1c9/cleanup_2021/hddm_modelspec_patsytest.py#L7
# patsy docs https://patsy.readthedocs.io/en/latest/categorical-coding.html

import pandas as pd
from patsy import dmatrix, dmatrices

stimulus = [-100., -50., -25., -12.5, -6.25, 0., 6.25, 12.5, 25., 50., 100]

print(dmatrix('1 + stimulus'))
# full-rank coding: each contrast level gets its own predictor term
print(dmatrix("0 + C(stimulus, Treatment)"))

# reduced rank coding: reference term for the 0-contrast (effectively the intercept)
print(dmatrix("C(stimulus, Treatment(0))"))


