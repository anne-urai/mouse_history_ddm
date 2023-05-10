# ============================================ #
# MODEL SPECIFICATION
# ============================================ #

import pandas as pd
import hddm, kabuki

def make_model(data, mname):
 
    print('making HDDMnn model')

    if mname == 'ddm_nohist':
                
        v_reg = {'model': 'v ~ 1 + signed_contrast', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1', 'link_func': lambda x:x}
        
        model = 'ddm' # start simple
        hddmnn_model = hddm.HDDMnnRegressor(data,
                                   [v_reg, z_reg],
                                   model = model,
                                   include = hddm.simulators.model_config[model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True, # hierarchical model, parameters per subject
                                   group_only_regressors = False,
                                   informative = False)

    if mname == 'ddm_nohist_stimcat':
        # confirm that the drift rate scales linearly with signed contrast
    
        v_reg = {'model': 'v ~ 1 + C:signed_contrast', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1', 'link_func': lambda x:x}
        
        model = 'ddm' # start simple
        hddmnn_model = hddm.HDDMnnRegressor(data,
                                   [v_reg, z_reg],
                                   model = model,
                                   include = hddm.simulators.model_config[model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True, # hierarchical model, parameters per subject
                                   group_only_regressors = False,
                                   informative = False)
        
    elif mname == 'ddm_prevresp_dcz':
        
        v_reg = {'model': 'v ~ 1 + stimulus + prevresp', 'link_func': lambda x:x}
        z_reg = {'model': 'z ~ 1 + prevresp', 'link_func': lambda x: x}
        
        model = 'ddm' # start simple
        hddmnn_model = hddm.HDDMnnRegressor(data,
                                   [v_reg, z_reg],
                                   model = model,
                                   include = hddm.simulators.model_config[model]['hddm_include'],
                                   p_outlier = 0.05,
                                   is_group_model = True,
                                   group_only_regressors=False,
                                   informative = False)


    return hddmnn_model