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