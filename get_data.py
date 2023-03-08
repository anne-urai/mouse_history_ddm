"""
get data from IBL mice
Anne Urai, Leiden University, 2023

"""

# ============================================ #
# GETTING STARTED
# ============================================ #

# %%
import pandas as pd
import numpy as np
import sys, os, time
import seaborn as sns
import matplotlib.pyplot as plt
from ibllib.io.extractors.training_wheel import extract_wheel_moves, extract_first_movement_times
from one.api import ONE # use ONE instead of DJ! more future-proof

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
        password='international')

# define path to save the data and figures
# ToDo: put this somewhere local, rather than in the git repo
datapath = 'data'

# %% 1. QUERY SESSIONS
# https://openalyx.internationalbrainlab.org/docs/#sessions-list

# ToDO: keep track of sessions https://int-brain-lab.github.io/ONE/notebooks/recording_data_access.html
sess_lifespan = one.alyx.rest('sessions', 'list', 
    tag='2021_Q1_IBL_et_al_Behaviour',
    performance_gte=80,
    dataset_types='trials.table',
    task_protocol='training')
# ToDo: more sophisticated selection of sessions based on training status
# https://int-brain-lab.slack.com/archives/CEM70EXTN/p1673429574225009
print(len(sess_lifespan))

# %% 2. LOAD TRIALS

behav = []
for sess in sess_lifespan: # dont use all for now...
    trials_obj = one.load_object(sess['id'], 'trials')
    trials_obj['signed_contrast'] = 100 * np.diff(np.nan_to_num(np.c_[trials_obj['contrastLeft'], 
                                                        trials_obj['contrastRight']]))
    
    # use a very crude measure of RT: trial duration 
    trials_obj['trial_duration'] = trials_obj['response_times'] - trials_obj['goCue_times']

    # better way to define RT: based on wheelMoves, Miles' code from Brainbox
    # copied from https://github.com/int-brain-lab/ibllib/blob/6372d0249289f704eb0f3640ad7243fe65a3c689/brainbox/io/one.py#L666
    # ToDo: talk to Peter Latham about how to define RT
    trials_obj['rt_wheel'] = trials_obj['firstMovement_times'] - trials_obj['goCue_times']

    trials = trials_obj.to_df() # to dataframe
    trials['trialnum'] = trials.index # to keep track of choice history
    trials['response'] = trials['choice'].map({1: 0, 0: np.nan, -1: 1})

    for k in ['subject', 'id', 'start_time']:
        trials[k] = sess[k]
    trials['subj_idx'] = trials.subject
    behav.append(trials)

df = pd.concat(behav)
# continue only with some columns we need
df = df[['id', 'subj_idx', 'start_time', 'signed_contrast', 
         'response', 'trial_duration', 'rt_wheel','feedbackType']]

# %% 4. REFORMAT AND SAVE TRIALS
df.to_csv(os.path.join(datapath, 'ibl_trainingchoiceworld.csv'))
print(os.path.join(datapath, 'ibl_trainingchoiceworld.csv'))
print('%d mice, %d trials'%(df.subj_idx.nunique(),  df.subj_idx.count()))

# %%
df.describe()