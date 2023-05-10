   
import pandas as pd
import numpy as np
   
def compute_choice_history(trials):

    # append choice history 
    trials['previous_choice']   = trials.response.shift(1)
    trials['previous_outcome']  = trials.feedbackType.shift(1)
    trials['previous_contrast'] = np.abs(trials.signed_contrast.shift(1))

    # also append choice future (for correction a la Lak et al.)
    trials['next_choice']       = trials.response.shift(-1)
    trials['next_outcome']      = trials.feedbackType.shift(-1)
    trials['next_contrast']     = np.abs(trials.signed_contrast.shift(-1))

    # remove when not consecutive based on trial_index
    trials_not_consecutive       = (trials.trialnum - trials.trialnum.shift(1)) != 1.
    for col in ['previous_choice', 'previous_outcome', 'previous_contrast']:
        trials.loc[trials_not_consecutive, col] = np.nan

    return trials

def clean_rts(trials, cutoff=None):

    # filter out impossible RTs
    for colname in ['trial_duration', 'rt_wheel']:
        trials.loc[trials[colname] < 0, colname] = np.nan # can't have negative RTs
        trials.loc[trials[colname] > 60, colname] = np.nan # RT can't be longer than the ITI
        if cutoff:
            trials.loc[trials[colname] < cutoff, colname] = np.nan # remove RTs below cutoff, helpful for HDDM

    return trials
