"""
basic choice history data on IBL trainingChoiceWorld
Anne Urai, Leiden University, 2023

"""
# %%
import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import brainbox as bb
import utils_plot as tools
import utils_choice_history as more_tools

## INITIALIZE A FEW THINGS
tools.seaborn_style()

datapath = 'data'
figpath = 'figures'

data = pd.read_csv(os.path.join(datapath, 'ibl_trainingchoiceworld_clean.csv'))

# %% ================================= #
# REGULAR PSYCHFUNCS
# ================================= #

fig = sns.FacetGrid(data, hue="subj_idx")
fig.map(tools.plot_psychometric, "signed_contrast", "response",
        "subj_idx", color='lightgrey', alpha=0.3)
# add means on top
for axidx, ax in enumerate(fig.axes.flat):
    tools.plot_psychometric(data.signed_contrast, data.response,
                      data.subj_idx, ax=ax, legend=False, color='darkblue', linewidth=2)

#fig.map(sns.lineplot, "signed_contrast", "response", color='gray', alpha=0.7)     
fig.despine(trim=True)
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
ax.set_title('a. Psychometric function (n = %d)'%data.subj_idx.nunique())
fig.savefig(os.path.join(figpath, "psychfuncs_allmice.png"), dpi=300)
fig.savefig(os.path.join(figpath, "psychfuncs_allmice.pdf"))

# %% ================================= #
# CHRONFUNCS on good RTs
# ================================= #

fig = sns.FacetGrid(data, hue="subj_idx")
fig.map(tools.plot_chronometric, "signed_contrast", "rt", 
    "subj_idx", color='lightgray', alpha=0.3)
for axidx, ax in enumerate(fig.axes.flat):
    tools.plot_chronometric(data.signed_contrast, data.rt,
                      data.subj_idx, ax=ax, legend=False, color='darkblue', linewidth=2)
fig.despine(trim=True)
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
ax.set_title('b. Chronometric function (n = %d)'%data.subj_idx.nunique())
fig.savefig(os.path.join(figpath, "chronfuncs_allmice.png"))
fig.savefig(os.path.join(figpath, "chronfuncs_allmice.pdf"))

