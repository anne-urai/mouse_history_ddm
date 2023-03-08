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
import paper_behavior_functions as tools
import choice_history_funcs as more_tools

## INITIALIZE A FEW THINGS
tools.seaborn_style()

datapath = 'data'
figpath = 'figures'

# %% ================================= #
# USE THE SAME FILE AS FOR HDDM FITS
# ================================= #

data = pd.read_csv(os.path.join(datapath, 'ibl_trainingchoiceworld.csv'))
data.head(n=10)
data['rt'] = data['rt_wheel']

## Compare different measures of RT
sns.scatterplot(x='trial_duration', y='rt_wheel', data=data, marker='.',
                alpha=0.1, hue='id', legend=False)
plt.savefig(os.path.join(figpath, "trialduration_vs_rtwheel.png"))

data['rt_diff'] = data['trial_duration'] - data['rt_wheel']
## Compare different measures of RT
sns.distplot(x='rt_diff', data=data)
plt.savefig(os.path.join(figpath, "trialduration_vs_rtwheel_hist.png"))

# %%
data = more_tools.clean_rts(cutoff=None)

# remove data without a well-estimated RT
data.dropna(subset=['rt'], inplace=True)

# %% ================================= #
#  SUPP: RT CDF
rt_cutoff = 10

f, ax = plt.subplots(1,1,figsize=[3,3])
hist, bins = np.histogram(data.rt, bins=1000)
logbins = np.append(0, np.logspace(np.log10(bins[1]), np.log10(bins[-1]), len(bins)))
ax.hist(data.rt, bins=logbins, cumulative=True, 
    density=True, histtype='step')
ax.set_xscale('log')

# indicate the cutoff we use here
ax.set_xlabel("RT (s)")
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: (
	'{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y), 0)))).format(y)))
ax.set_ylabel('CDF')
# indicate the percentage of trials excluded by rt cutoff
perc = (data.rt < rt_cutoff).mean()
sns.lineplot(x=[0, rt_cutoff], y=[perc, perc], style=0, color='k', 
            dashes={0: (2, 1)}, lw=1, legend=False)
sns.lineplot(x=[rt_cutoff, rt_cutoff], y=[0, perc], style=0, color='k', 
            dashes={0: (2, 1)}, lw=1, legend=False)
ax.set(ylim=[-0.01, 1], xlim=[0.02, 59])
sns.despine()
plt.tight_layout()
f.savefig(os.path.join(figpath, "rt_cdf.png"))
# ToDo: where in the session do slow RTs occur? presumably at the end

# %% ================================= #
## NOW REMOVE THOSE LOW RTs


# %% ================================= #
# REGULAR PSYCHFUNCS
# ================================= #

fig = sns.FacetGrid(data[data['rt'] <= rt_cutoff], hue="subj_idx")
fig.map(tools.plot_psychometric, "signed_contrast", "response",
        "subj_idx", color='gray', alpha=0.3)
# add means on top
for axidx, ax in enumerate(fig.axes.flat):
    tools.plot_psychometric(data.signed_contrast, data.response,
                      data.subj_idx, ax=ax, legend=False, color='k', linewidth=2)

#fig.map(sns.lineplot, "signed_contrast", "response", color='gray', alpha=0.7)     
fig.despine(trim=True)
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
ax.set_title('a. Psychometric function (n = %d)'%data.subj_idx.nunique())
fig.savefig(os.path.join(figpath, "psychfuncs_allmice.png"), dpi=300)

# %% ================================= #
# CHRONFUNCS on good RTs
# ================================= #

fig = sns.FacetGrid(data, hue="subj_idx")
fig.map(tools.plot_chronometric, "signed_contrast", "rt", 
    "subj_idx", color='gray', alpha=0.7)
for axidx, ax in enumerate(fig.axes.flat):
    tools.plot_chronometric(data.signed_contrast, data.rt,
                      data.subj_idx, ax=ax, legend=False, color='k', linewidth=2)
fig.despine(trim=True)
fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
ax.set_title('b. Chronometric function (n = %d)'%data.subj_idx.nunique())

fig.savefig(os.path.join(figpath, "chronfuncs_allmice.png"))
print('chronometric functions')
