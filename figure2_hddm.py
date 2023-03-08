"""
plot HDDM model, with history terms, to data from IBL mice
Anne Urai, 2018 CSHL

"""

# ============================================ #
# GETTING STARTED
# ============================================ #

import pandas as pd
import numpy as np
import scipy as sp
import sys, os, glob
import pickle

import matplotlib
matplotlib.use('Agg') # to still plot even when no display is defined
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

# more handy imports
from IPython import embed as shell
# import datajoint as dj
from hddm_funcs import corrfunc, results_long2wide
import corrstats

# define path to save the data and figures
datapath = os.path.join(os.path.expanduser('~'), 'Data/HDDM_IBL')

# MAKE THE FIGURE, divide subplots using gridspec
pal = sns.color_palette("Paired")
pal2 = pal[2:4] + pal[0:2] + pal[8:10]

# ============================================ #
# sanity check, drift rate by contrast
# ============================================ #

md = results_long2wide(pd.read_csv(os.path.join(datapath, 'trainingchoiceworld', 'nohist', 'results_combined.csv')))
plt.close('all')
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,4))
sns.stripplot(data=md['v'], ax=ax, color='grey', zorder=0)
sns.factorplot(data=md['v'], ax=ax, color='k', zorder=100)
ax.set(xlabel='Contrast (%)', ylabel='Drift rate (v)')
sns.despine(trim=True)
fig.tight_layout()
fig.savefig(os.path.join(datapath, 'figures', 'driftrate_nohist.pdf'))

# ============================================ #
# COMPUTE HISTORY SHIFT AND CORRELATE WITH BEHAVIOR
# ============================================ #

d = 'trainingchoiceworld'
data = pd.read_csv(os.path.join(datapath, 'ibl_%s.csv' % 'trainingchoiceworld'))
data['repeat'] = (data.response == data.previous_choice)
rep = data.groupby(['subj_idx', 'previous_outcome'])['repeat'].mean().reset_index()
rep = rep.pivot(index='subj_idx', columns='previous_outcome', values='repeat').reset_index()
rep = rep.rename(columns={1.0: 'repeat_prevcorrect', -1.0: 'repeat_preverror'})
# also add a measure of repetition without previous outcome
rep2 = data.groupby(['subj_idx'])['repeat'].mean().reset_index()
rep = pd.merge(rep, rep2, on='subj_idx')
rep = rep.sort_values(by=['repeat'])

# ============================================ #
# plot overall repetition barplot
# ============================================ #

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,4))
g = sns.scatterplot(y="subj_idx", x="repeat",
                data=rep, color="0.3", ax=ax)
#sns.despine(trim=True)
ax.set(ylabel='# Mouse', xlabel='P(repeat)', yticks=[])
ax.axvline(x=0.5, color='darkgrey')
#fig.despine(trim=True)
fig.tight_layout()
fig.savefig(os.path.join(datapath, 'figures', 'repetition.pdf'))

# ============================================ #
# FIRST, MODEL COMPARISON
# ============================================ #

models = ['nohist',  'prevchoice_z',  'prevchoiceoutcome_z',
          'prevchoice_dc','prevchoiceoutcome_dc',
          'prevchoice_dcz', 'prevchoiceoutcome_dcz']
models = ['nohist',  'prevchoice_z',
          'prevchoice_dc',
          'prevchoice_dcz']
mdcomp = pd.DataFrame()
for mod in models:
    print(mod)
    tmp_md = pd.read_csv(os.path.join(datapath, 'trainingchoiceworld', mod, 'model_comparison.csv'))
    tmp_md['model'] = mod
    mdcomp = mdcomp.append(tmp_md, ignore_index=True)

# subtract baseline model
cols = ['aic', 'bic', 'dic']
for c in cols:
    mdcomp[c] = mdcomp[c] - mdcomp.loc[mdcomp.model == 'nohist', c].item()
mdcomp = mdcomp[mdcomp.model != 'nohist']

fig, ax = plt.subplots(ncols=2, nrows=1)
pal3 = pal[3:4] + pal[1:2] + pal[9:10]
sns.catplot(x="model", y="dic", ax=ax[0], data=mdcomp, kind='bar', palette=pal3)
ax[0].set(ylabel=r'$\Delta$DIC', xlabel='', xticklabels=['z', 'vbias', 'both'])
sns.catplot(x="model", y="aic", ax=ax[1], data=mdcomp, kind='bar', palette=pal3)
ax[1].set(ylabel=r'$\Delta$AIC', xlabel='', xticklabels=['z', 'vbias', 'both'])
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=40, ha='right')
fig.tight_layout()
fig.savefig(os.path.join(datapath, 'figures', 'modelcomp_trainingchoiceworld.pdf'))

# ============================================ #
# now load in the models we need
# ============================================ #

md = pd.read_csv(os.path.join(datapath, d, 'prevchoiceoutcome_dcz', 'results_combined.csv'))
md_wide = results_long2wide(md)

# COMPUTE THE SAME THING FROM HDDM COLUMN NAMES
md_wide['dcshift_preverror']   = md_wide['dc']['1.0.-1.0'] - md_wide['dc']['0.0.-1.0']
md_wide['dcshift_prevcorrect'] = md_wide['dc']['1.0.1.0'] - md_wide['dc']['0.0.1.0']
md_wide['zshift_preverror']    = md_wide['z']['1.0.-1.0'] - md_wide['z']['0.0.-1.0']
md_wide['zshift_prevcorrect']  = md_wide['z']['1.0.1.0'] - md_wide['z']['0.0.1.0']
md_wide = md_wide[['subj_idx', 'dcshift_preverror', 'dcshift_prevcorrect', 'zshift_preverror', 'zshift_prevcorrect']]
md_wide.columns = md_wide.columns.droplevel(1)

# md_wide.dropna(inplace=True)
md_wide = pd.merge(md_wide, rep, on='subj_idx')

fig, ax = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=False)
corrfunc(x=md_wide.zshift_prevcorrect, y=md_wide.repeat_prevcorrect, ax=ax[0,0], color='0.3')
ax[0,0].set(xlabel='', ylabel='P(repeat) after correct')
corrfunc(x=md_wide.dcshift_prevcorrect, y=md_wide.repeat_prevcorrect, ax=ax[0,1], color='0.3')
ax[0,1].set(xlabel='', ylabel=' ')
corrfunc(x=md_wide.zshift_preverror, y=md_wide.repeat_preverror, ax=ax[1,0], color='firebrick')
ax[1,0].set(xlabel='History shift in z', ylabel='P(repeat) after error')
corrfunc(x=md_wide.dcshift_preverror, y=md_wide.repeat_preverror, ax=ax[1,1], color='firebrick')
ax[1,1].set(xlabel='History shift in drift bias', ylabel='')

sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(datapath, 'figures', 'scatterplot_trainingchoiceworld_prevchoiceoutcome_dcz.pdf'))

# =============== #
# ADD STATS BETWEEN THE TWO CORRELATION COEFFICIENTS
# =============== #

# huge correlation plot
g = sns.pairplot(md_wide, vars=['repeat_prevcorrect', 'repeat_preverror',
                          'zshift_prevcorrect', 'zshift_preverror',
                         'dcshift_prevcorrect', 'dcshift_preverror'],
                 kind='reg')
g.savefig(os.path.join(datapath, 'figures', 'scatterplot_prevchoiceoutcome_dcz_allvars.pdf'))

print(md_wide[['repeat_prevcorrect', 'repeat_preverror',
                          'zshift_prevcorrect', 'zshift_preverror',
                         'dcshift_prevcorrect', 'dcshift_preverror']].corr())

# ============================================ #
# now load in the models we need
# ============================================ #

md = pd.read_csv(os.path.join(datapath, d, 'prevchoice_dcz', 'results_combined.csv'))
md_wide = results_long2wide(md)

# COMPUTE THE SAME THING FROM HDDM COLUMN NAMES
md_wide['dcshift'] = md_wide['dc']['1.0'] - md_wide['dc']['0.0']
md_wide['zshift'] = md_wide['z']['1.0'] - md_wide['z']['0.0']
md_wide = md_wide[['subj_idx', 'dcshift', 'zshift']]
md_wide.columns = md_wide.columns.droplevel(1)
md_wide = pd.merge(md_wide, rep, on='subj_idx')

fig, ax = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=False, figsize=(6,3))
corrfunc(x=md_wide.zshift, y=md_wide.repeat, ax=ax[0], color=pal2[1])
ax[0].set(xlabel='History shift in z', ylabel='P(repeat)')
corrfunc(x=md_wide.dcshift, y=md_wide.repeat, ax=ax[1], color=pal2[3])
ax[1].set(xlabel='History shift in drift bias', ylabel=' ')

# ADD STEIGERS TEST ON TOP
# x = repeat, y = zshift, z = dcshift
tstat, pval = corrstats.dependent_corr(sp.stats.spearmanr(md_wide.zshift, md_wide.repeat, nan_policy='omit')[0],
                                       sp.stats.spearmanr(md_wide.dcshift, md_wide.repeat, nan_policy='omit')[0],
                                       sp.stats.spearmanr(md_wide.zshift, md_wide.dcshift, nan_policy='omit')[0],
                                        len(md_wide),
                                       twotailed=True, conf_level=0.95, method='steiger')
deltarho = sp.stats.spearmanr(md_wide.zshift, md_wide.repeat, nan_policy='omit')[0] - \
           sp.stats.spearmanr(md_wide.dcshift, md_wide.repeat, nan_policy='omit')[0]
if pval < 0.0001:
    fig.suptitle(r'$\Delta\rho$ = %.3f, p = < 0.0001'%(deltarho), fontsize=10)
else:
    fig.suptitle(r'$\Delta\rho$ = %.3f, p = %.4f' % (deltarho, pval), fontsize=10)
sns.despine(trim=True)
plt.tight_layout()
fig.savefig(os.path.join(datapath, 'figures', 'scatterplot_trainingchoiceworld_prevchoice_dcz.pdf'))

# ============================================ #
# CONDITIONAL BIAS FUNCTIONS
# ============================================ #


