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
import brainbox.behavior.pyschofit as psy
import paper_behavior_functions as tools
import choice_history_funcs as more_tools

## INITIALIZE A FEW THINGS
sns.set(style="ticks", context="paper", palette="colorblind")
tools.seaborn_style()

datapath = 'data'
figpath = 'figures'
rt_cutoff = 3

# %% ================================= #
# USE THE SAME FILE AS FOR HDDM FITS
# ================================= #

data = pd.read_csv(os.path.join(datapath, 'ibl_trainingchoiceworld.csv'))
data.head(n=10)

# %% ================================= #
# simple choice history psychfuncs
# ================================= #

data = more_tools.compute_choice_history(data) # compute (instead of saving, making the df much bigger)
data['previous_trial'] = 100*data.previous_outcome + 10*data.previous_choice  # for color coding

# plot one curve for each animal, one panel per lab
fig = sns.FacetGrid(data, hue='previous_trial', palette='Paired',
					hue_order=[-90., +110.,  -100., +100.])
fig.map(tools.plot_psychometric, "signed_contrast", "response", "subj_idx")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "psychfuncs_history_average.png"))
plt.close('all')

# plot one curve for each animal, one panel per lab
data['previous_outcome_name'] = data['previous_outcome'].map({1.0:'After rewarded trial',
                                                              -1.0:'After unrewarded trial'})
data['previous_choice_name'] = data['previous_choice'].map({1.0:'right',
                                                              0.0:'left'})
fig = sns.FacetGrid(data, hue='previous_choice_name',
                    row='previous_outcome_name', row_order=['After unrewarded trial', 'After rewarded trial'])
fig.map(tools.plot_psychometric, "signed_contrast", "response", "subj_idx")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.set_titles("{row_name}")
#fig._legend.set_title('Previous choice')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "psychfuncs_history_2cols.png"))
plt.close('all')

fig = sns.FacetGrid(data, hue='previous_choice_name')
fig.map(tools.plot_psychometric, "signed_contrast", "response", "subj_idx").add_legend()
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.set_titles("{row_name}")
fig._legend.set_title('Previous choice')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "psychfuncs_history_prevchoice.png"))
plt.close('all')

# plot one curve for each animal, one panel per lab
data['previous_trial'] = 100*data.previous_outcome + data.previous_choice # color coding
fig = sns.FacetGrid(data, col='subj_idx', col_wrap=7,
					hue='previous_trial',
					palette='Paired', hue_order=[-90., +110.,  -100., +100.])
fig.map(tools.plot_psychometric, "signed_contrast",
        "response", "subj_idx")
fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
fig.despine(trim=True)
fig.savefig(os.path.join(figpath, "psychfuncs_history_permouse.png"))

# %% ================================= #
# DEFINE HISTORY SHIFT FOR LAG 1
# ================================= #

print('fitting psychometric functions...')
data['choice2'] = data.response
data['choice'] = data.response
data['trial'] = data.index
pars = data.groupby(['subj_idx', 'previous_choice', 'previous_outcome']).apply(tools.fit_psychfunc).reset_index()

# instead of the bias in % contrast, take the choice shift at x = 0
# now read these out at the presented levels of signed contrast
pars2 = pd.DataFrame([])
xvec = data.signed_contrast.unique()
for index, group in pars.groupby(['subj_idx', 'previous_choice', 'previous_outcome']):
    # expand
    yvec = psy.erf_psycho_2gammas([group.bias.item(),
                                   group.threshold.item(),
                                   group.lapselow.item(),
                                   group.lapsehigh.item()], xvec)
    group2 = group.loc[group.index.repeat(
        len(yvec))].reset_index(drop=True).copy()
    group2['signed_contrast'] = xvec
    group2['response'] = 100 * yvec
    # add this
    pars2 = pars2.append(group2)

# only pick psychometric functions that were fit on a reasonable number of trials...
pars2 = pars2[(pars2.signed_contrast == 0)]

# compute history-dependent bias shift
pars3 = pd.pivot_table(pars2, values='response',
                       index=['subj_idx', 'previous_outcome'],
                       columns=['previous_choice']).reset_index()
pars3['history_shift'] = pars3[1.0] - pars3[0.0]
pars4 = pd.pivot_table(pars3, values='history_shift',
                       index=['subj_idx'],
                       columns=['previous_outcome']).reset_index()
print(pars4.describe())

# ================================= #
# STRATEGY SPACE
# ================================= #

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=[3.5, 3.5])
sns.scatterplot(x=pars4[1.], y=pars4[-1.], alpha=0.8, color='grey', ax=ax, legend=False)
ax.set_xlabel("History dependence after correct\n($\Delta$ rightward choice (%) at 0% contrast)")
ax.set_ylabel("History dependence after error\n($\Delta$ rightward choice (%) at 0% contrast)")
ax.set(xticks=[-20, 0, 20, 40, 60], yticks=[-20, 0, 20, 40, 60])

sns.despine(trim=True)
ax.axhline(linestyle=':', color='darkgrey')
ax.axvline(linestyle=':', color='darkgrey')
fig.tight_layout()
fig.savefig(os.path.join(figpath, "history_strategy.png"))
plt.close("all")
print('strategy space')

# ================================= #
# DEPENDENCE ON PREVIOUS CONTRAST
# ================================= #

print('fitting psychometric functions, NOW ALSO BASED ON PREVIOUS CONTRAST...')
data['choice2'] = data.response
data['choice'] = data.response
data['trial'] = data.index
pars = data.groupby(['subj_idx', 'previous_choice', 'previous_outcome', 'previous_contrast']).apply(
    tools.fit_psychfunc).reset_index()

# instead of the bias in % contrast, take the choice shift at x = 0
# now read these out at the presented levels of signed contrast
pars2 = pd.DataFrame([])
xvec = data.signed_contrast.unique()
for index, group in pars.groupby(['subj_idx', 'previous_choice', 'previous_outcome', 'previous_contrast']):
    # expand
    yvec = psy.erf_psycho_2gammas([group.bias.item(),
                                   group.threshold.item(),
                                   group.lapselow.item(),
                                   group.lapsehigh.item()], xvec)
    group2 = group.loc[group.index.repeat(
        len(yvec))].reset_index(drop=True).copy()
    group2['signed_contrast'] = xvec
    group2['response'] = 100 * yvec
    # add this
    pars2 = pars2.append(group2)

# only pick psychometric functions that were fit on a reasonable number of trials...
pars2 = pars2.loc[pars2.ntrials > 50, :]
pars2 = pars2[(pars2.signed_contrast == 0)]

# compute history-dependent bias shift
pars3 = pd.pivot_table(pars2, values='response',
                       index=['subj_idx', 'previous_outcome', 'previous_contrast'],
                       columns=['previous_choice']).reset_index()
pars3['history_shift'] = pars3[1.0] - pars3[0.0]
# move the 100% closer
pars3['previous_contrast'] = pars3.previous_contrast * 100
pars3.loc[pars3.previous_contrast == 100, 'previous_contrast'] = 40

# # ================================= #

print('fitting psychometric functions, NOW ALSO BASED ON NEXT CONTRAST...')
data['choice2'] = data.response
data['choice'] = data.response
data['trial'] = data.index
pars = data.groupby(['subj_idx', 'next_choice', 'next_outcome', 'next_contrast']).apply(
    tools.fit_psychfunc).reset_index()

# instead of the bias in % contrast, take the choice shift at x = 0
# now read these out at the presented levels of signed contrast
pars2 = pd.DataFrame([])
xvec = data.signed_contrast.unique()
for index, group in pars.groupby(['subj_idx', 'next_choice', 'next_outcome', 'next_contrast']):
    # expand
    yvec = psy.erf_psycho_2gammas([group.bias.item(),
                                   group.threshold.item(),
                                   group.lapselow.item(),
                                   group.lapsehigh.item()], xvec)
    group2 = group.loc[group.index.repeat(
        len(yvec))].reset_index(drop=True).copy()
    group2['signed_contrast'] = xvec
    group2['response'] = 100 * yvec
    # add this
    pars2 = pars2.append(group2)

# only pick psychometric functions that were fit on a reasonable number of trials...
pars2 = pars2[(pars2.signed_contrast == 0)]

# compute history-dependent bias shift
pars4 = pd.pivot_table(pars2, values='response',
                       index=['subj_idx', 'next_outcome', 'next_contrast'],
                       columns=['next_choice']).reset_index()
pars4['future_shift'] = pars4[1.0] - pars4[0.0]

pars4['previous_outcome'] = pars4.next_outcome
pars4['previous_contrast'] = pars4.next_contrast
pars4['previous_contrast'] = pars4.previous_contrast * 100
pars4.loc[pars4.previous_contrast == 100, 'previous_contrast'] = 40

# merge and subtract the future shift from each history shift
pars5 = pd.merge(pars4, pars3,
                 on=['subj_idx', 'previous_outcome', 'previous_contrast'])
pars5['history_shift_corrected'] = pars5['history_shift'] - pars5['future_shift']

# ================================= #
# PLOT PREVIOUS CONTRAST-DEPENDENCE
# ================================= #

plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=[6,3], sharex=True, sharey=True)
# sns.lineplot(data=pars3, x='previous_contrast', y='history_shift',
#              hue='previous_outcome', ax=ax[0], legend=False,
#              marker='o', units='subj_idx', estimator=None, linewidth=0, alpha=0.5,
#              hue_order=[-1., 1.], palette=sns.color_palette(["tomato", "seagreen"]))
sns.lineplot(data=pars3, x='previous_contrast', y='history_shift',
             hue='previous_outcome', ax=ax[0], legend=False, estimator=np.median,
             err_style='bars', marker='o', hue_order=[-1., 1.],
             palette=sns.color_palette(["firebrick", "forestgreen"]))
ax[0].set(ylabel='$\Delta$ Choice bias (%)',
          xlabel='Previous contrast (%)',
          xticks=[0, 6, 12, 25, 40],
          xticklabels=['0', '6', '12', '25', '100'],
          title='Uncorrected')
ax[0].axhline(color='grey')

# sns.lineplot(data=pars5, x='previous_contrast', y='history_shift_corrected',
#              hue='previous_outcome', ax=ax[1], legend=False,
#              marker='o', units='subj_idx', estimator=None, linewidth=0, alpha=0.5,
#              hue_order=[-1., 1.], palette=sns.color_palette(["tomato", "seagreen"]))
sns.lineplot(data=pars5, x='previous_contrast', y='history_shift_corrected',
             hue='previous_outcome', ax=ax[1], legend=False, estimator=np.median,
             err_style='bars', marker='o', hue_order=[-1., 1.],
             palette=sns.color_palette(["firebrick", "forestgreen"]))
ax[1].axhline(color='grey')

ax[1].set(ylabel='$\Delta$ Choice bias (%)',
          xlabel='Previous contrast (%)',
          xticks=[0, 6, 12, 25, 40],
          xticklabels=['0', '6', '12', '25', '100'],
          title='Corrected')
sns.despine(trim=True)
fig.tight_layout()
#fig.savefig(os.path.join(figpath, "history_prevcontrast.pdf"))
fig.savefig(os.path.join(figpath, "history_prevcontrast.png"))
plt.close("all")

# %%
