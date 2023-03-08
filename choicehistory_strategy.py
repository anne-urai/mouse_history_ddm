# show how many mice there are in the IBL database that I can use, their age, labs, and project codes

import pandas as pd
import numpy as np
import datajoint as dj
import aging_tables
import behavior_tables
from ibl_pipeline.analyses import behavior as behavioral_analyses

import matplotlib
matplotlib.use('Agg') # to still plot even when no display is defined
import matplotlib.pyplot as plt
import seaborn as sns
import os
figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# QUERY
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
subj_query = (aging_tables.AgeAtEvent & 'age_at_trained is NOT Null') \
             * (subject.Subject & 'subject_nickname NOT LIKE "%les%"' & 'subject_line="C57BL/6J"') \
             * subject.SubjectProject() * subject.SubjectLab() \
             * subject.Species * (subject.Strain & 'strain_name="C57BL/6J"') \
             * (behavioral_analyses.SessionTrainingStatus & 'training_status="trained_1a" OR ' \
                                                          'training_status="trained_1b"') \
             * behavior_tables.ChoiceHistory * acquisition.Session()

# restrict to the stuff we need to fetch
subj_query = subj_query.proj('subject_nickname', 'task_protocol', 'choicehistory_aftercorrect',
                             'choicehistory_aftererror', 'choicehistory_aftercorrect_corrected',
                             'choicehistory_aftererror_corrected', 'lab_name', 'age_at_trained',
                             'age_at_biased')
print(subj_query)
data = subj_query.fetch(format='frame').reset_index()
print(data.describe())
data['age_at_trained_weeks'] = data.age_at_trained / 7
data['task'] = data['task_protocol'].str[14:20]

# average per task protocol and mouse
df2 = data.groupby(['subject_nickname', 'lab_name', 'task']).mean().reset_index()

# PLOT HISTORY STRATEGY PLOT, REPLICATE FIGURE 4G
plt.close('all')
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
sns.lineplot(x='choicehistory_aftercorrect', y='choicehistory_aftererror',
                hue='lab_name', data=df2, ax=ax[0], legend=False,
             units='subject_nickname', estimator=None)
sns.scatterplot(x='choicehistory_aftercorrect', y='choicehistory_aftererror', 
                style='task', hue='lab_name', data=df2, ax=ax[0], legend=False,
                markers={'traini': 'o', 'biased': '^', 'ephysC':'s'})
ax[0].axhline(linestyle=':', color='darkgrey')
ax[0].axvline(linestyle=':', color='darkgrey')
ax[0].set_title('Uncorrected')
ax[0].set_aspect('equal', 'box')

sns.lineplot(x='choicehistory_aftercorrect_corrected', y='choicehistory_aftererror_corrected',
                hue='lab_name', data=df2, ax=ax[1], legend=False,
             units='subject_nickname', estimator=None)
sns.scatterplot(x='choicehistory_aftercorrect_corrected', y='choicehistory_aftererror_corrected',
                style='task', hue='lab_name', data=df2, ax=ax[1], legend=False,
                markers={'traini': 'o', 'biased': '^', 'ephysC':'s'})
ax[1].axhline(linestyle=':', color='darkgrey')
ax[1].axvline(linestyle=':', color='darkgrey')
ax[1].set_title('Corrected')
ax[1].set_aspect('equal', 'box')

plt.tight_layout()
fig.savefig(os.path.join(figpath, 'choice_history_strategy.pdf'))

