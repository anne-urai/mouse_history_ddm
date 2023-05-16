### Code to fit DDMs to behavioral data from IBL mice

Anne Urai, Leiden University, 2023
Questions? a.e.urai@fsw.leidenuniv.nl

---

##### Requirements
- install the [iblenv](https://github.com/int-brain-lab/iblenv) for getting IBL data
- install the [hddmnn_env](https://github.com/hddm-devs/hddm#installation) for fitting HDDMnn models. See [here](https://github.com/anne-urai/2022_Urai_choicehistory_MEG/blob/main/hddmnn_env2.yml) for a `.yml` environment file.

##### Instructions

`conda activate iblenv`

```python
get_data.py # will grab data from IBL public server
preprocess_data.py # select good RTs to work with
figure1a_plot_behavior.py # plots basic things about the data
figure1b_choice_history.py # fits basic psychometric functions with history terms
figure1c_history_strategy.py
```

Then, on ALICE:
`sbatch hddmnn_submit.sh 0 0` (will activate `hddmnn_env` and submit an ALICE job)
run several models on the same dataset: `for m in {0..4}: do sbatch hddmnn_submit.sh 0 $m; done`

And plot the results of these fits
```python
figure2_hddm.py
```

