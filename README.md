### Code to fit DDMs to behavioral data from IBL mice

Anne Urai, Leiden University, 2023
Questions? a.e.urai@fsw.leidenuniv.nl

---

##### Requirements
- install the [iblenv](https://github.com/int-brain-lab/iblenv) for getting IBL data
- install the [hddmnn_env2](https://github.com/hddm-devs/hddm#installation) for fitting HDDM models. See [here](https://github.com/anne-urai/2022_Urai_choicehistory_MEG/blob/main/hddmnn_env2.yml) for a `.yml` environment file.

##### Instructions

`conda activate iblenv`

```python
get_data.py # will grab data from IBL public server
figure1a_plot_behavior.py # plots basic things about the data
figure1b_choice_history.py # fits basic psychometric functions with history terms
```

Then, on ALICE:
`hddm_submit.sh -d 0 -m 0` (will activate `hddm_env2`)

And plot the results of these fits
```python
figure2_hddm.py
```

