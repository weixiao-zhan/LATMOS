
# V.A Automata Learning from Positive Demonstrations

1. Prepare the random walk data
```{bash}
cd data
python gen_spot_data.py
```

2. To evaluate ALERGIA: see `ALERGIA.ipynb`

3. To evaluate SPLearn: see `SPLearn.ipynb`

4. To evaluate LATMOS: see `LATMOS.ipynb`


### File Structure
```
.
├── ALERGIA.ipynb
├── LATMOS.ipynb
├── SPLearn.ipynb
├── data
│   ├── Aleria_{LTL_idx}.pdf
│   ├── gen_spot_data.ipynb
│   ├── gen_spot_data.py
│   └── spot_{LTL_idx}_{train_var}_{test_var}.npz
└── readme.md
```