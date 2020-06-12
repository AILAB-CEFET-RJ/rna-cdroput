## RNA Custom Dropout
## rna-cdroput
RNA using a custom droput to estimate cosmological redshifts.


### Usage
```shell script
usage: main.py [-h] EPOCHS DROPOUT RUNS LR NF DS

RNA Experiments

positional arguments:
  EPOCHS      Epochs.
  DROPOUT     Dropout class to use.
  RUNS        Total runs.
  LR          Learning rate.
  NF          Number of features.
  DS          Dataset to use [teddy|happy|kaggle|kaggle_bkp].

optional arguments:
  -h, --help  show this help message and exit
```