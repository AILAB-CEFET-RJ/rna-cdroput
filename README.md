## RNA Custom Dropout
## rna-cdroput
RNA using a custom droput to estimate cosmological redshifts.


### Usage
```shell script
usage: main.py [-h] [-e EPOCHS] [-dp DROPOUT] [-sc SCALER] [-runs RUNS]
               [-lr LR] [-f NF] [-dataset DS]

RNA Experiments

optional arguments:
  -h, --help   show this help message and exit
  -e EPOCHS    Epochs.
  -dp DROPOUT  Dropout class to use.
  -sc SCALER   Scaler class to use.
  -runs RUNS   Total runs.
  -lr LR       Learning rate.
  -f NF        Number of features.
  -dataset DS  Dataset to use [teddy|happy|kaggle|kaggle_bkp].
```