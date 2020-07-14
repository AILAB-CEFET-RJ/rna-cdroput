## RNA Custom Dropout
## rna-cdroput
ANN using a custom dropout to estimate cosmological redshifts.


### Usage
```shell script
usage: main.py [-h] [-e EPOCHS] [-dp DROPOUT] [-sc SCALER] [-runs RUNS]
               [-lr LR] [-f NF] [-dataset DS] [-gpu DEVICE] [-xgbr] [-noes]

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
  -gpu DEVICE  GPU device name. Default is device name position 0.
  -xgbr        Run XGBoostRegressor instead of ANN.
  -noes        Disable early stop.
```