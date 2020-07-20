## ANN Custom Dropout
Machine Learning methods to improve cosmological redshifts estimation.

### Datasets

### Experiments

### Requirements
Requires Python 3.6+ to run.

All project dependency are in `requirements.txt` file.

To install all dependencies, run as follows

```shell script
$ pip install -r requirements.txt
```

### Usage
The `main.py` file is the entrypoint script.

For help, run
`$ main.py -h` 

Output for this command shows as follows

```shell script
main.py [-h] [-e EPOCHS] [-dp DROPOUT] [-sc SCALER] [-runs RUNS]
               [-lr LR] [-f NF] [-dataset DS] [-gpu DEVICE] [-xgbr] [-noes]

ANN Experiments

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
### Arguments

### Run

##### ANN with Kalggle dataset Example
In this example, we train a Neural network with 
```shell script
$ main.py -e 2 -dp ErrorBasedDropoutIR -sc StandardScaler -runs 2 -lr 0.001 -f 10 -dataset kaggle_bkp
```
##### XGBoost Regressor with COIN:Teddy Example


### Citation
```latex
@article{
}
```

### Contact

Email me at `raphael.fialho@.eic.cefet-rj.br`
