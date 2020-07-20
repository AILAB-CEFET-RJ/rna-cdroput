## ANN Custom Dropout
Machine Learning methods for cosmological redshift estimation.

### Datasets
This project uses the public Kaggle and COIN(Happy/Teddy) datasets.
All datasets are dowloaded from the public source and saved loacally.

### Experiments
[ ... describe all experiments executed in paper (run_all.sh)  ... ]

### Requirements
Requires Python 3.6+ to run and all project dependencies are in `requirements.txt` file.

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

##### Datasets Options [-dataset]
The available argument options are teddy, happy, kaggle and kaggle_bkp.

##### Dropout implementation Options [-dp]
Provide the Droupout implematation class to use. In this project we have the particular `A` and `B` Dropout classes.
Provide ´none´ as argument to disable Dropout layer.

##### Scaler implementation Options [-sc]
Provide the Scaler implematation class to use. In this project we have the particular `A` and `B` Scaler classes.
Provide ´none´ as argument to disable the scaler.


### Run
To run a single experiment to estimate redshifts, run as follows in subsections bellow.

##### ANN with Kalggle dataset Example
In this example, we train a Neural network with 
```shell script
$ main.py -e 50 -dp ErrorBasedDropoutIR -sc StandardScaler -runs 3 -lr 0.001 -f 10 -dataset kaggle_bkp
```
##### XGBoost Regressor with COIN:Teddy Example

##### Run in batch
You can use the `run_all.sh` to setup a batch run of many experiments. 
Edit this file to customize the experiments as you wish.

Finally, run as follows
```shell script
$ ./run_all.sh
```


### Citation
```latex
@article{
}
```

### Contact

Email me at `raphael.fialho@.eic.cefet-rj.br`
