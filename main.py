import math
import os
import argparse

from numpy.random import seed

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from modules import dataset_handle as dh
from modules import training as t
from modules import regularization as reg


class Object(object):
    pass


def build_d(x_train, y_train, x_test, y_test, x_val, y_val):
    d = Object()
    d.x_train = x_train
    d.y_train = y_train
    d.x_test = x_test
    d.y_test = y_test
    d.x_val = x_val
    d.y_val = y_val

    return d


def build_cfg(D, neurons_0, neurons_1, learning_rate, epochs, num_runs, args):
    model_dir = '.'
    patience = int(0.2 * epochs)
    best_weights_filepath = os.path.join(model_dir, 'model_weights.hdf5')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience)

    mcp_save = ModelCheckpoint(
        monitor='val_loss', mode='min', filepath=best_weights_filepath, save_best_only=True
    )

    if args.noes:
        print("early_stopping is disabled")
        callbacks = [mcp_save]
    else:
        callbacks = [early_stopping, mcp_save]

    device_name = tf.test.gpu_device_name()

    if args.gpu:
        device_name = args.gpu

    cfg = Object()
    cfg.device_name = device_name
    cfg.callbacks = callbacks
    cfg.learning_rate = learning_rate
    cfg.epochs = epochs
    cfg.D = D
    cfg.l1_units = neurons_0
    cfg.l2_units = neurons_1
    cfg.num_runs = num_runs
    cfg.args = args

    return cfg


def parser():
   parser = argparse.ArgumentParser(description='ANN Experiments')
   parser.add_argument('-e', metavar='EPOCHS', type=int, help='Epochs.')
   parser.add_argument('-dp', metavar='DROPOUT', help='Dropout class to use.')
   parser.add_argument('-sc', metavar='SCALER', help='Scaler class to use.')
   parser.add_argument('-runs', metavar='RUNS', type=int, help='Total runs.')
   parser.add_argument('-lr', metavar='LR', type=float,help='Learning rate.')
   parser.add_argument('-f', metavar='NF', type=int, help='Number of features.')
   parser.add_argument('-dataset', metavar='DS', help='Dataset to use [teddy|happy|kaggle|kaggle_bkp].')
   parser.add_argument('-gpu', metavar='DEVICE', help='GPU device name. Default is device name position 0.')
   parser.add_argument('-xgbr', action='store_true', help='Run XGBoostRegressor instead of ANN.')
   parser.add_argument('-noes', action='store_true', help='Disable early stop.')
   parser.add_argument('-subs', metavar='SIZE', type=int, help='Subsample size. If pass, dataset full size will be used.')
   parser.add_argument('-rmne', action='store_true', help='Remove negative magnitude entries.')

   return parser


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    dataset_name = args.dataset
    dropout_opt = args.dp
    num_runs = args.runs
    epochs = args.e
    learning_rate = args.lr
    num_features = args.f
    scaler_opt = args.sc
    xgboost = args.xgbr
    subsample = args.subs

    seed(42)
    tf.random.set_seed(42)

    dh.download_data(dataset_name)
    df = dh.load_dataframe(dataset_name)
    scaler_to_use = reg.select_scaler(scaler_opt)

    dh.filter_col(df)

    if subsample is not None:
        subs_df = df.sample(n=subsample, random_state=42)
        print(f"Using subsample {subs_df.shape[0]} of {df.shape[0]}.")
        df = subs_df
    else:
        print(f"Using full sample {df.shape[0]}.")

    if args.rmne:
        df = dh.filter_negative_redshift(df)

    if dropout_opt == 'ErrorBasedDropoutIR':
        df = t.apply_isotonic_regression(df, dataset_name)
    if dropout_opt == 'ErrorBasedDropoutDT':
        df = t.apply_decision_tree_regression(df, dataset_name)

    x_train, y_train, x_test, y_test, x_val, y_val, scaler = dh.build_dataset(df, num_features, scaler_to_use)

    print('x_train.shape: ', x_train.shape)
    print('x_test.shape: ', x_test.shape)
    print('x_val.shape: ', x_val.shape)

    N = x_train.shape[0] # number of data points (train)
    D = x_train.shape[1]  # number of features

    d = build_d(x_train, y_train, x_test, y_test, x_val, y_val)
    outputs = {}

    if xgboost:
        print("## Run XGBoostRegressor ##")
        params = {'n_estimators': epochs,
                  'max_depth': 8,
                  'min_samples_split': 5,
                  'validation_fraction': 0.2,
                  'n_iter_no_change': int(0.2 * epochs),
                  'learning_rate': learning_rate,
                  'loss': 'ls',
                  'random_state': 0
                  }
        cfg = build_cfg(D, 0, 0, learning_rate, epochs, num_runs, args)
        model = t.do_xgbr_training_runs(d, cfg, params)
        outputs = model.predict(x_test)

    else:
        print("## Run ANN ##")
        dropout = reg.select_dropout(dropout_opt)
        f = D
        if dropout:
            f = 5

        neurons_0 = math.ceil(2 * f / 3)
        neurons_1 = math.ceil(f / 2)
        cfg = build_cfg(D, neurons_0, neurons_1, learning_rate, epochs, num_runs, args)

        print(f'input dim:{D}, feature dim: {f} for hl_0[{neurons_0}], hl_1[{neurons_1}]')
        model, hist, all_scores = t.do_training_runs(d, cfg, 0, dropout)
        outputs = model.predict(x_test)

    t.serialize_results(y_test.flatten(), outputs.flatten(), cfg)
