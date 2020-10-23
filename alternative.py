import argparse

from numpy.random import seed

import tensorflow as tf

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


def build_cfg(D, args):
    device_name = tf.test.gpu_device_name()

    args.dp = None
    args.ir = None
    args.dt = None
    args.xgbr = None
    args.hl1 = None
    args.hl2 = None

    cfg = Object()
    cfg.device_name = device_name
    cfg.D = D
    cfg.args = args
    cfg.learning_rate = 0
    cfg.epochs = None

    return cfg


def parser():
   parser = argparse.ArgumentParser(description='ANN Experiments')
   parser.add_argument('-mf', metavar='MF', help='Model file to pred')
   parser.add_argument('-dataset', metavar='DS', help='Dataset to use [teddy|happy|kaggle|kaggle_bkp].')
   parser.add_argument('-coin_val', metavar='VALSET', help='Use a validation set from COIN data [B|C|D].')
   parser.add_argument('-f', metavar='NF', type=int, help='Number of features.')
   parser.add_argument('-sc', metavar='SCALER', help='Scaler class to use.')

   return parser


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    dataset_name = args.dataset
    model_file = args.mf
    coin_val = args.coin_val
    num_features = args.f
    scaler_opt = args.sc

    seed(42)
    tf.random.set_seed(42)

    dh.download_data(dataset_name, coin_val)
    df, df_val = dh.load_dataframe(dataset_name, coin_val)

    dh.filter_col(df)
    scaler_to_use = reg.select_scaler(scaler_opt)

    if coin_val:
        dh.filter_col(df_val)
        x_train, y_train, x_test, y_test, x_val, y_val, scaler = dh.build_dataset_coin_data(df, df_val, num_features, scaler_to_use)
    else:
        x_train, y_train, x_test, y_test, x_val, y_val, scaler = dh.build_dataset(df, num_features, scaler_to_use)

    print('x_train.shape: ', x_train.shape)
    print('x_val.shape: ', x_val.shape)
    print('x_test.shape: ', x_test.shape)

    N = x_train.shape[0] # number of data points (train)
    D = x_train.shape[1]  # number of features

    d = build_d(x_train, y_train, x_test, y_test, x_val, y_val)
    outputs = {}

    print("## Run ANN ##")
    cfg = build_cfg(D, args)

    print("#### NO TRAINING ####")

    model = t.load_model_data(model_file)
    model_data = t.do_scoring_over(d, cfg, [model])

    outputs = model.predict(x_test)

    t.serialize_results(y_test.flatten(), outputs.flatten(), cfg, coin_val)
