import math
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


def build_cfg(D, neurons_0, neurons_1, learning_rate, epochs, num_runs, args):
    device_name = tf.test.gpu_device_name()

    if args.gpu:
        device_name = args.gpu

    cfg = Object()
    cfg.device_name = device_name
    cfg.learning_rate = learning_rate
    cfg.epochs = epochs
    cfg.D = D
    cfg.l1_units = neurons_0
    cfg.l2_units = neurons_1
    cfg.num_runs = num_runs
    cfg.args = args
    cfg.no_early_stopping = args.noes
    cfg.include_errors = args.ierr

    bs = 32
    if args.bs:
        bs = args.bs
    print(f'Using batch size: {bs}')
    cfg.batch_size = bs

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
   parser.add_argument('-blm', metavar='BASELINE', help='Run baseline methods [xgb|knn|rforest|lreg].')
   parser.add_argument('-noes', action='store_true', help='Disable early stop.')
   parser.add_argument('-subs', metavar='SIZE', type=int, help='Subsample size. If pass, dataset full size will be used.')
   parser.add_argument('-rmne', action='store_true', help='Remove negative entries.')
   parser.add_argument('-cut', action='store_true', help='Remove negative entries, cut 25 in u and 1.0 in all erros.')
   parser.add_argument('-hl1', metavar='HL1', type=int, default=0, help='Force amount of units in hidden layer 1.')
   parser.add_argument('-hl2', metavar='HL2', type=int, default=0, help='Force amount of units in hidden layer 2.')
   parser.add_argument('-bs', metavar='BATCH', type=int, default=0, help='Batch size.')
   parser.add_argument('-coin_val', metavar='VALSET', help='Use a validation set from COIN data [B|C|D].')
   parser.add_argument('-mo', action='store_true', help='Reload all models trained stored previously. Skip train phase.')
   parser.add_argument('-ir', action='store_true', help='Apply Isotonic Regression.')
   parser.add_argument('-dt', action='store_true', help='Apply Decision Tree.')
   parser.add_argument('-ierr', action='store_true', help='Include errors as features on custom ANNs.')
   parser.add_argument('-dz', action='store_true', default=False, help='Do regression for bands on custom ANNs.')
   parser.add_argument('-tt', action='store_true', default=False, help='Apply method on coin test dataset')



   return parser


def apply_transforms(dataframe, dropout_opt, subsample, dataset_name, rmne, cuts, args):
    df = dataframe

    if rmne or cuts:
        df = dh.filter_negative_data(df, dataset_name)

    if cuts:
        df = dh.cut_all_val_errs(df, dataset_name, 1.0)
        df = dh.cut_val_band(df, 'u', 25.0)
        df = dh.cut_flag(df, 'clean', 0)

    if subsample is not None:
        subs_df = df.sample(n=subsample, random_state=42)
        print(f"Using subsample {subs_df.shape[0]} of {df.shape[0]}.")
        df = subs_df
    else:
        print(f"Using full sample {df.shape[0]}.")

    if dropout_opt == 'ErrorBasedDropoutIR' or args.ir:
        df = t.apply_isotonic_regression(df, dataset_name, args.dz)
    if dropout_opt == 'ErrorBasedDropoutDT' or args.dt:
        df = t.apply_decision_tree_regression(df, dataset_name)

    return df


def run_xgbr(d, cfg):
    print("## Run XGBoostRegressor ##")
    if skip_training_over:
        print("#### SKIP TRAINING ####")
    else:
        t.do_xgbr_training_runs(d, cfg)
    models = t.load_baseline_models_data(cfg)
    model_data = t.do_baseline_scoring_over(d, cfg, models)
    model = t.get_best_model(model_data)
    outputs = model.predict(x_test)

    return outputs


def run_lreg(d, cfg):
    print("## Run LinearRegressor ##")
    if skip_training_over:
        print("#### SKIP TRAINING ####")
    else:
        t.do_lreg_training_runs(d, cfg)
    models = t.load_baseline_models_data(cfg)
    model_data = t.do_baseline_scoring_over(d, cfg, models)
    model = t.get_best_model(model_data)
    outputs = model.predict(x_test)

    return outputs


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
    subsample = args.subs
    coin_val = args.coin_val
    skip_training_over = args.mo
    cuts = args.cut
    include_errors = args.ierr
    baseline = args.blm

    seed(42)
    tf.random.set_seed(42)

    if 'ErrorOnlyDropout' == args.dp:
        args.tt = True

    dh.download_data(dataset_name, coin_val)
    df, df_val = dh.load_dataframe(dataset_name, coin_val)
    scaler_to_use = reg.select_scaler(scaler_opt)

    df = apply_transforms(df, dropout_opt, subsample, dataset_name, args.rmne, cuts, args)
    df = dh.filter_col(df)
    df_ids_map = dh.empty()

    if coin_val:
        dh.filter_col(df_val)
        if args.tt:
            if args.ir:
                df_val = t.apply_ir_over_test(df, df_val)
            if args.dt:
                df_val = t.apply_dt_over_test(df, df_val)

        x_train, y_train, x_test, y_test, x_val, y_val, scaler = dh.build_dataset_coin_data(df, df_val, num_features, scaler_to_use, args.tt)
    else:
        if args.dp:
            df.insert(df.columns.get_loc("objid"), 'refid', range(len(df)))
            df_ids_map = df[['objid', 'refid']]
            df_ids_map.to_csv('refids.csv')

        df.drop(columns=['objid'], axis=1, inplace=True)
        include_id = args.dp != None
        x_train, y_train, x_test, y_test, x_val, y_val, scaler = dh.build_dataset(df, num_features, scaler_to_use, include_id=include_id)

        if args.dp:
            dh.dump_test_set(x_test, y_test)

    print('x_train.shape: ', x_train.shape)
    print('x_val.shape: ', x_val.shape)
    print('x_test.shape: ', x_test.shape)

    N = x_train.shape[0] # number of data points (train)
    D = x_train.shape[1]  # number of features

    d = build_d(x_train, y_train, x_test, y_test, x_val, y_val)
    outputs = {}

    if baseline == 'xgb':
        cfg = build_cfg(D, 0, 0, learning_rate, epochs, num_runs, args)
        outputs = run_xgbr(d, cfg)

    if baseline == 'lreg':
        cfg = build_cfg(D, 0, 0, learning_rate, epochs, num_runs, args)
        outputs = run_lreg(d, cfg)

    if baseline == None:
        print("## Run ANN ##")
        dropout = reg.select_dropout(dropout_opt, include_errors)
        f = D
        if dropout:
            f = 5
        if include_errors:
            f = 10

        neurons_0 = math.ceil(2 * f / 3)
        neurons_1 = math.ceil(f / 2)

        if args.hl1:
            neurons_0 = args.hl1
        else:
            args.hl1 = neurons_0

        if args.hl2:
            neurons_1 = args.hl2
        else:
            args.hl2 = neurons_1

        cfg = build_cfg(D, neurons_0, neurons_1, learning_rate, epochs, num_runs, args)
        cfg.bias_output_layer = y_train.mean()
        cfg.df_ids_map = df_ids_map

        print(f'input dim:{D}, feature dim: {f} for hl_0[{neurons_0}], hl_1[{neurons_1}]')

        if skip_training_over:
            print("#### SKIP TRAINING ####")
        else:
            t.do_training_runs(d, cfg, 0, dropout)

        models = t.load_models_data(cfg)
        model_data = t.do_scoring_over(d, cfg, models)
        model = t.get_best_model(model_data)

        tf.print("##PREDS", output_stream='file://pred_id.log')
        outputs = model.predict(x_test, batch_size=cfg.batch_size)

    t.serialize_results(y_test.to_numpy().flatten(), outputs.flatten(), cfg, coin_val)
