import numpy as np
import pandas as pd
import os
import glob
import time
import random

from math import sqrt
from joblib import dump, load

from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import ensemble

from sklearn.linear_model import SGDRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from modules.regularization import ErrorBasedDropoutIR
from modules.regularization import ErrorBasedDropoutDT
from modules.regularization import ErrorBasedInvertedDropout
from modules.regularization import ErrorBasedInvertedRandomDropout
from modules.regularization import EBasedInvDynamicDp
from modules.regularization import ErrorOnlyDropout


_MAP_METHOD_NAMES = {
    None : 'RNA',
    'ErrorBasedDropoutIR': 'RNA-RI',
    'ErrorBasedDropoutDT': 'RNA-AD',
    'ErrorBasedInvertedDropout': 'RNA-',
    'ErrorBasedInvertedRandomDropout': 'RNA-R',
    'EBasedInvDynamicDp': 'RNA-D',
    'ErrorOnlyDropout': 'RNA-E'
}


class Object(object):
    pass


def custom_layer_register():
    return {
        'ErrorBasedDropoutIR': ErrorBasedDropoutIR,
        'ErrorBasedDropoutDT': ErrorBasedDropoutDT,
        'ErrorBasedInvertedDropout': ErrorBasedInvertedDropout,
        'ErrorBasedInvertedRandomDropout': ErrorBasedInvertedRandomDropout,
        'EBasedInvDynamicDp': EBasedInvDynamicDp,
        'ErrorOnlyDropout': ErrorOnlyDropout
    }


def preds_filename(cfg):
    return f"real_x_pred_{model_name_nmnic(cfg)}_{cfg.args.sc}_{cfg.args.dataset}_epochs_{cfg.epochs}_units_{cfg.args.hl1}_{cfg.args.hl2}"


def hist_filename(cfg, run):
    return f"hist_{model_name_nmnic(cfg)}_{cfg.args.sc}_{cfg.args.dataset}_epochs_{cfg.epochs}_units_{cfg.args.hl1}_{cfg.args.hl2}_run_{run}"


def model_name_nmnic(cfg):
    modelname = _MAP_METHOD_NAMES[cfg.args.dp]
    if cfg.args.ir:
        modelname=f"{modelname}RI-Inv"
    if cfg.args.dt:
        modelname=f"{modelname}AD-Inv"
    if cfg.args.blm:
        modelname = cfg.args.blm.upper()
    if cfg.args.dp:
        cf = 5
        if cfg.args.ierr:
            cf = 10

        modelname = f"{modelname}{cf:02d}"
    else:
        modelname = f"{modelname}{cfg.args.f:02d}"

    return modelname


def model_filename(cfg, run):
    modelname = model_name_nmnic(cfg)
    filename = f'model_{modelname}_{cfg.args.dataset}_{cfg.args.sc}_epochs_{cfg.epochs}_units_{cfg.args.hl1}_{cfg.args.hl2}'

    if run is not None:
        filename = f'{filename}_run_{run}'

    return filename


def clean_dir(cfg, run):
    model_file_mask = f"{model_filename(cfg, run)}*"
    model_files = glob.glob(model_file_mask)
    models = {}
    for mf in model_files:
        key = int(mf.split('_')[-1].split('.')[0])
        models[key] = mf

    models = {k: models[k] for k in sorted(models)}
    wrost_models_files = list(models)[:-1]

    for wmf in wrost_models_files:
        os.system(f"rm ./{models[wmf]}")


def serialize_results(real, pred, cfg, coin_val):
    data = np.array([real, pred], dtype='float32').T
    df = pd.DataFrame(data=data, columns=['Real', 'Pred'])
    dump_file = preds_filename(cfg)

    if coin_val:
        dump_file = f"{dump_file}_coin_valset_{coin_val}"

    df.to_csv(dump_file, index=False)
    print(f"Result[{dump_file}] dumped!")


def serialize(hist, cfg, i):
    dump_file = hist_filename(cfg, i)
    pd.DataFrame.from_dict(hist.history).to_csv(dump_file, index=False)
    print(f"Hist[{dump_file}] dumped!")


def dump_hist(cfg, run, train_score, val_score):
    hist = Object()
    hist.history = {'loss': train_score, 'val_loss': val_score}
    serialize(hist, cfg, run)


def get_best_model(data):
    data = sorted(data, key=lambda k: k['mse'])
    m = data[0]
    print(f'Best Model of all runs is: {m}')

    return m['model']


def load_models_data(cfg):
    models = []
    model_file_mask = f"{model_filename(cfg, None)}*"
    model_files = glob.glob(model_file_mask)
    model_files.sort()

    for model_file in model_files:
        with keras.utils.custom_object_scope(custom_layer_register()):
            model = tf.keras.models.load_model(model_file)
            print(f">>>  {model_file} Loaded !")
            models.append(model)

    return models


def load_model_data(model_file):
    with keras.utils.custom_object_scope(custom_layer_register()):
        model = tf.keras.models.load_model(model_file)
        print(f">>>  {model_file} Loaded !")

    return model


def load_baseline_models_data(cfg):
    models = []
    model_file_mask = f"{model_filename(cfg, None)}*"
    model_files = glob.glob(model_file_mask)
    model_files.sort()

    for model_file in model_files:
        model = load(model_file)
        print(f">>>  {model_file} Loaded !")
        models.append(model)

    return models


def score_mad(real, pred):
    dz_norm = (pred - real) / (1 + pred)
    return np.median(np.abs(dz_norm))


def do_scoring_over(d, cfg, models):
    print(f'Using device: {cfg.device_name}')
    with tf.device(cfg.device_name):
        all_scores = np.empty(shape=(0,3))
        model_data = np.array([])
        mses = np.array([])
        maes = np.array([])
        rmses = np.array([])
        r2s = np.array([])
        mads = np.array([])
        i = 0

        for model in models:
            scores = model.evaluate(d.x_test, d.y_test, verbose=10)
            print('Scores (loss, mse, mae) for run %d: %s' % (i, scores))
            all_scores = np.vstack((all_scores, scores))

            outputs = model.predict(d.x_test, batch_size=cfg.batch_size)
            mse = mean_squared_error(d.y_test, outputs)
            mae = mean_absolute_error(d.y_test, outputs)
            rmse = sqrt(mse)
            r2 = r2_score(d.y_test, outputs)
            mad = score_mad(d.y_test.to_numpy(), outputs.flatten())

            mses = np.append(mses, mse)
            maes = np.append(maes, mae)
            rmses = np.append(rmses, rmse)
            r2s = np.append(r2s, r2)
            mads = np.append(mads, mad)
            i = i + 1

            model_record = {
                'model': model,
                'mse': mse, 'r2': r2
            }

            model_data = np.append(model_data, model_record)

        print('================ ARGS USED ===================')
        print(cfg.args)
        print('================ RESULTS ===================')
        runs = len(models)
        print('Average over %d runs:' % runs)
        print(np.mean(all_scores, axis=0))
        print('Std deviation over %d runs:' % runs)
        print(np.std(all_scores, axis=0))

        print_scores(runs, cfg.learning_rate, mses, maes, rmses, r2s, mads)
        print_trace_scores(mses, maes, rmses, mads, r2s)

        return model_data


def do_baseline_scoring_over(d, cfg, models):
    mses = np.array([])
    maes = np.array([])
    rmses = np.array([])
    mads = np.array([])
    r2s = np.array([])
    model_data = np.array([])

    for model in models:
        outputs = model.predict(d.x_test)

        mse = mean_squared_error(d.y_test, outputs)
        mae = mean_absolute_error(d.y_test, outputs)
        rmse = sqrt(mse)
        mad = score_mad(d.y_test, outputs)
        r2 = r2_score(d.y_test, outputs)

        mses = np.append(mses, mse)
        maes = np.append(maes, mae)
        rmses = np.append(rmses, rmse)
        mads = np.append(mads, mad)
        r2s = np.append(r2s, r2)

        model_record = {
            'model': model,
            'mse': mse, 'r2': r2
        }

        model_data = np.append(model_data, model_record)

    print('================ ARGS USED ===================')
    print(cfg.args)
    print('================ RESULTS ===================')
    print_scores(cfg.num_runs, cfg.learning_rate, mses, maes, rmses, r2s, mads)
    print_trace_scores(mses, maes, rmses, mads, r2s)

    return model_data


def callbacks(cfg, run):
    model_dir = '.'
    best_model_filename = model_filename(cfg, run) + '_mse_{mse:.6f}_from_epoch_{epoch}.hdf5'
    patience = int(0.2 * cfg.epochs)
    best_weights_filepath = os.path.join(model_dir, best_model_filename)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience)

    mcp_save = ModelCheckpoint(
        monitor='val_loss', mode='min', filepath=best_weights_filepath, save_best_only=True
    )

    if cfg.no_early_stopping:
        print("early_stopping is disabled")
        callbacks = [mcp_save]
    else:
        callbacks = [early_stopping, mcp_save]

    return callbacks


def do_training_runs(d, cfg, verbose, customized_dropout):
    print(f'Using device: {cfg.device_name}')

    with tf.device(cfg.device_name):
        times = np.array([])
        tf.constant(cfg.df_ids_map, name='idmap')

        for i in range(cfg.num_runs):
            print('***Run #%d***' % i)
            start = time.perf_counter()
            model = neural_network(cfg, customized_dropout)
            hist = model.fit(d.x_train, d.y_train,
                            validation_data = (d.x_val, d.y_val),
                            epochs = cfg.epochs,
                            batch_size = cfg.batch_size,
                            verbose = verbose,
                            callbacks = callbacks(cfg, i))

            elapsed = time.perf_counter() - start
            times = np.append(times, elapsed)

            serialize(hist, cfg, i)
            clean_dir(cfg, i)

        print_times(times)

        return hist


def neural_network(cfg, dropout=None):
    model = keras.Sequential()

    if dropout:
        model.add(dropout)

    model.add(layers.Dense(cfg.l1_units, input_dim=cfg.D, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(cfg.l2_units, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(1, bias_initializer=initializers.Constant(cfg.bias_output_layer.to_numpy())))

    adam = tf.keras.optimizers.Adam(lr=cfg.learning_rate)
    model.compile(loss='mse',
                  optimizer=adam,
                  metrics=['mse', 'mae']
                  )

    return model


def print_times(times):
    print('--------------- Timing ----------------')
    mt = times.mean()
    stdt = times.std()
    tm = f"{mt}±{stdt}"
    print(f"Done in {tm} sec.")
    print(f"[CSV_t] {tm}")

    return tm


def print_trace_scores(mses,maes,rmses,mads,r2s):
    print('================ TRACE SCORES ===================')
    print(f"[CSV_mses] {mses}")
    print(f"[CSV_maes] {maes}")
    print(f"[CSV_rmses] {rmses}")
    print(f"[CSV_mads] {mads}")
    print(f"[CSV_r2s] {r2s}")


def print_scores(runs, lr, mses, maes, rmses, r2s, mads):
    p = '2f'
    m = 100
    print(f"Results after {runs} [lr: {lr}] x10¯²")
    print(f"MSE  {mses.mean()*m:.{p}}±{mses.std()*m:.{p}}")
    print(f"MAE  {maes.mean()*m:.{p}}±{maes.std()*m:.{p}}")
    print(f"RMSE {rmses.mean()*m:.{p}}±{rmses.std()*m:.{p}}")
    print(f"MAD {mads.mean() * m:.{p}}±{mads.std() * m:.{p}}")
    print(f"R2   {r2s.mean():.4f}±{r2s.std():.4f}")
    print('--------------------------------------------------------')
    print(f"[CSV] {mses.mean() * m:.{p}}±{mses.std() * m:.{p}},{maes.mean() * m:.{p}}±{maes.std() * m:.{p}},{rmses.mean() * m:.{p}}±{rmses.std() * m:.{p}},{mads.mean() * m:.{p}}±{mads.std() * m:.{p}},{r2s.mean():.4f}±{r2s.std():.4f}")


def xgbr_params(cfg):
    n_iter_no_change = int(0.2 * cfg.epochs)
    if cfg.no_early_stopping:
        n_iter_no_change = None

    rstate = random.randint(0, 1000)
    print(f"XGB at random_state: {rstate}")
    params = {'n_estimators': cfg.epochs,
              'max_depth': 8,
              'min_samples_split': 5,
              'validation_fraction': 0.2,
              'n_iter_no_change': n_iter_no_change,
              'learning_rate': cfg.learning_rate,
              'loss': 'ls',
              'random_state': rstate
              }

    return params


def do_xgbr_training_runs(d, cfg):
    times = np.array([])

    for i in range(cfg.num_runs):
        print('***Run #%d***' % i)
        start = time.perf_counter()

        params = xgbr_params(cfg)
        model = ensemble.GradientBoostingRegressor(**params)
        model.fit(d.x_train, d.y_train)

        elapsed = time.perf_counter() - start
        times = np.append(times, elapsed)

        # retrieve performance metrics
        train_score = model.train_score_
        val_score = np.zeros((params['n_estimators'],), dtype=np.float64)
        for k, y_pred in enumerate(model.staged_predict(d.x_val)):
            val_score[k] = model.loss_(d.y_val, y_pred)

        dump_hist(cfg, i, train_score, val_score)

        min_loss = min(train_score)
        trace_model_filename = f'{model_filename(cfg, i)}_mse_{min_loss:.6f}.joblib'
        dump(model, trace_model_filename)
        print(f"#### dump model [{trace_model_filename}]")

    print_times(times)


def do_lreg_training_runs(d, cfg):
    times = np.array([])

    for i in range(cfg.num_runs):
        print('***Run #%d***' % i)
        start = time.perf_counter()

        model = SGDRegressor(
            eta0=cfg.learning_rate, #learning rate bias
            max_iter=cfg.epochs,
            random_state=42
        )
        model.fit(d.x_train, d.y_train)

        elapsed = time.perf_counter() - start
        times = np.append(times, elapsed)

        trace_model_filename = f'{model_filename(cfg, i)}.joblib'
        dump(model, trace_model_filename)
        print(f"#### dump model [{trace_model_filename}]")

    print_times(times)


def apply_dt_over_test(df, df_test):
    print('# process_isotonic_regression in test dataframe')
    df_test = _process_dt_over_test(df, df_test, 'Err', '')

    return df_test

def _process_dt_over_test(df, df_test, e_sufix = '', e_prefix = ''):
    df_dt_err = df_test.copy(deep=True)
    idx = 10
    for b in 'ugriz':
        dt, _, _, _ = _apply_decision_tree_regression(df.copy(), b, e_prefix + b + e_sufix)
        pred = dt.predict(df_dt_err[[b]])
        df_dt_err.insert(idx, f"{b}ErrExp", pred, allow_duplicates=True)
        idx = idx + 1

    return df_dt_err

def apply_ir_over_test(df, df_test):
    print('# process_isotonic_regression in test dataframe')
    df_test = _process_ir_over_test(df, df_test, 'Err', '')

    return df_test

def _process_ir_over_test(df, df_test, e_sufix = '', e_prefix = ''):
    df_ir_err = df_test.copy(deep=True)
    idx = 10
    for b in 'ugriz':
        eb = e_prefix + b + e_sufix
        ir, _, _, _ = _apply_isotonic_regression(df.copy(), b, eb)
        pred = ir.predict(df_ir_err[b])
        df_ir_err.insert(idx, f"{b}ErrExp", pred, allow_duplicates=True)
        idx = idx + 1

    return df_ir_err


def apply_isotonic_regression(df, dataset_name, do_in_ugriz):
    print('# process_isotonic_regression in dataframe')
    if dataset_name == 'kaggle_bkp':
        df = _process_isotonic_regression(df, '', 'modelmagerr_', do_in_ugriz=do_in_ugriz)
    elif dataset_name == 'sdss':
        df = _process_isotonic_regression(df, '', 'err_', do_in_ugriz=do_in_ugriz)
    else:
        df = _process_isotonic_regression(df, 'Err', '', do_in_ugriz=do_in_ugriz)

    return df


def _process_isotonic_regression(df, e_sufix = '', e_prefix = '', do_in_ugriz=False):
    df_ir_err = df.copy(deep=True)
    idx = 10
    for b in 'ugriz':
        eb = e_prefix + b + e_sufix
        ir, _, _, _ = _apply_isotonic_regression(df.copy(), b, eb)
        pred = ir.predict(df_ir_err[b])
        df_ir_err.insert(idx, f"{b}ErrExp", pred, allow_duplicates=True)
        idx = idx + 1

    if do_in_ugriz:
        for b in 'ugriz':
            eb = f"{b}ErrExp"
            ir, _, _, _ = _apply_isotonic_regression(df_ir_err.copy(), eb, b)
            pred = ir.predict(df_ir_err[eb])
            df_ir_err.insert(idx, f"{b}Exp", pred, allow_duplicates=True)
            idx = idx + 1

    return df_ir_err


def _apply_isotonic_regression(df, mag, magErr):
  df.sort_values(by=[mag], inplace=True)
  df = df.reset_index(drop=True)
  x = df[mag]
  y = df[magErr]
  ir = IsotonicRegression()
  y_expected = ir.fit_transform(x, y)

  return ir, x, y, y_expected


def apply_decision_tree_regression(df, dataset_name):
    print('# process_decision_tree_regression in dataframe')
    if dataset_name == 'kaggle_bkp':
        df = _process_decision_tree_regression(df, '', 'modelmagerr_')
    elif dataset_name == 'sdss':
        df = _process_decision_tree_regression(df, '', 'err_')
    else:
        df = _process_decision_tree_regression(df, 'Err', '')

    return df


def _process_decision_tree_regression(df, e_sufix = '', e_prefix = ''):
    df_dt_err = df.copy()
    idx = 10
    for b in 'ugriz':
        dt, _, _, _ = _apply_decision_tree_regression(df_dt_err, b, e_prefix + b + e_sufix)
        pred = dt.predict(df_dt_err[[b]])
        df_dt_err.insert(idx, f"{b}ErrExp", pred, allow_duplicates=True)
        idx = idx + 1

    return df_dt_err


def _apply_decision_tree_regression(df, magCol, errCol, max_depth=5):
    X = df[[magCol]]
    y = df[[errCol]]
    regressor = DecisionTreeRegressor(random_state=0, max_depth = max_depth)
    y_expected =regressor.fit(X, y)

    return regressor, X, y, y_expected
