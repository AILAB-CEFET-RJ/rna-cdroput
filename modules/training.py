import numpy as np
import pandas as pd
import os

from math import sqrt
from time import sleep

from sklearn.isotonic import IsotonicRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import ensemble

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from modules.regularization import ErrorBasedDropoutIR
from modules.regularization import ErrorBasedDropoutDT
from modules.regularization import ErrorBasedDropoutZero


MODEL_FILE = 'model_weights.hdf5'


def custom_layer_register():
    return {
        "ErrorBasedDropoutIR": ErrorBasedDropoutIR,
        "ErrorBasedDropoutDT": ErrorBasedDropoutDT,
        "ErrorBasedDropoutZero": ErrorBasedDropoutZero
    }


def serialize_results(real, pred, cfg):
    data = np.array([real, pred], dtype='float32').T
    df = pd.DataFrame(data=data, columns=['Real', 'Pred'])
    dump_file = f"real_x_pred_{cfg.args.dp}_{cfg.args.sc}_{cfg.args.dataset}"

    if cfg.args.xgbr:
        dump_file = "XGBR" + dump_file

    df.to_csv(dump_file, index=False)
    print(f"Result[{dump_file}] dumped!")


def serialize(hist, cfg, i):
    dump_file = f"hist_{cfg.args.dp}_{cfg.args.sc}_{cfg.args.dataset}_run_{i}"
    pd.DataFrame.from_dict(hist.history).to_csv(dump_file, index=False)
    print(f"Hist[{dump_file}] dumped!")


def get_best_model(data):
    print(f'Models: {data}')
    data = sorted(data, key=lambda k: k['mse'])
    m = data[0]
    print(f'Model file load: {m}')

    with keras.utils.custom_object_scope(custom_layer_register()):
        return tf.keras.models.load_model(m['file'])


def wait_model_dump():
    print('... dump model wait')
    tries = 0
    while not os.path.exists(MODEL_FILE):
        if tries == 10:
            print('Give up !! Proceed with current model !!')
            print('Epoch generates worst model !!')
            return False
        print('.', end='')
        sleep(1)
        tries = tries + 1
    print('dump model done!')
    return True


def do_training_runs(d, cfg, verbose=0, customized_dropout=None):
    print(f'Using device: {cfg.device_name}')
    with tf.device(cfg.device_name):
        all_scores = np.empty(shape=(0,3))
        mses = np.array([])
        maes = np.array([])
        rmses = np.array([])
        r2s = np.array([])
        model_data = np.array([])

        for i in range(cfg.num_runs):
            print('***Run #%d***' % i)
            model = neural_network(cfg, customized_dropout)
            hist = model.fit(d.x_train, d.y_train,
                            validation_data = (d.x_val, d.y_val),
                            epochs = cfg.epochs,
                            verbose = verbose,
                            callbacks = cfg.callbacks)

            dumped = wait_model_dump()

            if dumped:
                with keras.utils.custom_object_scope(custom_layer_register()):
                    best_model = tf.keras.models.load_model(MODEL_FILE)

                model = best_model
                best_model_filename = f'model_weights_{cfg.args.dataset}_{cfg.args.sc}_{cfg.args.dp}_{cfg.epochs}_run_{i}.hdf5'
                cmd = f'mv {MODEL_FILE} {best_model_filename}'
                os.system(cmd)

            scores = model.evaluate(d.x_test, d.y_test, verbose=0)
            print('Scores (loss, mse, mae) for run %d: %s' % (i, scores))
            all_scores = np.vstack((all_scores, scores))

            outputs = model.predict(d.x_test)
            mse = mean_squared_error(d.y_test, outputs)
            mae = mean_absolute_error(d.y_test, outputs)
            rmse = sqrt(mse)
            r2 = r2_score(d.y_test, outputs)

            if dumped:
                model_record = {
                    'file': best_model_filename,
                    'mse': mse, 'r2': r2
                }
                model_data = np.append(model_data, model_record)

            mses = np.append(mses, mse)
            maes = np.append(maes, mae)
            rmses = np.append(rmses, rmse)
            r2s = np.append(r2s, r2)
            serialize(hist, cfg, i)

        print('================ ARGS USED ===================')
        print(cfg.args)
        print('================ RESULTS ===================')
        print('Average over %d runs:' % cfg.num_runs)
        print(np.mean(all_scores, axis=0))
        print('Std deviation over %d runs:' % cfg.num_runs)
        print(np.std(all_scores, axis=0))

        print(f"Results after {cfg.num_runs} [lr: {cfg.learning_rate}]")
        print(f"MSE  {mses.mean():.4f}±{mses.std():.4f}")
        print(f"MAE  {maes.mean():.4f}±{maes.std():.4f}")
        print(f"RMSE {rmses.mean():.4f}±{rmses.std():.4f}")
        print(f"R2   {r2s.mean():.4f}±{r2s.std():.4f}")

        all_runs_best_model = get_best_model(model_data)
        return all_runs_best_model, hist, all_scores


def neural_network(cfg, dropout=None):
    model = keras.Sequential()

    if dropout:
        model.add(dropout)

    model.add(layers.Dense(cfg.l1_units, input_dim=cfg.D, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(cfg.l2_units, kernel_initializer='normal', activation='relu'))
    model.add(layers.Dense(1))

    adam = tf.keras.optimizers.Adam(lr=cfg.learning_rate)
    model.compile(loss='mse',
                  optimizer=adam,
                  metrics=['mse', 'mae']
                  )

    return model


def runGradientBoost(x_train, y_train, params):
  reg = ensemble.GradientBoostingRegressor(**params)
  reg.fit(x_train, y_train)

  return reg


def do_xgbr_training_runs(d, cfg, params):
    print(f'Using device: {cfg.device_name}')
    with tf.device(cfg.device_name):
        mses = np.array([])
        maes = np.array([])
        rmses = np.array([])
        r2s = np.array([])

        for i in range(cfg.num_runs):
            print('***Run #%d***' % i)
            model = runGradientBoost(d.x_train, d.y_train, params)
            outputs = model.predict(d.x_test)

            mse = mean_squared_error(d.y_test, outputs)
            mae = mean_absolute_error(d.y_test, outputs)
            rmse = sqrt(mse)
            r2 = r2_score(d.y_test, outputs)

            mses = np.append(mses, mse)
            maes = np.append(maes, mae)
            rmses = np.append(rmses, rmse)
            r2s = np.append(r2s, r2)

        print('================ ARGS USED ===================')
        print(cfg.args)
        print('================ RESULTS ===================')
        print(f"Results after {cfg.num_runs} [lr: {cfg.learning_rate}]")
        print(f"MSE  {mses.mean():.4f}±{mses.std():.4f}")
        print(f"MAE  {maes.mean():.4f}±{maes.std():.4f}")
        print(f"RMSE {rmses.mean():.4f}±{rmses.std():.4f}")
        print(f"R2   {r2s.mean():.4f}±{r2s.std():.4f}")

        return model


def apply_isotonic_regression(df, dataset_name):
    print('# process_isotonic_regression in dataframe')
    if dataset_name == 'kaggle_bkp':
        df = _process_isotonic_regression(df, '', 'modelmagerr_')
    elif dataset_name == 'sdss':
        df = _process_isotonic_regression(df, '', 'err_')
    else:
        df = _process_isotonic_regression(df, 'Err', '')

    return df


def _process_isotonic_regression(df, e_sufix = '', e_prefix = ''):
    df_ir_err = df.copy(deep=True)
    idx = 10
    for b in 'ugriz':
        ir, _, _, _ = _apply_isotonic_regression(df.copy(), b, e_prefix + b + e_sufix)
        pred = ir.predict(df_ir_err[b])
        df_ir_err.insert(idx, f"{b}ErrExp", pred, allow_duplicates=True)
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
        ir, _, _, y_ = _apply_decision_tree_regression(df_dt_err, b, e_prefix + b + e_sufix)
        df_dt_err.insert(idx, f"{b}ErrExp", y_, allow_duplicates=True)
        idx = idx + 1

    return df_dt_err


def _apply_decision_tree_regression(df, magCol, errCol, max_depth=5):
    X = df[[magCol]]
    y = df[[errCol]]
    regressor = DecisionTreeRegressor(random_state=0, max_depth = max_depth)
    regressor.fit(X, y)
    y_expected = regressor.predict(X)

    return regressor, X, y, y_expected
