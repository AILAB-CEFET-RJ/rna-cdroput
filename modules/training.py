import numpy as np
import pandas as pd

from math import sqrt

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import ensemble

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def serialize_results(real, pred, cfg):
    data = np.array([real, pred], dtype='float32').T
    df = pd.DataFrame(data=data, columns=['Real', 'Pred'])
    dump_file = f"real_x_pred_{cfg.args.dp}_{cfg.args.sc}_{cfg.args.dataset}"

    if cfg.args.xgbr:
        dump_file = "XGBR_" + dump_file

    df.to_csv(dump_file, index=False)
    print(f"Result[{dump_file}] dumped!")


def serialize(hist, cfg, i):
    dump_file = f"hist_run_{i}_{cfg.args.dp}_{cfg.args.sc}_{cfg.args.dataset}"
    pd.DataFrame.from_dict(hist.history).to_csv(dump_file, index=False)
    print(f"Hist[{dump_file}] dumped!")


def do_training_runs(d, cfg, verbose=0, customized_dropout=None):
    print(f'Using device: {cfg.device_name}')
    with tf.device(cfg.device_name):
        all_scores = np.empty(shape=(0,3))
        mses = np.array([])
        maes = np.array([])
        rmses = np.array([])
        r2s = np.array([])

        for i in range(cfg.num_runs):
            print('***Run #%d***' % i)
            model = neural_network(cfg, customized_dropout)
            hist = model.fit(d.x_train, d.y_train,
                            validation_data = (d.x_val, d.y_val),
                            epochs = cfg.epochs,
                            verbose = verbose,
                            callbacks = cfg.callbacks)
            scores = model.evaluate(d.x_test, d.y_test, verbose=0)
            print('Scores (loss, mse, mae) for run %d: %s' % (i, scores))
            all_scores = np.vstack((all_scores, scores))

            outputs = model.predict(d.x_test)
            mse = mean_squared_error(d.y_test, outputs)
            mae = mean_absolute_error(d.y_test, outputs)
            rmse = sqrt(mse)
            r2 = r2_score(d.y_test, outputs)

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

        return model, hist, all_scores


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


def process_isotonic_regression(df, feature_num):
    df_ir_err = df.copy(deep=True)
    idx = 10
    for b in 'ugriz':
        ir, _, _, _ = apply_isotonic_regression(df.copy(), b, b + 'Err')
        pred = ir.predict(df_ir_err[b])
        df_ir_err.insert(idx, f"{b}ErrExp", pred, allow_duplicates=True)
        idx = idx + 1

    return df_ir_err


def apply_isotonic_regression(df, mag, magErr):
  df.sort_values(by=[mag], inplace=True)
  df = df.reset_index(drop=True)
  x = df[mag]
  y = df[magErr]
  ir = IsotonicRegression()
  y_expected = ir.fit_transform(x, y)

  return ir, x, y, y_expected
