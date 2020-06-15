import numpy as np
import pandas as pd

from math import sqrt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
        print(f"MSE  {mses.mean():.3f}±{mses.std():.3f}")
        print(f"MAE  {maes.mean():.3f}±{maes.std():.3f}")
        print(f"RMSE {rmses.mean():.3f}±{rmses.std():.3f}")
        print(f"R2   {r2s.mean():.3f}±{r2s.std():.3f}")

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
