import os
import argparse
import math
import time
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from numpy.random import seed

from src.modules import utils
from src.modules.regularization import ErrorBasedInvertedDropoutV2



def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments')
    parse.add_argument('-n', metavar='NAME', help='Experiment name. This name is included in output files.')
    parse.add_argument('-e', metavar='EPOCHS', type=int, help='Epochs.')
    parse.add_argument('-dp', metavar='DROPOUT', help='Dropout class to use.')
    parse.add_argument('-runs', metavar='RUNS', type=int, help='Total runs.')
    parse.add_argument('-lr', metavar='LR', type=float, help='Learning rate.')
    parse.add_argument('-trainset', metavar='TRAINSET', help='Train dataset file.')
    parse.add_argument('-valset', metavar='VALSET', help='Validation dataset file.')
    parse.add_argument('-noes', action='store_true', help='Disable early stop.')
    parse.add_argument('-gpu', metavar='DEVICE', help='GPU device name. Default is device name position 0.')
    parse.add_argument('-feature', metavar='FEAT', help='Feature options: [B | A]. B for bands. A for bands and errors. Default is A.')
    parse.add_argument('-bs', metavar='BATCH', type=int, default=0, help='Batch size.')
    parse.add_argument('-layers', metavar='LAYERS', help='Force amount of units in each hidden layer. '
                                                         'Use "20:10" value for 2 hidden layers with 20 neurons in first and 10 in seccond. The Default is 2 hidden layers'
                                                         ' and neurons are computed based on the size of the features.')

    return parse


def select_dropout(dropout_opt):
    if dropout_opt:
        return ErrorBasedInvertedDropoutV2()
    return None


def callbacks(no_early_stopping, epochs, modelname, run):
    model_dir = f"./output/models/epochs_{epochs}/run_{run}/"
    model_filename = f"model_{modelname}" + '_mse_{mse:.6f}_from_epoch_{epoch}.hdf5'
    patience = int(0.2 * epochs)
    weights_filepath = os.path.join(model_dir, model_filename)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=patience)

    mcp_save = ModelCheckpoint(
        monitor='val_loss', mode='min', filepath=weights_filepath, save_best_only=True
    )

    if no_early_stopping:
        utils.rna_cdrpout_print("early_stopping is disabled")
        callbacks = [mcp_save]
    else:
        callbacks = [early_stopping, mcp_save]

    return callbacks


def neural_network(dropout, p_layers, learning_rate, n_features, bias_output_layer):
    model = keras.Sequential()

    if dropout:
        model.add(dropout)

    lu = p_layers.split(':')

    model.add(layers.Dense(lu[0], input_dim=n_features, kernel_initializer='normal', activation='relu'))
    for l in lu[1:]:
        model.add(layers.Dense(l, kernel_initializer='normal', activation='relu'))

    model.add(layers.Dense(1, bias_initializer=initializers.Constant(bias_output_layer)))

    #FIXME only works default adam
    adam = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    return model


def serialize(hist, modelname, epochs, run):
    dump_dir = f"./output/hists/epochs_{epochs}/"
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    dump_file = f"hist_{modelname}_run_{run}"
    filepath = os.path.join(dump_dir, dump_file)
    pd.DataFrame.from_dict(hist.history).to_csv(filepath, index=False)
    utils.rna_cdrpout_print(f"Hist[{dump_file}] dumped!")


def print_times(times):
    utils.rna_cdrpout_print('--------------- Timing ----------------')
    mt = times.mean()
    stdt = times.std()
    tm = f"{mt}±{stdt}"
    utils.rna_cdrpout_print(f"Done in {tm} sec.")
    utils.rna_cdrpout_print(f"[CSV_t] {tm}")

    return tm


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    utils.rna_cdrpout_print("Stage 05: Training")

    ugriz = list('ugriz')
    errors = list(map(lambda b: f"err_{b}", ugriz))
    exp_errors = list(map(lambda eb: f"{eb}_exp", errors))
    target = ['redshift']

    features = ugriz+errors

    dropout = select_dropout(args.dp)
    utils.rna_cdrpout_print(f"Selected Dropout: {args.dp}")

    if dropout:
        features = features + exp_errors

    f = len(features)

    train_df = pd.read_csv(f"./src/data/{args.trainset}", comment='#')
    utils.rna_cdrpout_print(f"Train set loaded! Shape = {train_df.shape}")
    val_df = pd.read_csv(f"./src/data/{args.valset}", comment='#')
    utils.rna_cdrpout_print(f"Validation set loaded! Shape = {val_df.shape}")

    x_train = train_df[features]
    y_train = train_df[target]
    x_val = train_df[features]
    y_val = train_df[target]

    if args.layers is None:
        m = 1
        N = train_df.shape[0]
        l1 = round(math.sqrt((m + 2) * N) + 2 * math.sqrt(N / (m + 2)))
        l2 = round(m * math.sqrt(N / (m + 2)))
        args.layers = f"{l1}:{l2}"
        utils.rna_cdrpout_print(f"Using layers= {args.layers}")

    device_name = tf.test.gpu_device_name()
    if args.gpu:
        device_name = args.gpu
    utils.rna_cdrpout_print(f"Device Name to use: '{device_name}'")

    batch_size = 32
    if args.bs:
        batch_size = args.bs
    utils.rna_cdrpout_print(f"Using batch size: {batch_size}")

    seed(42)
    tf.random.set_seed(42)

    with tf.device(device_name):
        times = np.array([])

        for i in range(args.runs):
            utils.rna_cdrpout_print(f"*** Run {i} ***")
            start = time.perf_counter()
            model = neural_network(dropout, args.layers, args.lr, f, y_train.mean().to_numpy())
            #model.summary()
            hist = model.fit(x_train, y_train,
                             validation_data=(x_val, y_val),
                             epochs=args.e,
                             batch_size=batch_size,
                             verbose=0,
                             callbacks=callbacks(args.noes, args.e, args.n, i)
                             )

            elapsed = time.perf_counter() - start
            times = np.append(times, elapsed)

            serialize(hist, args.n, args.e, i)

        print_times(times)
