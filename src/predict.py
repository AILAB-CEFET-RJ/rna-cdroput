import os
import argparse

import tensorflow as tf
import pandas as pd

from src.modules import utils

from src.modules.regularization import ErrorBasedInvertedDropoutV2


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments')
    parse.add_argument('-n', metavar='NAME', help='Experiment name. This name is included in output files.')
    parse.add_argument('-models', nargs='+', metavar='MODELS', help='Model filenames.')
    parse.add_argument('-dp', metavar='DROPOUT', help='Dropout class to use.')
    parse.add_argument('-testset', metavar='TESTSET', help='Test dataset file.')
    parse.add_argument('-bs', metavar='BATCH', type=int, default=32, help='Batch size.')

    return parse


def select_dropout(dropout_opt):
    if dropout_opt:
        return ErrorBasedInvertedDropoutV2()
    return None


def custom_layer_register():
    return {
        'ErrorBasedInvertedDropoutV2': ErrorBasedInvertedDropoutV2
    }


def load_model(file):
    with tf.keras.utils.custom_object_scope(custom_layer_register()):
        m = tf.keras.models.load_model(file)
        utils.rna_cdropout_print(f"{file} Loaded !")

        return m


def serialize_results(real, preds, mfile, name):
    df = pd.DataFrame()
    df['Real'] = real
    df['Pred'] = preds.flatten()

    file_name = mfile.split('/')[-1].split('.hdf5')[0]

    dump_file = f"real_x_pred_{file_name}.csv"
    if name:
        dump_file = f"real_x_pred_{name}_{file_name}.csv"

    epochs = mfile.split('/')[2]
    run = mfile.split('/')[3]
    dump_dir = f"./output/preds/{epochs}/{run}/"
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    filepath = os.path.join(dump_dir, dump_file)

    df.to_csv(filepath, index=False)

    utils.rna_cdropout_print(f"Result[{dump_file}] dumped!")


def predict(model_files, testset, batch_size, dropout, name):
    for model_file in model_files:
        model = load_model(model_file)
        utils.rna_cdropout_print(f"Using batch size: {batch_size}")
        test_df = pd.read_csv(f"./src/data/{testset}", comment='#')
        utils.rna_cdropout_print(f"Train set loaded! Shape = {test_df.shape}")

        ugriz = list('ugriz')
        errors = list(map(lambda b: f"err_{b}", ugriz))
        exp_errors = list(map(lambda eb: f"{eb}_exp", errors))
        target = ['redshift']

        features = ugriz + errors
        if dropout:
            features = features + exp_errors

        x_test = test_df[features]
        y_test = test_df[target]

        outputs = model.predict(x_test, batch_size=batch_size)
        serialize_results(y_test, outputs, model_file, name)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    predict(args.models, args.testset, args.bs, args.dp, args.n)
