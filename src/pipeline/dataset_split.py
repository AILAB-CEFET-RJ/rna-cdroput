import argparse
import pandas as pd

from src.modules import utils

from sklearn.model_selection import train_test_split

def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script for dataset splitting.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')
    parse.add_argument('-p', metavar='PROPORTION', help='Split proportion for train:test:val | train:val percentages.')

    return parse


def split_data(dataset_name, proportion):
    data = pd.read_csv(f"./src/data/{dataset_name}", comment='#')
    name, ext = dataset_name.split('.')

    props = proportion.split(':')

    orig_size = data.shape[0]

    if len(props) == 2:
        s = int(props[1]) / 100
        train_data, val_data = train_test_split(data, test_size=s, random_state=42)

        train_data.to_csv(f"./src/data/{name}_train.{ext}", index=False)
        val_data.to_csv(f"./src/data/{name}_val.{ext}", index=False)

        train_size = train_data.shape[0]
        val_size = val_data.shape[0]
        print(f"Train: {train_size} of {orig_size}")
        print(f"Val: {val_size} of {orig_size}")

    else:
        s0 = int(props[1]) / 100
        s1 = int(props[2]) / 100
        s = s0 + s1
        s2 = s1 / s

        train_data, test_data = train_test_split(data, test_size=s, random_state=42)
        test_data, val_data = train_test_split(test_data, test_size=s2, random_state=42)

        train_data.to_csv(f"./src/data/{name}_train.{ext}", index=False)
        test_data.to_csv(f"./src/data/{name}_test.{ext}", index=False)
        val_data.to_csv(f"./src/data/{name}_val.{ext}", index=False)

        train_size = train_data.shape[0]
        test_size = test_data.shape[0]
        val_size = val_data.shape[0]

        utils.rna_cdropout_print(f"Train: {train_size} of {orig_size}")
        utils.rna_cdropout_print(f"Test: {test_size} of {orig_size}")
        utils.rna_cdropout_print(f"Val: {val_size} of {orig_size}")


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    utils.rna_cdropout_print(f"Stage 03: Splitting {args.dataset}")
    split_data(args.dataset, args.p)
