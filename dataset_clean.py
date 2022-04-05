import argparse
import pandas as pd


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script for dataset cleaning.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')
    parse.add_argument('-rn', action='store_true', help='Remove negative entries.')
    parse.add_argument('-u25', action='store_true', help='Cut >= 25 in u band.')
    parse.add_argument('-e1', action='store_true', help='Cut >= 1.0 in all band errors.')
    parse.add_argument('-rf', action='store_true', help='Cut clean flag data = 0.')

    return parse


def filter_negative_data(dataframe):
    orig_size = dataframe.shape[0]

    for b in 'ugriz':
        dataframe = dataframe[dataframe[f"{b}"] > 0]

    clean_size = dataframe.shape[0]
    print(f"Negative data removed: {orig_size - clean_size}.")

    return dataframe


def cut_all_val_errs(df, val):
    orig_size = df.shape[0]

    for b in 'ugriz':
        df = df[df[f"{b}"] <= val]

    clean_size = df.shape[0]
    print(f"Errors values cut off: {orig_size - clean_size}.")

    return df


def cut_val_band(df, band, val):
    orig_size = df.shape[0]

    df = df[df[band] <= val]

    clean_size = df.shape[0]
    print(f"Attr in {band} band cut: {orig_size - clean_size}.")

    return df


def cut_flag(df, attr, value):
    orig_size = df.shape[0]

    df = df[df[attr] != value]

    clean_size = df.shape[0]
    print(f"Flag values cut off: {orig_size - clean_size}.")

    return df


def clean_data(dataset_name, rm_negatives, cut_u_25, cut_1_errs, cut_clean_flag):
    data = pd.read_csv(dataset_name, comment='#')

    data_orig_size = data.shape[0]

    if rm_negatives:
        data = filter_negative_data(data)

    if cut_u_25:
        data = cut_val_band(data, 'u', 25.0)

    if cut_1_errs:
        data = cut_all_val_errs(data, 1.0)

    if cut_clean_flag:
        data = cut_flag(data, 'clean', 0)

    data_clean_size = data.shape[0]
    print(f"Total data removed: {data_orig_size - data_clean_size}.")

    name, ext = dataset_name.split('.')
    data.to_csv(f"{name}_clean.{ext}", index=False)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    clean_data(args.dataset, args.rn, args.u25, args.e1, args.rf)
