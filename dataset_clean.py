import argparse
import pandas as pd


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script for dataset cleaning.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')
    parser.add_argument('-rn', action='store_true', help='Remove negative entries.')
    parser.add_argument('-u25', action='store_true', help='Cut >= 25 in u band.')
    parser.add_argument('-e1', action='store_true', help='Cut >= 1.0 in all band errors.')
    parser.add_argument('-rf', action='store_true', help='Cut clean flag data = 0.')

    return parse


def filter_negative_data(dataframe):
    orig_size = dataframe.shape[0]

    for b in 'ugriz':
        dataframe = dataframe[dataframe[f"{b}"] > 0]

    clean_size = dataframe.shape[0]
    print(f"Negative data removed: {orig_size - clean_size}.")

    return dataframe


def cut_all_val_errs(df, val):
    for b in 'ugriz':
        df = df[df[f"{b}"] <= val]

    return df


def cut_val_band(df, band, val):
    return df[df[band] <= val]


def cut_flag(df, attr, value):
    chunk_size = df[df[attr] == value].shape[0]
    print(f"{chunk_size} values cut off.")

    return df[df[attr] != value]


def clean_data(dataset_name, rm_negatives, cut_u_25, cut_1_errs, cut_clean_flag):
    data = pd.read_csv(dataset_name, comment='#',
                       delim_whitespace=True,
                       names=[
                           'ID', 'u', 'g', 'r', 'i', 'z',
                           'uErr', 'gErr', 'rErr', 'iErr', 'zErr',
                           'redshift', 'redshiftErr'
                       ])

    if rm_negatives:
        data = filter_negative_data(data)

    if cut_u_25:
        data = cut_val_band(data, 'u', 25.0)

    if cut_1_errs:
        data = cut_all_val_errs(data, 1.0)

    if cut_clean_flag:
        data = cut_flag(data, 'clean', 0)

    data.to_csv(f"{dataset_name}_clean", index=False)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    clean_data(*args)
