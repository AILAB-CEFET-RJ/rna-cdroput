import argparse
import pandas as pd


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script for dataset sampling.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')
    parse.add_argument('-s', metavar='SIZE', type=int, help='Subsample size.')

    return parse


def sample_data(dataset_name, size):
    data = pd.read_csv("./src/data/{dataset_name}", comment='#')

    subs_df = data.sample(n=size, random_state=42)
    print(f"Using subsample {subs_df.shape[0]} of {data.shape[0]}.")
    data = subs_df

    name, ext = dataset_name.split('.')
    data.to_csv(f"./src/data/{name}_resampled.{ext}", index=False)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    sample_data(args.dataset, args.s)
