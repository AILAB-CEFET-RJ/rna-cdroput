import argparse
import os
import gdown
import pandas as pd


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script for dataset downloads. Formats all data to csv.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset to download [teddy|happy|sdss|all].')

    return parse


def download_teddy():
    for data_chunk in 'ABCD':
        os.system(f"wget https://raw.githubusercontent.com/COINtoolbox/photoz_catalogues/master/Teddy/forTemplateBased/teddyT_{data_chunk}.cat")

        data = pd.read_csv(f"teddyT_{data_chunk}.cat", comment='#',
                           delim_whitespace=True,
                           names=[
                               'objid', 'u', 'g', 'r', 'i', 'z',
                               'err_u', 'err_g', 'err_r', 'err_i', 'err_z',
                               'redshift', 'err_redshift'
                           ])

        if data_chunk == 'A':
            data.to_csv('teddy_data.csv', index=False)
        else:
            data.to_csv(f"teddy_test_data_{data_chunk}.csv", index=False)

        os.system(f"rm teddyT_{data_chunk}.cat")


def download_happy():
    for data_chunk in 'ABCD':
        os.system(f"wget https://raw.githubusercontent.com/COINtoolbox/photoz_catalogues/master/Happy/forTemplateBased/happyT_{data_chunk}")

        data = pd.read_csv(f"happyT_{data_chunk}", comment='#',
                           delim_whitespace=True,
                           names=[
                               'objid', 'u', 'g', 'r', 'i', 'z',
                               'err_u', 'err_g', 'err_r', 'err_i', 'err_z',
                               'redshift', 'err_redshift'
                           ])

        if data_chunk == 'A':
            data.to_csv('happy_data.csv', index=False)
        else:
            data.to_csv(f"happy_test_data_{data_chunk}.csv", index=False)

        os.system(f"rm happyT_{data_chunk}")


def download_sdss():
    url = f"https://zenodo.org/record/4752020/files/sdss_train_data.csv?download=1"
    output = 'sdss_data.csv'
    gdown.download(url, output, quiet=False)


def download_data(dataset_name):
    if dataset_name == 'teddy':
        download_teddy()

    if dataset_name == 'happy':
        download_happy()

    if dataset_name == 'sdss':
        download_sdss()

    if dataset_name == 'all' or dataset_name is None:
        download_teddy()
        download_happy()
        download_sdss()


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    dataset_name = args.dataset

    download_data(dataset_name)
