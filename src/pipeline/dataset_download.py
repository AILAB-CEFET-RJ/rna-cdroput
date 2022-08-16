import argparse
import pandas as pd

from urllib import request

from src.modules import utils

def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script for dataset downloads. Formats all data to csv.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset to download [teddy|happy|sdss|all].')

    return parse

def download_teddy():
    for data_chunk in 'ABCD':
        dataset_download_url = f"https://raw.githubusercontent.com/COINtoolbox/photoz_catalogues/master/Teddy/forTemplateBased/teddyT_{data_chunk}.cat"

        dataset_download_path = f"./src/data/teddyT_{data_chunk}.cat"

        request.urlretrieve(dataset_download_url, dataset_download_path)

        data = pd.read_csv(f"./src/data/teddyT_{data_chunk}.cat", comment='#',
                           delim_whitespace=True,
                           names=[
                               'objid', 'u', 'g', 'r', 'i', 'z',
                               'err_u', 'err_g', 'err_r', 'err_i', 'err_z',
                               'redshift', 'err_redshift'
                           ])

        if data_chunk == 'A':
            data.to_csv('./src/data/teddy_data.csv', index=False)
        else:
            data.drop(columns=['err_redshift'], axis=1, inplace=True)
            data.to_csv(f"./src/data/teddy_test_data_{data_chunk}.csv", index=False)


def download_happy():
    for data_chunk in 'ABCD':
        dataset_download_url = f"https://raw.githubusercontent.com/COINtoolbox/photoz_catalogues/master/Happy/forTemplateBased/happyT_{data_chunk}"

        dataset_download_path = f"./src/data/happyT_{data_chunk}.cat"

        request.urlretrieve(dataset_download_url, dataset_download_path)

        data = pd.read_csv(f"./src/data/happyT_{data_chunk}.cat", comment='#',
                           delim_whitespace=True,
                           names=[
                               'objid', 'u', 'g', 'r', 'i', 'z',
                               'err_u', 'err_g', 'err_r', 'err_i', 'err_z',
                               'redshift', 'err_redshift'
                           ])

        if data_chunk == 'A':
            data.to_csv('./src/data/happy_data.csv', index=False)
        else:
            data.drop(columns=['err_redshift'], axis=1, inplace=True)
            data.to_csv(f"./src/data/happy_test_data_{data_chunk}.csv", index=False)

def download_sdss():
    dataset_download_url = f"https://zenodo.org/record/4752020/files/sdss_train_data.csv?download=1"

    dataset_download_path = './src/data/sdss_data.csv'

    request.urlretrieve(dataset_download_url, dataset_download_path)

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

    utils.rna_cdropout_print(f"Stage 01: Downloading ({dataset_name})")
    download_data(dataset_name)
