import glob
import argparse
import numpy as np
import pandas as pd

from modules import dataset_handle as dh
from matplotlib import cm
from matplotlib.colors import ListedColormap

from matplotlib.ticker import NullFormatter

def parser():
   parser = argparse.ArgumentParser(description='Reports module')
   parser.add_argument('-dataset', metavar='DATASET', help='Dataset file.')
   parser.add_argument('-ref_file', metavar='REF', help='Result ref id file.')
   parser.add_argument('-output', metavar='OUT', help='Result file.')

   return parser


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    dh.download_data(args.dataset, None)
    df, _ = dh.load_dataframe(args.dataset, None)

    df_ref_id = pd.read_csv(args.ref_file)

    list = []
    with open(args.output, 'r') as file:
        preds_found = False
        for line in file:
            if '##PREDS' in line:
                preds_found = True
                continue
            if 'Result[real_' in line:
                preds_found = False

            if preds_found:
                v = line.replace('id: [', '')\
                    .replace('[', '')\
                    .replace('\n', '')\
                    .replace(']', '').strip().split(' ')
                v = [int(i) for i in v]
                list.append(v)

    print(list)



