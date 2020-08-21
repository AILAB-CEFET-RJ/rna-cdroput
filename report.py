import os
import argparse
import numpy as np
import pandas as pd


def parser():
   parser = argparse.ArgumentParser(description='Reports module')
   parser.add_argument('-dir', metavar='DIR', help='Result file directory.')
   parser.add_argument('-std_perc', action='store_true', help='Report of standard deviations distances.')

   return parser


def gen_std_perc_report(files_dir):
    tbl = pd.DataFrame(columns=[
        'method', 'scaler', 'dropout', 'dataset',
        '0 - 1', '1 - 2', '2 - 3', '3 - 4', '4 +'
    ])

    for file in os.listdir(files_dir):
        print(f"######## {file} ########")
        data = {
            'method': None, 'scaler': None, 'dropout': None, 'dataset': None,
            '0 - 1': None, '1 - 2': None, '2 - 3': None, '3 - 4': None, '4 +': None
        }
        infos = file.split('_')
        if infos[0] == 'real':
            data['method'] = 'ANN'
        else:
            data['method'] = 'XGB'

        data['scaler'] = infos[4]
        data['dropout'] = infos[3]
        data['dataset'] = infos[5]

        pred_df = pd.read_csv(f"{files_dir}/{file}")
        b = [0, 1, 2, 3, 4]
        result = percent_distance_std(pred_df['Pred'], pred_df['Real'], b)
        print(result)
        print(f"============================================================")
        format_std_perc_report(b, result)
        result = result * 100

        data['0 - 1'] = np.round(result[0], 2)
        data['1 - 2'] = np.round(result[1], 2)
        data['2 - 3'] = np.round(result[2], 2)
        data['3 - 4'] = np.round(result[3], 2)
        data['4 +'] = np.round(result[4], 2)
        tbl = tbl.append([data], ignore_index=True)

        print(f"############################################################")

    latex_std_perc_report(tbl)

def latex_std_perc_report(tbl):
    print(tbl.to_csv(index=False))


def format_std_perc_report(b , result):
    j = 0
    for i in range(len(b)):
        if i > 0:
            acc = result[j] * 100
            print(f"total entre {b[i - 1]} e {b[i]} std: {acc:.2f}%")
            j = j + 1

    if j != len(result):
        acc = result[-1] * 100
        print(f"total não classificado: {acc:.2f}%")


def percent_distance_std(pred, real, bins):
    bins = np.array(bins, dtype='float32')
    diff = np.power(real - pred, 2)
    std = diff.std(ddof=0)

    print(f"std = {std}")
    bins = bins * std

    interval = []

    for i in range(len(bins)):
        if i > 0:
            interval.append(pd.Interval(left=bins[i - 1], right=bins[i]))

    ii = pd.IntervalIndex(interval,
                          closed='right',
                          dtype='interval[float32]')

    cut = pd.cut(diff, bins=ii)

    s = pd.Series(cut).value_counts(dropna=False).sort_index()
    counts = s.to_numpy()
    total = counts.sum()
    print(s)

    result = np.array([])
    for c in counts:
        percent = c / total
        result = np.append(result, percent)

    return result


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    files_dir = args.dir
    std_perc_report = args.std_perc

    if std_perc_report:
        gen_std_perc_report(files_dir)
