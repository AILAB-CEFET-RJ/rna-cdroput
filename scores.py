import numpy as np
import argparse
import glob

import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments')
    parse.add_argument('-dir', metavar='DIR', help='Dir of pred run files.')

    return parse


def score_mad(real, pred):
    dz_norm = (pred - real) / (1 + pred)
    return np.median(np.abs(dz_norm))


def scores(dir):
    result_df = pd.DataFrame(columns=['Name',
                                      'MSE_MEAN', 'MSE_STD',
                                      'RMSE_MEAN', 'RMSE_STD',
                                      'MAD_MEAN', 'MAD_STD',
                                      'MAE_MEAN', 'MAE_STD',
                                      'R2_MEAN', 'R2_STD',
                                      ])
    data = {}

    preds_root_file_mask = f"{dir}/run_*"
    run_dirs = glob.glob(preds_root_file_mask)

    for run in run_dirs:
        preds_file_mask = f"{run}/real_x_pred_*"
        preds_files = glob.glob(preds_file_mask)

        for pred_file in preds_files:
            name = pred_file.split('real_x_pred_model_')[-1].split('_mse')[0]
            pred_df = pd.read_csv(pred_file, comment='#')

            mse = mean_squared_error(pred_df['Real'], pred_df['Pred'])
            rmse = sqrt(mse)
            mad = score_mad(pred_df['Real'], pred_df['Pred'])
            mae = mean_absolute_error(pred_df['Real'], pred_df['Pred'])
            r2 = r2_score(pred_df['Real'], pred_df['Pred'])

            if name in data:
                data[name]['mse'] = np.append(data[name]['mse'], mse)
                data[name]['rmse'] = np.append(data[name]['rmse'], rmse)
                data[name]['mad'] = np.append(data[name]['mad'], mad)
                data[name]['mae'] = np.append(data[name]['mae'], mae)
                data[name]['r2'] = np.append(data[name]['r2'], r2)
            else:
                data[name] = {}
                data[name]['mse'] = np.array([mse])
                data[name]['rmse'] = np.array([rmse])
                data[name]['mad'] = np.array([mad])
                data[name]['mae'] = np.array([mae])
                data[name]['r2'] = np.array([mse])

    for name in data:
        result_df.loc[len(result_df)] = [
            name,
            np.mean(data[name]['mse']),
            np.std(data[name]['mse']),
            np.mean(data[name]['rmse']),
            np.std(data[name]['rmse']),
            np.mean(data[name]['mad']),
            np.std(data[name]['mad']),
            np.mean(data[name]['mae']),
            np.std(data[name]['mae']),
            np.mean(data[name]['r2']),
            np.std(data[name]['r2'])
        ]

    print(result_df)
    result_df.to_csv('output/scores.csv', index=False)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    scores(args.dir)
