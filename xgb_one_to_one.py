import argparse
import pandas as pd

from xgboost import XGBRegressor

def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script to add expected errors computed by decision tree in the dataset.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')
    #parse.add_argument('-objective', metavar='WE', help='choose weight')

    return parse


def apply_xgb_for_band(df, mag_col, err_col):
    X = df[[mag_col]]
    y = df[[err_col]]
    regressor = XGBRegressor(max_depth=5, objective='reg:squarederror')
    y_expected = regressor.fit(X, y)

    return regressor, X, y, y_expected


def apply_xgb(df):
    print('# process_xgb in dataframe')
    df_dt_err = df.copy(deep=True)

    idx = df_dt_err.columns.get_loc('err_z') + 1

    for b in 'ugriz':
        eb = f"err_{b}"
        dt, _, _, _ = apply_xgb_for_band(df.copy(), b, eb)
        pred = dt.predict(df_dt_err[[b]])
        df_dt_err.insert(idx, f"err_{b}_exp", pred, allow_duplicates=True)
        idx = idx + 1

    return df_dt_err


def add_expected_errors_data(dataset_name):
    data = pd.read_csv(dataset_name, comment='#')
    name, ext = dataset_name.split('.')

    data = apply_xgb(data)

    data.to_csv(f"{name}_xgb_experrs.{ext}", index=False)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    add_expected_errors_data(args.dataset)
