import argparse
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script to add expected errors computed by decision tree in the dataset.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')
    #parse.add_argument('-estimator', metavar='ES', help='Quantity estimators apply')
    #parse.add_argument('-dep', metavar='DP', help='Max choose Depth')

    return parse


def apply_random_forest_for_band(df, mag_col, err_col):
    X = df[[mag_col]]
    y = df[[err_col]]
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0, max_depth = 10)
    y_expected = regressor.fit(X, y)

    return regressor, X, y, y_expected


def apply_random_forest(df):
    print('# process_random_forest in dataframe')
    df_dt_err = df.copy(deep=True)

    idx = df_dt_err.columns.get_loc('err_z') + 1

    for b in 'ugriz':
        eb = f"err_{b}"
        dt, _, _, _ = apply_random_forest_for_band(df.copy(), b, eb)
        pred = dt.predict(df_dt_err[[b]])
        df_dt_err.insert(idx, f"err_{b}_exp", pred, allow_duplicates=True)
        idx = idx + 1

    return df_dt_err


def add_expected_errors_data(dataset_name):
    data = pd.read_csv(dataset_name, comment='#')
    name, ext = dataset_name.split('.')

    data = apply_random_forest(data)

    data.to_csv(f"{name}_rf_experrs.{ext}", index=False)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    add_expected_errors_data(args.dataset)
