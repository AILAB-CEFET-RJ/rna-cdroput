import argparse
import pandas as pd

from sklearn.isotonic import IsotonicRegression


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script to add expected errors computed by isotonic regression in the dataset.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')

    return parse


def apply_isotonic_regression_for_band(df, mag, mag_err):
    df.sort_values(by=[mag], inplace=True)
    df = df.reset_index(drop=True)
    x = df[mag]
    y = df[mag_err]
    ir = IsotonicRegression()
    y_expected = ir.fit_transform(x, y)

    return ir, x, y, y_expected


def apply_isotonic_regression(df):
    print('# process_isotonic_regression in dataframe')
    df_ir_err = df.copy(deep=True)

    idx = df_ir_err.columns.get_loc('err_z') + 1

    for b in 'ugriz':
        eb = f"err_{b}"
        ir, _, _, _ = apply_isotonic_regression_for_band(df.copy(), b, eb)
        pred = ir.predict(df_ir_err[b])
        df_ir_err.insert(idx, f"err_{b}_exp", pred, allow_duplicates=True)
        idx = idx + 1

    return df_ir_err


def add_expected_errors_data(dataset_name):
    data = pd.read_csv(dataset_name, comment='#')
    name, ext = dataset_name.split('.')

    data = apply_isotonic_regression(data)

    data.to_csv(f"{name}_ir_experrs.{ext}", index=False)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    add_expected_errors_data(args.dataset)
