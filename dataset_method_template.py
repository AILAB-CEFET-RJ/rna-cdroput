import argparse
import pandas as pd


def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script to add expected errors computed by ?? in the dataset.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')

    return parse


def apply_method(df):
    print('# process_?? in dataframe')
    df_exp_err = df.copy(deep=True)

    idx = df_exp_err.columns.get_loc('err_z') + 1

    #TODO
    # 1. implement method call here
    # 2. insert expected errors after idx position with column name like 'err_<band>_exp.
    #       See 'dataset_isotonic_regression.py'
    # 3. replace all '??' by implemented method name

    return df_exp_err


def add_expected_errors_data(dataset_name):
    data = pd.read_csv(dataset_name, comment='#')
    name, ext = dataset_name.split('.')

    data = apply_method(data)

    data.to_csv(f"{name}_??_experrs.{ext}", index=False)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    add_expected_errors_data(args.dataset)
