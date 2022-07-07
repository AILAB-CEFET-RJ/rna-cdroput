import argparse
import pandas as pd  
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

X_FEATURE_COLUMNS = [letter for letter in 'ugriz']
Y_TARGET_COLUMNS  = ['err_' + column for column in X_FEATURE_COLUMNS]
TEST_TRAIN_SPLIT_RATIO = 1 / 5

def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script to add expected errors computed by decision tree in the dataset.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')

    return parse

def split_feature_target(dataset_df):
  X = dataset_df[X_FEATURE_COLUMNS]
  y = dataset_df[Y_TARGET_COLUMNS]

  return X, y

def split_train_test(X, y):
  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state = 0,
    test_size    = TEST_TRAIN_SPLIT_RATIO,
  )

  return X_train, X_test, y_train, y_test

def apply_dt_for_band(df):
    X, y = split_feature_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    return X_train, X_test, y_train, y_test


def apply_dt(df, max_depth=5):
    print('# process_dt in dataframe')
    df_dt_err = df.copy(deep=True)
    X_train, X_test, y_train, y_test = apply_dt_for_band(df_dt_err)

    dt = DecisionTreeRegressor(random_state=0, max_depth=max_depth)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    df_pred = pd.DataFrame(pred)
    df_pred.columns = ['u', 'g', 'r', 'i', 'z']

    idx = df_dt_err.columns.get_loc('err_z') + 1

    for b in 'ugriz':
        df_dt_err.insert(idx, f"err_{b}_exp", df_pred[b], allow_duplicates=True)
        idx = idx + 1

    return df_dt_err


def add_expected_errors_data(dataset_name):
    data = pd.read_csv(dataset_name, comment='#')
    name, ext = dataset_name.split('.')

    data = apply_dt(data)

    data.to_csv(f"{name}_dt_experrs.{ext}", index=False)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    add_expected_errors_data(args.dataset)
