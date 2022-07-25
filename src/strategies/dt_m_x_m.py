import argparse
import pandas as pd

from src.modules import utils

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script to add expected errors computed by decision tree in the dataset.')
    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')

    return parse

def apply_dt_m_x_m(dataset: pd.DataFrame):
    dataset_err = dataset.copy(deep=True)

    dt_m_x_m_grid_search_cv = GridSearchCV(
        estimator  = DecisionTreeRegressor(random_state = utils.RANDOM_STATE),
        cv         = utils.CROSS_VALIDATION_FOLDS,
        n_jobs     = utils.PARALLEL_JOBS,
        param_grid = {
            'max_depth'   : [5, 10, 15],
        },
    )

    model = utils.find_best_model_m_x_m(dataset, dt_m_x_m_grid_search_cv, "dt")

    features = dataset[utils.X_FEATURE_COLUMNS]

    pred = model.predict(features)

    df_err = pd.DataFrame(pred)
    df_err.columns = utils.X_FEATURE_COLUMNS

    idx = dataset_err.columns.get_loc('err_z') + 1

    for b in 'ugriz':
        dataset_err.insert(idx, f"err_{b}_exp", df_err[b], allow_duplicates=True)
        idx = idx + 1

    return dataset_err


def insert_expected_errors_data(dataset_name):
    dataset = pd.read_csv(f"./src/data/{dataset_name}", comment='#')

    dataset.Name = dataset_name

    name, ext = dataset_name.split('.')

    dataset_err = apply_dt_m_x_m(dataset)

    dataset_err.to_csv(f"./src/data/{name}_dt_m_x_m_experrs.{ext}", index=False)


if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    insert_expected_errors_data(args.dataset)
