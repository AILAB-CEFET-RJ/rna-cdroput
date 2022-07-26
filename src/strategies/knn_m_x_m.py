import argparse
import pandas as pd

from src.modules import utils

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV

def parser():
    parse = argparse.ArgumentParser(description='ANN Experiments. Script to add expected errors computed by decision tree in the dataset.')

    parse.add_argument('-dataset', metavar='DS', help='Dataset file to use.')

    return parse

def insert_expected_errors(dataset: pd.DataFrame):
    dataset_err = dataset.copy(deep=True)

    grid_search_cv = GridSearchCV(
        estimator  = KNeighborsRegressor(),
        cv         = utils.CROSS_VALIDATION_FOLDS,
        n_jobs     = utils.PARALLEL_JOBS,
        param_grid = {
            'n_neighbors': [4, 5, 6],
            'weights'    : ['uniform', 'distance'],
            'algorithm'  : ['ball_tree', 'kd_tree', 'brute'],
            'p'          : [1, 2, 3],
        },
    )

    model = utils.find_best_model_m_x_m(dataset, grid_search_cv, "knn")

    features = dataset[utils.X_FEATURE_COLUMNS]

    pred = model.predict(features)

    df_err = pd.DataFrame(pred)

    df_err.columns = utils.X_FEATURE_COLUMNS

    idx = dataset_err.columns.get_loc('err_z') + 1

    for b in 'ugriz':
        dataset_err.insert(idx, f"err_{b}_exp", df_err[b], allow_duplicates=True)
        idx = idx + 1

    return dataset_err

def init_expected_errors(dataset_name):
    dataset = pd.read_csv(f"./src/data/{dataset_name}", comment='#')

    name, ext = dataset_name.split('.')

    dataset.Name = name

    dataset_err = insert_expected_errors(dataset)

    dataset_err.to_csv(f"./src/data/{name}_knn_m_x_m_experrs.{ext}", index=False)

if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    utils.rna_cdropout_print("Stage 02: Predicting errors with (knn_m_x_m)")
    init_expected_errors(args.dataset)
