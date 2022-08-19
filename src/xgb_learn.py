import argparse
import pandas as pd

from src.modules import utils

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from statistics import mean

from sklearn import metrics

def parser():
    parse = argparse.ArgumentParser(description='XGB Experiments')
    parse.add_argument('-trainset', metavar='TRAINSET', help='Experiment Trainset.')
    parse.add_argument('-testsets', nargs='+', metavar='TESTSET', help='Experiment Testsets.')

    return parse

def learn(trainset: str, testsets: list[str]):
    trainset_df = pd.read_csv(f"./src/data/{trainset}", comment='#')

    testsets_dfs = [pd.read_csv(f"./src/data/{testset}", comment='#') for testset in testsets]

    grid_search_cv = GridSearchCV(
        estimator  = XGBRegressor(),
        cv         = utils.CROSS_VALIDATION_FOLDS,
        n_jobs     = utils.PARALLEL_JOBS,
        param_grid = {
            'max_depth': [5, 6, 10],
            'objective': ['reg:squarederror'],
        },
    )

    features = [*utils.X_FEATURE_COLUMNS, *utils.Y_TARGET_COLUMNS]

    target = 'redshift'

    X = trainset_df[features]
    y = trainset_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state = utils.RANDOM_STATE,
        test_size    = utils.TEST_TRAIN_SPLIT_RATIO,
    )

    utils.rna_cdropout_print(f"Training model in trainset: {trainset}...")
    grid_search_cv.fit(X_train, y_train)

    utils.generate_grid_search_cv_results(grid_search_cv.cv_results_, trainset)

    best_model = grid_search_cv.best_estimator_

    train_mse = metrics.mean_squared_error(y_test, best_model.predict(X_test))
    train_mae = metrics.mean_absolute_error(y_test, best_model.predict(X_test))
    train_r2 = metrics.r2_score(y_test, best_model.predict(X_test))

    utils.write_xgb_result_dataset(f"{trainset},{train_mse:.6f},{train_mae:.6f},{train_r2:.6f}")

    for (testset, testset_df) in zip(testsets, testsets_dfs):
        utils.rna_cdropout_print(f"Testing model in testset: {trainset}...")

        test_X = testset_df[features]
        test_y = testset_df[target]

        test_mse = metrics.mean_squared_error(test_y, best_model.predict(test_X))
        test_mae = metrics.mean_absolute_error(test_y, best_model.predict(test_X))
        test_r2 = metrics.r2_score(test_y, best_model.predict(test_X))

        utils.write_xgb_result_dataset(f"{testset},{test_mse:.6f},{test_mae:.6f},{test_r2:.6f}")

if __name__ == '__main__':
    parser = parser()
    args = parser.parse_args()

    utils.rna_cdropout_print("Starting XGB Experiments...")

    learn(args.trainset, args.testsets)
