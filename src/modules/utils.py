import os, time
import re
import pandas as pd

from statistics import mean

from sklearn import metrics

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# MODEL INPUT AND OUTPUT
X_FEATURE_COLUMNS = [letter for letter in 'ugriz']
Y_TARGET_COLUMNS  = ['err_' + column for column in X_FEATURE_COLUMNS]

# MODEL CONFIG
RANDOM_STATE = 0
USE_ALL_CPU_CORES = -1
TEST_TRAIN_SPLIT_RATIO = 1 / 5
CROSS_VALIDATION_FOLDS = 5
PARALLEL_JOBS = USE_ALL_CPU_CORES

# WRITE RESULT DATASET FUNCTION
RESULTS_DATASET_PATH = f"./src/report/tables/errors_results.csv"

def write_result_dataset(line):
    if os.path.isfile(RESULTS_DATASET_PATH):
      with open(RESULTS_DATASET_PATH, 'a') as results_dataset_file:
        results_dataset_file.write(f"{line},{time.time()}\n")
    else:
      with open(RESULTS_DATASET_PATH, 'a') as results_dataset_file:
        results_dataset_file.write("dataset,regressor,estratégia,mse,mae,r2,timestamp\n")
        results_dataset_file.write(f"{line},{time.time()}\n")

def split_feature_target(dataset):
  X = dataset[X_FEATURE_COLUMNS]
  y = dataset[Y_TARGET_COLUMNS]

  return X, y

def split_train_test(X, y):
  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state = RANDOM_STATE,
    test_size    = TEST_TRAIN_SPLIT_RATIO,
  )

  return X_train, X_test, y_train, y_test

def split_dataset(dataset):
    return split_train_test(*split_feature_target(dataset))

def find_best_model_m_x_m(dataset: pd.DataFrame, grid_search_cv: GridSearchCV, regressor: str):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    grid_search_cv.fit(X_train, y_train)

    generate_grid_search_cv_results(
        grid_search_cv.cv_results_,
        f"{dataset.Name}_{regressor}_m_x_m",
    )

    best_model = grid_search_cv.best_estimator_

    mse = metrics.mean_squared_error(y_test, best_model.predict(X_test))
    mae = metrics.mean_absolute_error(y_test, best_model.predict(X_test))
    r2 = metrics.r2_score(y_test, best_model.predict(X_test))

    write_result_dataset(f"{dataset.Name},{regressor},m_x_m,{mse:.6f},{mae:.6f},{r2:.6f}")

    return best_model

def find_best_model_m_x_1(dataset: pd.DataFrame, grid_search_cv: GridSearchCV, regressor: str):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    best_models = []
    mses = []
    maes = []
    r2s = []
    for target_column in Y_TARGET_COLUMNS:
        y_train_1_target = y_train[target_column]
        y_test_1_target = y_test[target_column]

        grid_search_cv.fit(X_train, y_train_1_target)

        generate_grid_search_cv_results(
            grid_search_cv.cv_results_,
            f"{dataset.Name}_{regressor}_m_x_1_{Y_TARGET_COLUMNS}",
        )

        best_model = grid_search_cv.best_estimator_

        best_models.append(best_model)

        mses.append(metrics.mean_squared_error(y_test_1_target, best_model.predict(X_test)))
        maes.append(metrics.mean_absolute_error(y_test_1_target, best_model.predict(X_test)))
        r2s.append(metrics.r2_score(y_test_1_target, best_model.predict(X_test)))

    write_result_dataset(f"{dataset.Name},{regressor},m_x_1,{mean(mses):.6f},{mean(maes):.6f},{mean(r2s):.6f}")

    return best_models

def find_best_model_1_x_1(dataset: pd.DataFrame, grid_search_cv: GridSearchCV, regressor: str):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    best_models = []
    mses = []
    maes = []
    r2s = []
    for (feature_column, target_column) in zip(X_FEATURE_COLUMNS, Y_TARGET_COLUMNS):
        X_train_1_feature = X_train[feature_column].to_numpy().reshape(-1, 1)
        X_test_1_feature = X_test[feature_column].to_numpy().reshape(-1, 1)

        y_train_1_target = y_train[target_column]
        y_test_1_target = y_test[target_column]

        grid_search_cv.fit(X_train_1_feature, y_train_1_target)

        generate_grid_search_cv_results(
            grid_search_cv.cv_results_,
            f"{dataset.Name}_{regressor}_1_x_1_{X_FEATURE_COLUMNS}_{Y_TARGET_COLUMNS}",
        )

        best_model = grid_search_cv.best_estimator_

        best_models.append(best_model)

        mses.append(metrics.mean_squared_error(y_test_1_target, best_model.predict(X_test_1_feature)))
        maes.append(metrics.mean_absolute_error(y_test_1_target, best_model.predict(X_test_1_feature)))
        r2s.append(metrics.r2_score(y_test_1_target, best_model.predict(X_test_1_feature)))

    write_result_dataset(f"{dataset.Name},{regressor},1_x_1,{mean(mses):.6f},{mean(maes):.6f},{mean(r2s):.6f}")

    return best_models

def rna_cdropout_print(line: str):
    print(f"RNA-CDROPOUT >> {line}")

def generate_grid_search_cv_results(results, filename: str):
    filepath = f"./src/report/tables/{filename}"

    results_df = pd.DataFrame(results)
    results_df.to_csv(filepath + ".csv")
    results_df.to_latex(
        buf=filepath + ".tex",
        label=filename,
        caption=f"Hiperparâmetros: {filename}",
    )
