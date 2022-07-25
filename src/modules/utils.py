import os
import pandas as pd

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
RESULTS_DATASET_PATH = f"./src/data/errors_results.csv"

def write_result_dataset(line):
    if os.path.isfile(RESULTS_DATASET_PATH):
      with open(RESULTS_DATASET_PATH, 'a') as results_dataset_file:
        results_dataset_file.write(f"{line}\n")
    else:
      with open(RESULTS_DATASET_PATH, 'a') as results_dataset_file:
        results_dataset_file.write("dataset,regressor,estratégia,mse, mae, r2\n")
        results_dataset_file.write(f"{line}\n")


def split_feature_target(dataset_df):
  X = dataset_df[X_FEATURE_COLUMNS]
  y = dataset_df[Y_TARGET_COLUMNS]

  return X, y

def split_train_test(X, y):
  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state = RANDOM_STATE,
    test_size    = TEST_TRAIN_SPLIT_RATIO,
  )

  return X_train, X_test, y_train, y_test

def find_best_model_m_x_m(dataset: pd.DataFrame, gridSearch: GridSearchCV, regressor: str):
    X = dataset[X_FEATURE_COLUMNS]
    y = dataset[Y_TARGET_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
      X,
      y,
      random_state = RANDOM_STATE,
      test_size    = TEST_TRAIN_SPLIT_RATIO,
    )

    gridSearch.fit(X_train, y_train)

    best_model = gridSearch.best_estimator_

    mse = metrics.mean_squared_error(y_test, best_model.predict(X_test))
    mae = metrics.mean_absolute_error(y_test, best_model.predict(X_test))
    r2 = metrics.r2_score(y_test, best_model.predict(X_test))

    write_result_dataset(f"{dataset.Name},{regressor},m_x_m,{mse:.6f},{mae:.6f},{r2:.6f}")

    return best_model
