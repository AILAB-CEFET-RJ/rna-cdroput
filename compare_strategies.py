import os
import pandas as pd
import sklearn.metrics as metrics

def calculate_regression_results(true_values, pred_values):
    explained_variance_score=metrics.explained_variance_score(true_values, pred_values)
    median_absolute_error=metrics.median_absolute_error(true_values, pred_values)
    mean_absolute_error=metrics.mean_absolute_error(true_values, pred_values)
    mean_squared_error=metrics.mean_squared_error(true_values, pred_values)
    r2_score=metrics.r2_score(true_values, pred_values)

    return (
        explained_variance_score,
        median_absolute_error,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

def calculate_multiple_regression_results(
    dataset_name: str,
    dataframe: pd.DataFrame,
    true_target_names: list[str],
    pred_target_names: list[str],
):
    regression_results = []
    for (true_target_name, pred_target_name) in zip(true_target_names, pred_target_names):
        true_values = dataframe[true_target_name]
        pred_values = dataframe[pred_target_name]
        regression_results.append(calculate_regression_results(true_values, pred_values))

    regression_results_df = pd.DataFrame(regression_results, columns=[
        "explained_variance_score",
        "median_absolute_error",
        "mean_absolute_error",
        "mean_squared_error",
        "r2_score",
    ])
    
    regression_results_df["target_names"] = true_target_names

    print(regression_results_df)

def main():
    root_files = [f"./{filename}" for filename in os.listdir(f"./")]

    strategy_files = list(filter(lambda filename: filename.find("experrs") > 0, root_files))

    true_target_names = ['u', 'g', 'r', 'i', 'z']
    pred_target_names = [f"err_{letter}" for letter in true_target_names]

    for strategy_file in strategy_files:
        calculate_multiple_regression_results(
            dataset_name=strategy_file,
            dataframe=pd.read_csv(strategy_file),
            true_target_names=true_target_names,
            pred_target_names=pred_target_names,
        )



if __name__ == '__main__':
    main()
