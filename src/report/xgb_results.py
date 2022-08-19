import pandas as pd

if __name__ == "__main__":
    results_df = pd.read_csv("./src/report/tables/xgb_results.csv")

    results_df = results_df.drop(columns=["timestamp"])

    results_df = results_df.drop_duplicates()

    results_df = results_df.sort_values(by=["MSE"])

    results_df = results_df.set_index("MODEL")

    results_df.index = results_df.index.map(
        lambda model: model
            .replace("_test_data", "")
            .replace("_experrs.csv", "")
    )

    results_df.to_latex(
        buf="./src/report/tables/xgb_results.tex",
        caption="Predição de redshift (XGBoost)",
        label="tab:xgb_results",
    )
