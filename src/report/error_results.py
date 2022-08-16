import pandas as pd

if __name__ == "__main__":
    results_df = pd.read_csv("./src/report/tables/errors_results.csv")

    results_df.drop(columns=["timestamp"], inplace=True)

    results_df.drop_duplicates(inplace=True)

    results_df.rename(columns={
        "mse": "MSE",
        "mae": "MAE",
        "r2": "R2",
    }, inplace=True)

    results_df["MODEL"] = results_df["dataset"] + "_" + results_df["regressor"] + "_" +  results_df["estratégia"]

    results_df["MODEL"].apply(lambda name: name.replace("_data", ""))

    results_df.drop(columns=["dataset", "regressor", "estratégia"], inplace=True)

    results_df.sort_values(by=["MSE"], inplace=True)

    results_df.set_index("MODEL", inplace=True)

    results_df.to_latex(
        buf="./src/report/tables/errors_results.tex",
        caption="Predição de Erros nos modelos de Regressão",
        label="tab:errors_results",
    )
