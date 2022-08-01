import pandas as pd

if __name__ == "__main__":
    results_df = pd.read_csv("./src/report/tables/errors_results.csv")

    results_df.drop(columns=["timestamp"], inplace=True)

    results_df.set_index("mse", inplace=True)

    results_df.sort_index(inplace=True)

    results_df.drop_duplicates(inplace=True)

    results_df.to_latex(
        buf="./src/report/tables/errors_results.tex",
        caption="Predição de Erros nos modelos de Regressão",
        label="tab:errors_results",
    )
