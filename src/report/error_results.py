import pandas as pd

if __name__ == "__main__":
    results_df = pd.read_csv("./src/report/tables/errors_results.csv")
    results_df.to_latex(
        buf="./src/report/tables/errors_results.tex",
        caption="Predição de Erros nos modelos de Regressão",
        label="errors_results",
    )
