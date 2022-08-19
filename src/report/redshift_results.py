import pandas as pd

def init_redshift_results():
    results_path = "output/scores.csv"

    results_df = pd.read_csv(results_path)

    results_df.sort_values("MSE_MEAN", inplace=True)

    results_df.set_index("Name", inplace=True)

    results_df.index.name = "model"

    results_df = results_df[["MSE_MEAN", "RMSE_MEAN", "MAD_MEAN", "R2_MEAN"]]

    results_df.index = results_df.index.map(
        lambda model: model
            .split("real_x_pred_")[1]
            .replace("_training", "")
            .replace("_model", "")
            .replace("_teddy", "")
            .replace("_happy", "")
    )

    results_df.to_latex(
        buf="src/report/tables/redshift_results" + ".tex",
        label="tab:redshift_results",
        caption="Predição de Redshifts com os Modelos de Regressão",
    )

if __name__ == "__main__":
    init_redshift_results()

