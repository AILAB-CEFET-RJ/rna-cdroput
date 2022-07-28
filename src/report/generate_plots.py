import glob, os
import pandas as pd
import seaborn as sns

from src.modules import utils

def generate_pairplots(dataset: pd.DataFrame):
    sns.set_theme(
        style="darkgrid",
        font_scale=2,
    )

    sns.pairplot(
        dataset,
        x_vars=utils.X_FEATURE_COLUMNS,
        y_vars=utils.X_FEATURE_COLUMNS,
    ).savefig(f"src/report/plots/{dataset.Name}_bands_vs_bands.png")

    sns.pairplot(
        dataset,
        x_vars=utils.Y_TARGET_COLUMNS,
        y_vars=utils.Y_TARGET_COLUMNS,
    ).savefig(f"src/report/plots/{dataset.Name}_errors_vs_errors.png")

    sns.pairplot(
        dataset,
        x_vars=utils.X_FEATURE_COLUMNS,
        y_vars=utils.Y_TARGET_COLUMNS,
    ).savefig(f"src/report/plots/{dataset.Name}_bands_vs_errors.png")

def generate_regplots(dataset: pd.DataFrame):
    sns.set_theme(
        style="darkgrid",
        color_codes=True,
    )

    for (feature_name, target_name) in zip(utils.X_FEATURE_COLUMNS, utils.Y_TARGET_COLUMNS):
        figure = sns.regplot(
            x=feature_name,
            y=target_name,
            data=dataset,
            line_kws={"color": "red"},
        ).get_figure()

        figure.savefig(f"src/report/plots/{dataset.Name}_{feature_name}_vs_{target_name}.png")

        figure.clf()

def generate_plots():
    found_datasets_paths = glob.glob("src/data/*data.csv")

    for dataset_path in found_datasets_paths:
        dataset = pd.read_csv(dataset_path)

        dataset.Name = os.path.basename(dataset_path.split(".")[0])

        generate_pairplots(dataset)

        generate_regplots(dataset)

if __name__ == "__main__":
    generate_plots()
