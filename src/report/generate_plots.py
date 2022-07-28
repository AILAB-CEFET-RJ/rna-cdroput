import glob, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.modules import utils

def generate_pairplots(dataset: pd.DataFrame):
    utils.rna_cdropout_print(f"Generating pairplots for {dataset.Name}")

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

    plt.clf()

def generate_regplots(dataset: pd.DataFrame):
    utils.rna_cdropout_print(f"Generating regplots for {dataset.Name}")

    sns.set_theme(
        style="darkgrid",
        color_codes=True,
        font_scale=1.2,
    )

    for (feature_name, target_name) in zip(utils.X_FEATURE_COLUMNS, utils.Y_TARGET_COLUMNS):
        utils.rna_cdropout_print(f"Generating regplot for {feature_name} x {target_name}")

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

    found_datasets_paths.append(*glob.glob("src/data/*clean.csv"))

    for dataset_path in found_datasets_paths:
        if "sdss" in dataset_path and "clean" not in dataset_path:
            continue

        dataset = pd.read_csv(dataset_path, low_memory=False, comment="#")

        dataset.Name = os.path.basename(dataset_path.split(".")[0])

        utils.rna_cdropout_print(f"Generating plots for {dataset.Name}")

        generate_pairplots(dataset)

        generate_regplots(dataset)

if __name__ == "__main__":
    generate_plots()
