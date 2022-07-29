import glob, os
import pandas as pd
import seaborn as sns

from src.modules import utils

sns.set_theme(font_scale=1.5, rc={'figure.figsize':(11.7,8.27)})

def generate_pairplots(dataset: pd.DataFrame):
    utils.rna_cdropout_print(f"Generating pairplots for {dataset.Name}")

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

def generate_plots():
    found_datasets_paths = glob.glob("src/data/*data.csv")

    found_datasets_paths.append(*glob.glob("src/data/*clean.csv"))

    for dataset_path in found_datasets_paths:
        if "sdss" in dataset_path and "clean" not in dataset_path:
            continue

        dataset = pd.read_csv(dataset_path, low_memory=False, comment="#")

        dataset.Name = os.path.basename(dataset_path.split(".")[0])

        generate_pairplots(dataset)

if __name__ == "__main__":
    generate_plots()
