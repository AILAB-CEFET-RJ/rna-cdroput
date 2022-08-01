import glob, os

import pandas as pd

from src.modules import utils

def describe_dataset(dataset: pd.DataFrame):
    utils.rna_cdropout_print(f"Describing {dataset.Name}...")

    dataset.drop(columns=["objid"], inplace=True)

    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    describe_df = dataset.describe()

    describe_df_features = describe_df[utils.X_FEATURE_COLUMNS]

    describe_df_features.to_latex(
        buf=f"./src/report/tables/{dataset.Name}_describe_features.tex",
        caption=f"Propiedades Estatísicas do conjunto de dados ({dataset.Name})".replace("_", "\_"),
        label=f"tab:{dataset.Name}_describe_features",
    )

    describe_df_targets = describe_df.drop(columns=utils.X_FEATURE_COLUMNS)

    describe_df_targets.to_latex(
        buf=f"./src/report/tables/{dataset.Name}_describe_targets.tex",
        caption=f"Propiedades Estatísicas do conjunto de dados ({dataset.Name})".replace("_", "\_"),
        label=f"tab:{dataset.Name}_describe_targets",
    )


def init_describe_train_datasets():
    found_datasets_paths = glob.glob("src/data/*data.csv")

    found_datasets_paths.append(*glob.glob("src/data/*clean.csv"))

    for dataset_path in found_datasets_paths:
        if "sdss" in dataset_path and "clean" not in dataset_path:
            continue

        dataset = pd.read_csv(dataset_path, low_memory=False, comment="#")

        dataset.Name = os.path.basename(dataset_path.split(".")[0])

        describe_dataset(dataset)



if __name__ == "__main__":
    init_describe_train_datasets()
