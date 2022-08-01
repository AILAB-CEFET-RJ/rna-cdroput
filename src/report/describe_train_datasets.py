import glob, os

import pandas as pd

from src.modules import utils

def describe_dataset(dataset: pd.DataFrame):
    utils.rna_cdropout_print(f"Describing {dataset.Name}...")

    dataset.drop(columns=["objid"], inplace=True)

    describe_df = dataset.describe()

    describe_df.to_latex(
        buf=f"./src/report/tables/{dataset.Name}_describe.tex",
        caption=f"Propiedades Estatísicas do conjunto de dados ({dataset.Name})",
        label=f"tab:{dataset.Name}_describe",
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
