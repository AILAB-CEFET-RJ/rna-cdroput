import pandas as pd

import glob, os

def split_table(csv_path: str):
    split_counter = 0
    filename = os.path.basename(csv_path).split(".")[0]
    table_df = pd.read_csv(csv_path)

    while len(table_df.index) >= 25:
        split_counter += 1

        splitted_table_df = table_df.head(25)

        splitted_table_df.to_latex(
            buf=f"src/report/tables_splitted/{filename}_split_{split_counter:02d}" + ".tex",
            label=f"table_{filename}_split_{split_counter:02d}",
            caption=f"Hiperparâmetros: {filename}".replace("_", "\_"),
        )

        table_df = table_df.iloc[25:]


def init_split_tables():
    csv_paths = glob.glob("src/report/tables/*.csv")

    for csv_path in csv_paths:
        split_table(csv_path)

if __name__ == "__main__":
    init_split_tables()
