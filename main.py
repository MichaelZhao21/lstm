import requests
import zipfile
import os
import pandas as pd
from model.lstm import Model


def get_year_list() -> list[str]:
    """Formats the year list to match the dataset's naming convention."""

    # Get 5 years of data
    years = range(2019, 2024)
    versions = ["a", "b"]
    return [f"{year}{version}" for year in years for version in versions]


def download_data():
    """Downloads 5 years of data from the dataset website."""
    print("Downloading data...")

    year_list = get_year_list()
    BASE_URL = "https://www.bgc-jena.mpg.de/wetter/mpi_roof_"

    # Download all the data
    for y in year_list:
        # Download the file
        response = requests.get(f"{BASE_URL}{y}.zip")
        with open("data.zip", "wb") as f:
            f.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile("data.zip", "r") as zip_ref:
            zip_ref.extractall()

        # Remove the original zip file
        os.remove("data.zip")

        print(f"Downloaded {y}")

    print("DONE!\n")


def load_data() -> pd.DataFrame:
    """Load all datafiles and concatenate it into one big dataframe"""

    year_list = get_year_list()

    # Concat all dataframes
    dfs = []
    for y in year_list:
        dfs.append(pd.read_csv(f"mpi_roof_{y}.csv", encoding="ISO-8859-1"))
    df = pd.concat(dfs, ignore_index=True)
    return df


def main():
    download_data()
    df = load_data()
    lstm = Model()
    lstm.pre_process(df)
    lstm.setup()
    lstm.train(epochs=5)
    


if __name__ == "__main__":
    main()
