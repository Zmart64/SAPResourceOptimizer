import pandas as pd


def split_data(df: pd.DataFrame, split_size: float, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # first sort the values by time
    df = df.sort_values('time').reset_index(drop=True)

    split_index = int(len(df) * split_size)

    train_df = df.iloc[:split_index]
    simulate_df = df.iloc[split_index:]

    print(f"Number of columns: {train_df.shape[1]}")

    print(f"Size of the training data: {len(train_df)}")
    print(f"Size of the simulation data: {len(simulate_df)}")

    # save the new files
    train_df.to_csv(output_dir + "/train_data.csv", index=False, sep=";")
    simulate_df.to_csv(output_dir + "/simulate_data.csv", index=False, sep=";")

    return train_df, simulate_df


if __name__ == "__main__":
    df = pd.read_csv("build-data-sorted.csv", delimiter=";")
    output_dir = "split_data"

    split_data(df, 0.8, output_dir)
