import pandas as pd

# Read the CSV file (semicolon-delimited)
df = pd.read_csv("../build-data.csv", sep=";", engine="python")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Convert the 'time' column to datetime
df['time'] = pd.to_datetime(df['time'].str.strip())

# Sort the DataFrame by 'time'
df_sorted = df.sort_values(by='time').reset_index(drop=True)

# Write the sorted DataFrame to a new CSV file
df_sorted.to_csv("build-data-sorted.csv", sep=";", index=False)
