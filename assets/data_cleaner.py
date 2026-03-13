
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def remove_outliers(df, column, threshold=3):
    """Remove outliers using the Z-score method."""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]

def normalize_column(df, column):
    """Normalize a column using Min-Max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_data(df, numeric_columns):
    """Main cleaning function."""
    for col in numeric_columns:
        df = remove_outliers(df, col)
        df = normalize_column(df, col)
    return df.dropna()

if __name__ == "__main__":
    data = load_data("raw_data.csv")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cleaned_data = clean_data(data, numeric_cols)
    cleaned_data.to_csv("cleaned_data.csv", index=False)
    print("Data cleaning completed. Saved to cleaned_data.csv")