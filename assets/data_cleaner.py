
import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    """
    Load CSV data, remove outliers using z-score,
    and normalize numerical columns.
    """
    df = pd.read_csv(filepath)
    
    # Identify numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Remove outliers using z-score (threshold = 3)
    z_scores = np.abs(stats.zscore(df[numeric_cols].fillna(0)))
    outlier_mask = (z_scores < 3).all(axis=1)
    df_clean = df[outlier_mask].copy()
    
    # Normalize numerical columns
    for col in numeric_cols:
        if df_clean[col].std() > 0:
            df_clean[col] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
    
    return df_clean

def save_cleaned_data(df, output_path):
    """Save cleaned dataframe to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = load_and_clean_data(input_file)
    save_cleaned_data(cleaned_df, output_file)def remove_duplicates(data_list):
    """
    Remove duplicate entries from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    unique_list = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def clean_numeric_data(values, default=0):
    """
    Clean numeric data by converting strings to floats.
    Invalid values are replaced with the default.
    """
    cleaned = []
    for val in values:
        try:
            cleaned.append(float(val))
        except (ValueError, TypeError):
            cleaned.append(default)
    return cleaned

if __name__ == "__main__":
    sample_data = [1, 2, 2, 3, 4, 4, 5]
    print("Original:", sample_data)
    print("Cleaned:", remove_duplicates(sample_data))