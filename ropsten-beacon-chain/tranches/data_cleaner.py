import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def remove_outliers(df, column, threshold=3):
    """Remove outliers using the Z-score method."""
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return df
    
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    filtered_df = df[z_scores < threshold]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'.")
    return filtered_df

def normalize_column(df, column):
    """Normalize a column to range [0, 1]."""
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return df
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        print(f"Column '{column}' has constant values. Normalization skipped.")
        return df
    
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    print(f"Column '{column}' normalized successfully.")
    return df

def clean_data(df, numeric_columns, outlier_threshold=3):
    """Main data cleaning function."""
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return df
    
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers(cleaned_df, col, outlier_threshold)
            cleaned_df = normalize_column(cleaned_df, col)
        else:
            print(f"Skipping column '{col}' as it does not exist.")
    
    print(f"Data cleaning completed. Final shape: {cleaned_df.shape}")
    return cleaned_df

if __name__ == "__main__":
    data = load_data("sample_data.csv")
    
    if data is not None:
        numeric_cols = ['age', 'income', 'score']
        cleaned_data = clean_data(data, numeric_cols)
        
        cleaned_data.to_csv("cleaned_data.csv", index=False)
        print("Cleaned data saved to 'cleaned_data.csv'")