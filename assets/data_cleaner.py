import pandas as pd
import numpy as np

def load_data(filepath):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def remove_outliers_iqr(df, column):
    """Remove outliers from a column using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'.")
    return filtered_df

def normalize_column(df, column):
    """Normalize a column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val != 0:
        df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
        print(f"Column '{column}' normalized.")
    else:
        print(f"Warning: Column '{column}' has constant values. Normalization skipped.")
    return df

def clean_data(df, numeric_columns):
    """Main data cleaning function."""
    if df is None:
        return None
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
        else:
            print(f"Warning: Column '{col}' not found in data.")
    
    print(f"Data shape after outlier removal: {df.shape}")
    
    for col in numeric_columns:
        if col in df.columns:
            df = normalize_column(df, col)
    
    print("Data cleaning completed.")
    return df

if __name__ == "__main__":
    data_path = "sample_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    data = load_data(data_path)
    cleaned_data = clean_data(data, numeric_cols)
    
    if cleaned_data is not None:
        output_path = "cleaned_data.csv"
        cleaned_data.to_csv(output_path, index=False)
        print(f"Cleaned data saved to '{output_path}'.")