
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns):
    """Normalize data using min-max scaling."""
    df_normalized = df.copy()
    for col in columns:
        min_val = df_normalized[col].min()
        max_val = df_normalized[col].max()
        df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized

def clean_dataset(input_path, output_path, numeric_columns):
    """Main cleaning pipeline."""
    df = load_dataset(input_path)
    print(f"Original dataset shape: {df.shape}")
    
    df_no_outliers = remove_outliers_iqr(df, numeric_columns)
    print(f"After outlier removal: {df_no_outliers.shape}")
    
    df_normalized = normalize_data(df_no_outliers, numeric_columns)
    print(f"Normalization completed")
    
    df_normalized.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    
    return df_normalized

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ["age", "income", "score"]
    
    cleaned_df = clean_dataset(input_file, output_file, numeric_cols)