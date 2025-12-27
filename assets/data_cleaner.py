
import pandas as pd
import numpy as np
from pathlib import Path

def load_csv_data(file_path):
    """Load CSV file into pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def clean_missing_values(df, strategy='mean'):
    """Handle missing values in DataFrame."""
    if df is None or df.empty:
        print("DataFrame is empty or None.")
        return df
    
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        print("No missing values found.")
        return df
    
    print(f"Found {missing_count} missing values.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
    
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    print(f"Missing values cleaned using '{strategy}' strategy.")
    return df

def remove_duplicates(df):
    """Remove duplicate rows from DataFrame."""
    if df is None or df.empty:
        return df
    
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    
    if removed > 0:
        print(f"Removed {removed} duplicate rows.")
    else:
        print("No duplicates found.")
    
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned DataFrame to CSV."""
    if df is None or df.empty:
        print("No data to save.")
        return False
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

def main():
    """Main function to execute data cleaning pipeline."""
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    print("Starting data cleaning pipeline...")
    
    df = load_csv_data(input_file)
    if df is None:
        return
    
    print(f"Original data shape: {df.shape}")
    
    df = clean_missing_values(df, strategy='median')
    df = remove_duplicates(df)
    
    print(f"Cleaned data shape: {df.shape}")
    
    save_cleaned_data(df, output_file)
    print("Data cleaning completed.")

if __name__ == "__main__":
    main()