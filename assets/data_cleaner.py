import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values and removing invalid columns.
    
    Args:
        filepath (str): Path to the CSV file.
        fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero').
        drop_threshold (float): Threshold for dropping columns with too many missing values (0.0 to 1.0).
    
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Drop columns with missing values above threshold
    missing_ratio = df.isnull().sum() / len(df)
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    df = df.drop(columns=columns_to_drop)
    
    # Fill missing values based on strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif fill_strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif fill_strategy == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
    elif fill_strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)
    else:
        raise ValueError("Invalid fill_strategy. Choose from 'mean', 'median', 'mode', or 'zero'.")
    
    # For non-numeric columns, fill with most frequent value
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index after cleaning
    df = df.reset_index(drop=True)
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a new CSV file.
    
    Args:
        df (pandas.DataFrame): Cleaned DataFrame.
        output_path (str): Path to save the cleaned CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(input_file, fill_strategy='median', drop_threshold=0.3)
        save_cleaned_data(cleaned_df, output_file)
        print(f"Original shape: {pd.read_csv(input_file).shape}")
        print(f"Cleaned shape: {cleaned_df.shape}")
    except Exception as e:
        print(f"Error during data cleaning: {e}")