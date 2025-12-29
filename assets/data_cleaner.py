import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and removing specified columns.
    
    Parameters:
    filepath (str): Path to the CSV file
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns_to_drop (list): List of column names to remove from dataset
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if missing_strategy == 'drop':
            df = df.dropna()
            print("Removed rows with missing values")
        elif missing_strategy in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    if missing_strategy == 'mean':
                        fill_value = df[col].mean()
                    else:
                        fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    print(f"Filled missing values in '{col}' with {missing_strategy}: {fill_value:.2f}")
        elif missing_strategy == 'mode':
            for col in df.columns:
                if df[col].isnull().any():
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else None
                    if fill_value is not None:
                        df[col] = df[col].fillna(fill_value)
                        print(f"Filled missing values in '{col}' with mode: {fill_value}")
    
    cleaned_shape = df.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Removed {original_shape[0] - cleaned_shape[0]} rows and {original_shape[1] - cleaned_shape[1]} columns")
    
    return df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if df is None or df.empty:
        print("Validation failed: Dataframe is empty or None")
        return False
    
    if len(df) < min_rows:
        print(f"Validation failed: Dataframe has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed")
    return True

def save_cleaned_data(df, output_path, index=False):
    """
    Save cleaned dataframe to CSV file.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to save
    output_path (str): Path for output CSV file
    index (bool): Whether to include index in output
    """
    
    if df is not None and not df.empty:
        df.to_csv(output_path, index=index)
        print(f"Cleaned data saved to: {output_path}")
        return True
    else:
        print("Cannot save empty dataframe")
        return False

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'temperature': [22.5, np.nan, 24.0, 25.5, np.nan],
        'humidity': [45, 50, np.nan, 55, 60],
        'pressure': [1013, 1012, 1015, np.nan, 1014],
        'category': ['A', 'B', 'A', 'B', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='mean', columns_to_drop=['id'])
    
    if validate_dataframe(cleaned_df, required_columns=['temperature', 'humidity']):
        save_cleaned_data(cleaned_df, 'cleaned_test_data.csv')
    
    import os
    if os.path.exists('test_data.csv'):
        os.remove('test_data.csv')
    if os.path.exists('cleaned_test_data.csv'):
        os.remove('cleaned_test_data.csv')