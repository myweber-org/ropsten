import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (list, optional): Column labels to consider for duplicates.
        keep (str, optional): Which duplicates to keep.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean a numeric column by handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to clean.
        fill_method (str): Method for filling missing values.
    
    Returns:
        pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if fill_method == 'mean':
        fill_value = df[column_name].mean()
    elif fill_method == 'median':
        fill_value = df[column_name].median()
    elif fill_method == 'zero':
        fill_value = 0
    else:
        raise ValueError("Invalid fill_method. Choose from 'mean', 'median', or 'zero'")
    
    df_cleaned = df.copy()
    df_cleaned[column_name] = df_cleaned[column_name].fillna(fill_value)
    
    return df_cleaned
import pandas as pd
import numpy as np

def clean_missing_data(file_path, strategy='mean', columns=None):
    """
    Load a CSV file and handle missing values using specified strategy.
    
    Args:
        file_path (str): Path to the CSV file
        strategy (str): Method for handling missing values ('mean', 'median', 'mode', 'drop')
        columns (list): Specific columns to clean, if None cleans all columns
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    
    if columns is None:
        columns = df.columns
    
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in dataframe")
            continue
            
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            print(f"Column '{column}' has {missing_count} missing values")
            
            if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].mean()
                df[column].fillna(fill_value, inplace=True)
                print(f"  Filled with mean: {fill_value:.2f}")
                
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].median()
                df[column].fillna(fill_value, inplace=True)
                print(f"  Filled with median: {fill_value:.2f}")
                
            elif strategy == 'mode':
                fill_value = df[column].mode()[0] if not df[column].mode().empty else None
                if fill_value is not None:
                    df[column].fillna(fill_value, inplace=True)
                    print(f"  Filled with mode: {fill_value}")
                else:
                    print(f"  Could not determine mode for column '{column}'")
                    
            elif strategy == 'drop':
                df.dropna(subset=[column], inplace=True)
                print(f"  Dropped rows with missing values in column '{column}'")
                
            else:
                print(f"  Strategy '{strategy}' not applicable for column '{column}'")
    
    print(f"Data cleaning complete. Remaining missing values: {df.isnull().sum().sum()}")
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Args:
        df (pandas.DataFrame): Dataframe to save
        output_path (str): Path for output CSV file
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    return False

if __name__ == "__main__":
    # Example usage
    input_file = "sample_data.csv"
    output_file = "cleaned_data.csv"
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['a', 'b', np.nan, 'd', 'e']
    })
    sample_data.to_csv(input_file, index=False)
    
    print("Starting data cleaning process...")
    cleaned_df = clean_missing_data(input_file, strategy='mean', columns=['A', 'B'])
    
    if cleaned_df is not None:
        save_cleaned_data(cleaned_df, output_file)
        print("Process completed successfully")
    else:
        print("Process failed")