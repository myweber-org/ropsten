
import pandas as pd
import numpy as np

def clean_csv_data(filepath, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and removing specified columns.
    
    Parameters:
    filepath (str): Path to the CSV file
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    columns_to_drop (list): List of column names to remove from the dataset
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop, errors='ignore')
            print(f"Dropped columns: {columns_to_drop}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if missing_strategy == 'drop':
                df = df.dropna()
                print("Removed rows with missing values")
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if df[col].isnull().any():
                        if missing_strategy == 'mean':
                            fill_value = df[col].mean()
                        elif missing_strategy == 'median':
                            fill_value = df[col].median()
                        elif missing_strategy == 'mode':
                            fill_value = df[col].mode()[0]
                        else:
                            fill_value = 0
                        
                        df[col] = df[col].fillna(fill_value)
                        print(f"Filled missing values in '{col}' with {missing_strategy}: {fill_value}")
        
        print(f"Final data shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    required_columns (list): List of column names that must be present
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Error: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed")
    return True

def save_cleaned_data(df, output_path):
    """
    Save cleaned dataframe to CSV file.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to save
    output_path (str): Path where to save the cleaned data
    
    Returns:
    bool: True if save successful, False otherwise
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(
        filepath=input_file,
        missing_strategy='median',
        columns_to_drop=['id', 'unused_column']
    )
    
    if cleaned_df is not None and validate_dataframe(cleaned_df):
        save_cleaned_data(cleaned_df, output_file)