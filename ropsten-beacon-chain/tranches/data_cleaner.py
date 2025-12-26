import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values.
                               Options: 'mean', 'median', 'drop', 'zero'
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        original_rows = len(df)
        
        df = df.drop_duplicates()
        print(f"Removed {original_rows - len(df)} duplicate rows")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            for col in numeric_columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
        elif missing_strategy == 'zero':
            df.fillna(0, inplace=True)
        elif missing_strategy == 'drop':
            df.dropna(inplace=True)
        
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_columns:
            if df[col].isnull().any():
                df[col].fillna('Unknown', inplace=True)
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to: {output_file}")
        print(f"Final data shape: {df.shape}")
        
        missing_summary = df.isnull().sum()
        if missing_summary.sum() == 0:
            print("No missing values remaining")
        else:
            print("Remaining missing values by column:")
            print(missing_summary[missing_summary > 0])
            
        return df
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    if df.isnull().sum().sum() > 0:
        print("Warning: DataFrame contains missing values")
    
    return True

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    cleaned_data = clean_csv_data(input_csv, output_csv, missing_strategy='mean')
    
    if cleaned_data is not None:
        is_valid = validate_dataframe(cleaned_data)
        if is_valid:
            print("Data cleaning completed successfully")
        else:
            print("Data validation failed")