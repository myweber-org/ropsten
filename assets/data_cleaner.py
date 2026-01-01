import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file, missing_strategy='mean'):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
        missing_strategy (str): Strategy for handling missing values.
                                Options: 'mean', 'median', 'drop', 'zero'
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if missing_strategy == 'drop':
                df_cleaned = df.dropna()
                print(f"Dropped rows with missing values. New shape: {df_cleaned.shape}")
            elif missing_strategy == 'mean':
                df_cleaned = df.fillna(df.mean(numeric_only=True))
                print("Filled missing values with column means")
            elif missing_strategy == 'median':
                df_cleaned = df.fillna(df.median(numeric_only=True))
                print("Filled missing values with column medians")
            elif missing_strategy == 'zero':
                df_cleaned = df.fillna(0)
                print("Filled missing values with zeros")
            else:
                print(f"Unknown strategy: {missing_strategy}. Using 'mean' as default")
                df_cleaned = df.fillna(df.mean(numeric_only=True))
        else:
            df_cleaned = df
            print("No missing values found")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        
        return df_cleaned
        
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
        df (pd.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
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
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe()
        print("Numeric columns statistics:")
        print(stats)
    
    return True

if __name__ == "__main__":
    cleaned_df = clean_csv_data('input_data.csv', 'cleaned_data.csv', 'mean')
    if cleaned_df is not None:
        validation_passed = validate_dataframe(cleaned_df)
        if validation_passed:
            print("Data cleaning completed successfully")
        else:
            print("Data cleaning completed but validation failed")