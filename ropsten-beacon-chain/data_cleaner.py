import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    file_path (str): Path to input CSV file
    output_path (str): Path for cleaned output file (optional)
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        original_rows = len(df)
        
        df = df.drop_duplicates()
        print(f"Removed {original_rows - len(df)} duplicate rows")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
                    print(f"Filled missing values in {col} with mean: {mean_val:.2f}")
        
        elif missing_strategy == 'median':
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    print(f"Filled missing values in {col} with median: {median_val:.2f}")
        
        elif missing_strategy == 'drop':
            df = df.dropna()
            print(f"Dropped rows with missing values")
        
        else:
            print(f"Unknown strategy: {missing_strategy}. Using 'mean' as default.")
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    df[col].fillna(mean_val, inplace=True)
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        print(f"Final data shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    
    if df is None or df.empty:
        print("Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    print(f"Data validation passed. Shape: {df.shape}, Columns: {list(df.columns)}")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'value': [10.5, None, 15.2, 10.5, None, 18.7],
        'category': ['A', 'B', 'A', 'A', 'B', 'C']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', 'cleaned_data.csv', 'mean')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
        print(f"Validation result: {validation_result}")