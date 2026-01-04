
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_method='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Args:
        file_path (str): Path to input CSV file
        fill_method (str): Method for filling missing values ('mean', 'median', 'mode', 'zero')
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print("Missing values per column:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count}")
            
            for col in df.columns:
                if df[col].isnull().any():
                    if fill_method == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].mean()
                    elif fill_method == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].median()
                    elif fill_method == 'mode':
                        fill_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
                    elif fill_method == 'zero':
                        fill_value = 0
                    else:
                        fill_value = df[col].ffill().bfill().iloc[0] if not df[col].dropna().empty else np.nan
                    
                    df[col].fillna(fill_value, inplace=True)
        
        df = df.drop_duplicates()
        print(f"Cleaned data shape: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            if df[col].isnull().any():
                print(f"Validation warning: Column '{col}' still contains NaN values")
    
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': ['x', 'y', 'z', 'x', np.nan]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='mean')
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
        print(f"Data validation result: {is_valid}")
        
        import os
        os.remove('sample_data.csv')