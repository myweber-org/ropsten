import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_method='mean', output_path=None):
    """
    Load a CSV file, handle missing values, and optionally save cleaned data.
    
    Args:
        filepath (str): Path to input CSV file.
        fill_method (str): Method to fill missing values ('mean', 'median', 'zero').
        output_path (str, optional): Path to save cleaned CSV. If None, returns DataFrame.
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data with shape: {df.shape}")
        
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            print("Missing values per column:")
            print(missing_counts[missing_counts > 0])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df[col].isnull().any():
                    if fill_method == 'mean':
                        fill_value = df[col].mean()
                    elif fill_method == 'median':
                        fill_value = df[col].median()
                    elif fill_method == 'zero':
                        fill_value = 0
                    else:
                        raise ValueError(f"Unsupported fill_method: {fill_method}")
                    
                    df[col].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in '{col}' with {fill_method}: {fill_value:.2f}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list, optional): List of required column names.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df is None or df.empty:
        print("Validation failed: DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5]
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_method='median')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
        print(f"Validation result: {validation_result}")
        print("\nCleaned DataFrame:")
        print(cleaned_df)import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
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

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to numeric and filling NaN with mean.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
    return df

def validate_dataframe(df, required_columns):
    """
    Validate that DataFrame contains required columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    required_columns (list): List of required column names.
    
    Returns:
    bool: True if all required columns are present.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    return True

def main():
    # Example usage
    data = {
        'id': [1, 2, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 30, 35, None],
        'score': [85, 90, 90, 95, 88]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    
    # Remove duplicates
    df_clean = remove_duplicates(df, subset=['id', 'name'], keep='first')
    print("\nAfter removing duplicates:")
    print(df_clean)
    
    # Clean numeric columns
    df_clean = clean_numeric_columns(df_clean, ['age', 'score'])
    print("\nAfter cleaning numeric columns:")
    print(df_clean)
    
    # Validate DataFrame
    is_valid = validate_dataframe(df_clean, ['id', 'name', 'age', 'score'])
    print(f"\nDataFrame validation: {is_valid}")

if __name__ == "__main__":
    main()