
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    if max_val == min_val:
        return data[column]
    return (data[column] - min_val) / (max_val - min_val)

def standardize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    if std_val == 0:
        return data[column]
    return (data[column] - mean_val) / std_val

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='standard'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        if outlier_removal:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
        elif normalization == 'standard':
            cleaned_df[col] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_cleaning(df, original_df, numeric_columns):
    report = {}
    for col in numeric_columns:
        if col in df.columns and col in original_df.columns:
            report[col] = {
                'original_mean': original_df[col].mean(),
                'cleaned_mean': df[col].mean(),
                'original_std': original_df[col].std(),
                'cleaned_std': df[col].std(),
                'rows_removed': len(original_df) - len(df)
            }
    return pd.DataFrame.from_dict(report, orient='index')
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, convert_types=True):
    """
    Clean a pandas DataFrame by removing duplicates and converting data types.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    convert_types (bool): Whether to convert columns to optimal data types
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if convert_types:
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                try:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                    print(f"Converted column '{col}' to datetime")
                except (ValueError, TypeError):
                    try:
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
                        if cleaned_df[col].dtype != 'object':
                            print(f"Converted column '{col}' to numeric")
                    except Exception:
                        pass
    
    cleaned_df = cleaned_df.reset_index(drop=True)
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    return cleaned_df

def handle_missing_values(df, strategy='mean', fill_value=None):
    """
    Handle missing values in DataFrame columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    strategy (str): One of 'mean', 'median', 'mode', 'constant', or 'drop'
    fill_value: Value to use when strategy is 'constant'
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    df_handled = df.copy()
    
    for col in df_handled.columns:
        if df_handled[col].isnull().any():
            missing_count = df_handled[col].isnull().sum()
            
            if strategy == 'drop':
                df_handled = df_handled.dropna(subset=[col])
                print(f"Dropped {missing_count} rows with missing values in column '{col}'")
            
            elif strategy == 'mean' and np.issubdtype(df_handled[col].dtype, np.number):
                fill_val = df_handled[col].mean()
                df_handled[col] = df_handled[col].fillna(fill_val)
                print(f"Filled {missing_count} missing values in column '{col}' with mean: {fill_val:.2f}")
            
            elif strategy == 'median' and np.issubdtype(df_handled[col].dtype, np.number):
                fill_val = df_handled[col].median()
                df_handled[col] = df_handled[col].fillna(fill_val)
                print(f"Filled {missing_count} missing values in column '{col}' with median: {fill_val:.2f}")
            
            elif strategy == 'mode':
                fill_val = df_handled[col].mode()[0] if not df_handled[col].mode().empty else None
                if fill_val is not None:
                    df_handled[col] = df_handled[col].fillna(fill_val)
                    print(f"Filled {missing_count} missing values in column '{col}' with mode: {fill_val}")
            
            elif strategy == 'constant' and fill_value is not None:
                df_handled[col] = df_handled[col].fillna(fill_value)
                print(f"Filled {missing_count} missing values in column '{col}' with constant: {fill_value}")
    
    return df_handled

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of column names that must be present
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4],
        'value': [10.5, 20.3, 20.3, None, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-02', 'invalid', '2023-01-05', '2023-01-06']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned)
    
    validated, message = validate_dataframe(cleaned, required_columns=['id', 'value'])
    print(f"\nValidation: {message}")