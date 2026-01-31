import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): List of columns to check for missing values. If None, checks all columns.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', 'drop', or a scalar value)
    remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns_to_check is None:
        columns_to_check = df_clean.columns.tolist()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    for col in columns_to_check:
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                print(f"Column '{col}' has {missing_count} missing values")
                
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"  Filled with mean: {fill_value:.2f}")
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"  Filled with median: {fill_value:.2f}")
                elif fill_missing == 'mode':
                    fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else None
                    df_clean[col] = df_clean[col].fillna(fill_value)
                    print(f"  Filled with mode: {fill_value}")
                elif fill_missing == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
                    print(f"  Dropped rows with missing values in column '{col}'")
                elif isinstance(fill_missing, (int, float, str)):
                    df_clean[col] = df_clean[col].fillna(fill_missing)
                    print(f"  Filled with constant: {fill_missing}")
                else:
                    print(f"  Warning: Could not fill missing values in column '{col}'")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    return df_clean

def validate_dataframe(df, required_columns=None, numeric_columns=None):
    """
    Validate DataFrame structure and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    numeric_columns (list): List of columns that should be numeric
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'non_numeric_columns': [],
        'missing_values': {}
    }
    
    if required_columns:
        for col in required_columns:
            if col not in df.columns:
                validation_results['missing_columns'].append(col)
                validation_results['is_valid'] = False
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation_results['non_numeric_columns'].append(col)
                    validation_results['is_valid'] = False
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            validation_results['missing_values'][col] = missing_count
    
    return validation_results

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of columns to check for outliers
    multiplier (float): IQR multiplier (default 1.5)
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    for col in columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    removed = initial_rows - len(df_clean)
    print(f"Removed {removed} outliers using IQR method")
    return df_clean

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10.5, 20.3, np.nan, 40.7, 50.1, 50.1, 1000],
        'category': ['A', 'B', 'A', np.nan, 'C', 'C', 'D'],
        'score': [85, 92, 78, 88, np.nan, 88, 95]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, fill_missing='mean', remove_duplicates=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_dataframe(cleaned_df, 
                                   required_columns=['id', 'value', 'category', 'score'],
                                   numeric_columns=['id', 'value', 'score'])
    print("\nValidation Results:")
    print(validation)
    
    df_no_outliers = remove_outliers_iqr(cleaned_df, ['value', 'score'])
    print("\nDataFrame after outlier removal:")
    print(df_no_outliers)