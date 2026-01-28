
import pandas as pd
import numpy as np

def remove_missing_rows(df, threshold=0.5):
    """
    Remove rows with missing values exceeding threshold percentage.
    
    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Maximum allowed missing percentage per row (0-1)
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    missing_per_row = df.isnull().mean(axis=1)
    return df[missing_per_row <= threshold].reset_index(drop=True)

def fill_missing_with_median(df, columns=None):
    """
    Fill missing values with column median.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Specific columns to fill, None for all numeric columns
    
    Returns:
        pd.DataFrame: Dataframe with filled values
    """
    df_filled = df.copy()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            median_val = df[col].median()
            df_filled[col] = df[col].fillna(median_val)
    
    return df_filled

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Specific columns to check, None for all numeric columns
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: Dataframe without outliers
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    mask = pd.Series([True] * len(df))
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            mask = mask & col_mask
    
    return df_clean[mask].reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): Specific columns to standardize
    
    Returns:
        pd.DataFrame: Dataframe with standardized columns
    """
    df_std = df.copy()
    
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                df_std[col] = (df[col] - mean_val) / std_val
    
    return df_std

def clean_dataset(df, missing_threshold=0.3, outlier_multiplier=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe
        missing_threshold (float): Threshold for missing value removal
        outlier_multiplier (float): Multiplier for outlier detection
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print(f"Initial shape: {df.shape}")
    
    # Step 1: Remove rows with excessive missing values
    df_clean = remove_missing_rows(df, threshold=missing_threshold)
    print(f"After missing value removal: {df_clean.shape}")
    
    # Step 2: Fill remaining missing values
    df_clean = fill_missing_with_median(df_clean)
    
    # Step 3: Remove outliers
    df_clean = remove_outliers_iqr(df_clean, multiplier=outlier_multiplier)
    print(f"After outlier removal: {df_clean.shape}")
    
    # Step 4: Standardize numeric columns
    df_clean = standardize_columns(df_clean)
    
    return df_clean
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_missing='mean', remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): List of columns to check for missing values, defaults to all columns
    fill_missing (str): Method to fill missing values - 'mean', 'median', 'mode', or 'drop'
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
                
                if fill_missing == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
                elif fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif fill_missing == 'mode':
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                else:
                    df_clean[col] = df_clean[col].fillna(0)
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {df_clean.shape}")
    return df_clean

def validate_data(df, required_columns=None, numeric_columns=None):
    """
    Validate data quality after cleaning.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    numeric_columns (list): List of columns that should be numeric
    
    Returns:
    dict: Dictionary with validation results
    """
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'data_types': {},
        'validation_passed': True
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['missing_columns'] = missing_cols
            validation_results['validation_passed'] = False
    
    for col in df.columns:
        validation_results['missing_values'][col] = df[col].isnull().sum()
        validation_results['data_types'][col] = str(df[col].dtype)
    
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation_results['validation_passed'] = False
                    if 'non_numeric_columns' not in validation_results:
                        validation_results['non_numeric_columns'] = []
                    validation_results['non_numeric_columns'].append(col)
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', 'Frank'],
        'age': [25, 30, None, 35, 28, 28, 40],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, fill_missing='mean', remove_duplicates=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_data(cleaned_df, 
                              required_columns=['id', 'name', 'age', 'score'],
                              numeric_columns=['age', 'score'])
    
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")