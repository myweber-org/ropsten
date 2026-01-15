
import pandas as pd
import numpy as np

def clean_dataset(df, strategy='mean', outlier_method='iqr', threshold=1.5):
    """
    Clean a dataset by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    strategy (str): Strategy for missing values ('mean', 'median', 'mode', 'drop')
    outlier_method (str): Method for outlier detection ('iqr', 'zscore')
    threshold (float): Threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif strategy == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    elif strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Handle outliers
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)
            cleaned_df = cleaned_df[mask]
        
        elif outlier_method == 'zscore':
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            cleaned_df = cleaned_df[z_scores < threshold]
    
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if len(df) < min_rows:
        return False, f"Dataset must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numerical columns in the dataset.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of columns to normalize (None for all numeric)
    method (str): Normalization method ('minmax', 'standard')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    
    normalized_df = df.copy()
    
    if columns is None:
        columns = normalized_df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[col]):
            if method == 'minmax':
                col_min = normalized_df[col].min()
                col_max = normalized_df[col].max()
                if col_max != col_min:
                    normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            
            elif method == 'standard':
                col_mean = normalized_df[col].mean()
                col_std = normalized_df[col].std()
                if col_std != 0:
                    normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned = clean_dataset(df, strategy='median', outlier_method='iqr')
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    is_valid, message = validate_data(cleaned, required_columns=['A', 'B', 'C'])
    print(f"Validation: {is_valid} - {message}")
    print()
    
    normalized = normalize_data(cleaned, method='minmax')
    print("Normalized DataFrame:")
    print(normalized)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, handle_nulls='drop', null_threshold=0.5):
    """
    Clean a pandas DataFrame by handling duplicates and null values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    handle_nulls (str): Strategy for handling nulls - 'drop', 'fill', or 'threshold'
    null_threshold (float): Threshold for column null percentage when handle_nulls='threshold'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    null_counts = cleaned_df.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls > 0:
        print(f"Found {total_nulls} null values in dataset")
        
        if handle_nulls == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Dropped rows with null values")
            
        elif handle_nulls == 'fill':
            for column in cleaned_df.columns:
                if cleaned_df[column].dtype in ['int64', 'float64']:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                else:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown')
            print("Filled null values with appropriate replacements")
            
        elif handle_nulls == 'threshold':
            columns_to_drop = []
            for column in cleaned_df.columns:
                null_percentage = null_counts[column] / len(cleaned_df)
                if null_percentage > null_threshold:
                    columns_to_drop.append(column)
            
            if columns_to_drop:
                cleaned_df = cleaned_df.drop(columns=columns_to_drop)
                print(f"Dropped columns with >{null_threshold*100}% nulls: {columns_to_drop}")
            
            cleaned_df = cleaned_df.dropna()
            print("Dropped remaining rows with null values")
    
    print(f"Cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', None],
        'age': [25, 30, 30, None, 35, 40],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, handle_nulls='fill')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    print("\nData Summary:")
    summary = get_data_summary(cleaned)
    for key, value in summary.items():
        if key != 'numeric_stats':
            print(f"{key}: {value}")
    
    print("\nValidation check:")
    is_valid = validate_dataframe(cleaned, required_columns=['id', 'name', 'age'])
    print(f"DataFrame is valid: {is_valid}")
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_method='iqr', columns=None):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for missing values. Options: 'mean', 'median', 'mode', 'drop'.
    outlier_method (str): Method for outlier detection. Options: 'iqr', 'zscore'.
    columns (list): Specific columns to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if df_clean[col].dtype in [np.float64, np.int64]:
            # Handle missing values
            if missing_strategy == 'mean':
                fill_value = df_clean[col].mean()
            elif missing_strategy == 'median':
                fill_value = df_clean[col].median()
            elif missing_strategy == 'mode':
                fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0
            elif missing_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                continue
            else:
                fill_value = 0
                
            df_clean[col] = df_clean[col].fillna(fill_value)
            
            # Handle outliers
            if outlier_method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean[col] = np.where(
                    df_clean[col] < lower_bound, 
                    lower_bound, 
                    np.where(
                        df_clean[col] > upper_bound, 
                        upper_bound, 
                        df_clean[col]
                    )
                )
                
            elif outlier_method == 'zscore':
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                if std_val > 0:
                    z_scores = np.abs((df_clean[col] - mean_val) / std_val)
                    df_clean[col] = np.where(z_scores > 3, mean_val, df_clean[col])
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    min_rows (int): Minimum number of rows required.
    
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

# Example usage (commented out for production)
# if __name__ == "__main__":
#     sample_data = {
#         'A': [1, 2, np.nan, 4, 100],
#         'B': [5, 6, 7, np.nan, 8],
#         'C': ['a', 'b', 'c', 'd', 'e']
#     }
#     
#     df = pd.DataFrame(sample_data)
#     print("Original DataFrame:")
#     print(df)
#     
#     cleaned_df = clean_dataframe(df, missing_strategy='mean', outlier_method='iqr')
#     print("\nCleaned DataFrame:")
#     print(cleaned_df)
#     
#     is_valid, message = validate_dataframe(cleaned_df, required_columns=['A', 'B', 'C'])
#     print(f"\nValidation: {is_valid}, Message: {message}")