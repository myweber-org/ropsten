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
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True, missing_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to drop duplicate rows
        fill_missing (bool): Whether to fill missing values
        missing_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'constant')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_df)
        print(f"Removed {removed_rows} duplicate rows")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if missing_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif missing_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif missing_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                elif missing_strategy == 'constant':
                    fill_value = 0
                else:
                    raise ValueError(f"Unknown strategy: {missing_strategy}")
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in column '{column}' using {missing_strategy} strategy")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df.empty:
        print("DataFrame is empty")
        return False
    
    if len(df) < min_rows:
        print(f"DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def main():
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', 'C', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    cleaned_df = clean_dataframe(df, drop_duplicates=True, fill_missing=True, missing_strategy='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'], min_rows=1)
    print(f"\nDataFrame validation: {'PASS' if is_valid else 'FAIL'}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, columns=None, factor=1.5):
    """
    Remove outliers from specified columns using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process. If None, process all numeric columns.
    factor (float): Multiplication factor for IQR. Default is 1.5.
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize specified columns using selected method.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to normalize
    method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
    pd.DataFrame: Dataframe with normalized columns
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_norm = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0
                
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                df_norm[col] = (df[col] - mean_val) / std_val
            else:
                df_norm[col] = 0
    
    return df_norm

def handle_missing_values(df, columns=None, strategy='mean'):
    """
    Handle missing values in specified columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    columns (list): List of column names to process
    strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    
    Returns:
    pd.DataFrame: Dataframe with handled missing values
    """
    if columns is None:
        columns = df.columns.tolist()
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df_processed[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df_processed[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df_processed[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
        elif strategy == 'drop':
            df_processed = df_processed.dropna(subset=[col])
    
    return df_processed.reset_index(drop=True)
import pandas as pd
import re

def clean_dataframe(df, column_mapping=None, drop_duplicates=True, text_columns=None):
    """
    Clean a pandas DataFrame by standardizing column names,
    removing duplicates, and cleaning text data.
    """
    cleaned_df = df.copy()
    
    if column_mapping:
        cleaned_df.rename(columns=column_mapping, inplace=True)
    
    if drop_duplicates:
        cleaned_df.drop_duplicates(inplace=True)
        cleaned_df.reset_index(drop=True, inplace=True)
    
    if text_columns:
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(clean_text)
    
    return cleaned_df

def clean_text(text):
    """
    Standardize text by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s-]', '', text)
    
    return text

def validate_email(email):
    """
    Validate email format using regex pattern.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email))) if pd.notna(email) else False
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_data(input_list):
    if not isinstance(input_list, list):
        raise TypeError("Input must be a list")
    return remove_duplicates_preserve_order(input_list)