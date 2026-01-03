import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val - min_val == 0:
        return df[column].apply(lambda x: 0.0)
    return (df[column] - min_val) / (max_val - min_val)

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_dataframe(df):
    required_checks = [
        (lambda x: isinstance(x, pd.DataFrame), "Input must be a pandas DataFrame"),
        (lambda x: not x.empty, "DataFrame cannot be empty"),
        (lambda x: x.isnull().sum().sum() == 0, "DataFrame contains null values")
    ]
    
    for check, message in required_checks:
        if not check(df):
            raise ValueError(message)
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(2.0, 200),
        'category': np.random.choice(['X', 'Y', 'Z'], 200)
    })
    
    try:
        validate_dataframe(sample_data)
        cleaned = clean_dataset(sample_data, ['feature_a', 'feature_b'])
        print(f"Original shape: {sample_data.shape}")
        print(f"Cleaned shape: {cleaned.shape}")
        print("Data cleaning completed successfully.")
    except ValueError as e:
        print(f"Validation error: {e}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to remove duplicate rows.
    fill_missing (str): Method to fill missing values ('mean', 'median', 'mode', or 'drop').
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
        print("Removed rows with missing values.")
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        print(f"Filled missing numeric values with {fill_missing}.")
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
        print("Filled missing values with mode.")
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if len(df) < min_rows:
        print(f"Validation failed: DataFrame has less than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Validation failed: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, None, 10, 20, 30, 30],
        'C': ['x', 'y', 'x', 'z', None, 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaning data...")
    
    cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    validation_result = validate_data(cleaned, required_columns=['A', 'B'], min_rows=1)
    print(f"\nValidation result: {validation_result}")import pandas as pd
import numpy as np

def clean_dataset(df, column_mapping=None, drop_duplicates=True, normalize_text=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing text columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    column_mapping (dict, optional): Dictionary mapping original column names to new names.
    drop_duplicates (bool): Whether to remove duplicate rows.
    normalize_text (bool): Whether to normalize text columns (strip, lower case).
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if column_mapping:
        df_clean = df_clean.rename(columns=column_mapping)
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows.")
    
    if normalize_text:
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        print(f"Normalized text in columns: {list(text_columns)}")
    
    df_clean = df_clean.replace(['none', 'null', 'nan', ''], np.nan)
    
    return df_clean

def validate_data(df, required_columns=None, unique_columns=None):
    """
    Validate the cleaned DataFrame for required columns and uniqueness.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    unique_columns (list): List of column names that should have unique values.
    
    Returns:
    dict: Dictionary containing validation results.
    """
    validation_results = {}
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing
        if missing:
            print(f"Warning: Missing required columns: {missing}")
    
    if unique_columns:
        for col in unique_columns:
            if col in df.columns:
                unique_count = df[col].nunique()
                total_count = len(df[col])
                if unique_count < total_count:
                    validation_results[f'duplicates_in_{col}'] = total_count - unique_count
                    print(f"Warning: Column '{col}' has {total_count - unique_count} duplicate values.")
    
    validation_results['total_rows'] = len(df)
    validation_results['total_columns'] = len(df.columns)
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'Name': ['John Doe', 'Jane Smith', 'John Doe', ' Bob Johnson ', 'ALICE'],
        'Email': ['john@example.com', 'jane@example.com', 'john@example.com', 'bob@example.com', 'alice@example.com'],
        'Age': [25, 30, 25, 35, 28],
        'City': ['New York', 'Los Angeles', 'New York', ' Chicago ', 'BOSTON']
    }
    
    df_raw = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df_raw)
    print("\n" + "="*50 + "\n")
    
    df_cleaned = clean_dataset(df_raw, drop_duplicates=True, normalize_text=True)
    print("\nCleaned DataFrame:")
    print(df_cleaned)
    
    validation = validate_data(df_cleaned, required_columns=['Name', 'Email'], unique_columns=['Email'])
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")