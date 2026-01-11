
def remove_duplicates(input_list):
    """
    Remove duplicate elements from a list while preserving order.
    Returns a new list with unique elements.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def clean_numeric_strings(string_list):
    """
    Clean a list of strings by converting numeric strings to integers.
    Non-numeric strings are kept as-is.
    """
    cleaned = []
    for s in string_list:
        s = s.strip()
        if s.isdigit():
            cleaned.append(int(s))
        else:
            cleaned.append(s)
    return cleaned

def filter_by_type(data_list, data_type):
    """
    Filter a list to include only elements of a specific type.
    """
    return [item for item in data_list if isinstance(item, data_type)]

def main():
    # Example usage
    sample_data = [1, 2, 2, 3, '4', '4', 'five', 5.0, 5.0]
    
    print("Original data:", sample_data)
    
    unique_data = remove_duplicates(sample_data)
    print("After removing duplicates:", unique_data)
    
    cleaned_data = clean_numeric_strings([str(x) for x in unique_data])
    print("After cleaning numeric strings:", cleaned_data)
    
    integers_only = filter_by_type(cleaned_data, int)
    print("Integers only:", integers_only)

if __name__ == "__main__":
    main()
import pandas as pd

def clean_dataset(df, missing_strategy='drop', duplicate_strategy='drop_first'):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        'drop': Drop rows with any missing values.
        'fill_mean': Fill missing values with column mean (numeric only).
        'fill_median': Fill missing values with column median (numeric only).
        'fill_mode': Fill missing values with column mode.
    duplicate_strategy (str): Strategy for handling duplicate rows.
        'drop_first': Drop duplicates keeping first occurrence.
        'drop_last': Drop duplicates keeping last occurrence.
        'keep_none': Drop all duplicates.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'fill_mean':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
    elif missing_strategy == 'fill_median':
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
    elif missing_strategy == 'fill_mode':
        for col in cleaned_df.columns:
            mode_val = cleaned_df[col].mode()
            if not mode_val.empty:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val.iloc[0])
    
    # Handle duplicates
    if duplicate_strategy == 'drop_first':
        cleaned_df = cleaned_df.drop_duplicates(keep='first')
    elif duplicate_strategy == 'drop_last':
        cleaned_df = cleaned_df.drop_duplicates(keep='last')
    elif duplicate_strategy == 'keep_none':
        cleaned_df = cleaned_df.drop_duplicates(keep=False)
    
    return cleaned_df

def validate_dataset(df, required_columns=None, min_rows=1):
    """
    Validate a DataFrame for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    tuple: (is_valid, message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has fewer than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "Dataset is valid"
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    strategy (str): Strategy for imputation ('mean', 'median', 'mode', 'drop').
    columns (list): List of columns to process. If None, process all numeric columns.
    
    Returns:
    pd.DataFrame: DataFrame with missing values handled.
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif strategy == 'median':
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif strategy == 'mode':
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to normalize.
    
    Returns:
    pd.DataFrame: DataFrame with normalized column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_normalized = df.copy()
    min_val = df_normalized[column].min()
    max_val = df_normalized[column].max()
    
    if max_val == min_val:
        df_normalized[column] = 0.5
    else:
        df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
    
    return df_normalized

def get_data_summary(df):
    """
    Generate a summary statistics DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: Summary statistics for numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    summary = pd.DataFrame({
        'mean': numeric_df.mean(),
        'median': numeric_df.median(),
        'std': numeric_df.std(),
        'min': numeric_df.min(),
        'max': numeric_df.max(),
        'missing': numeric_df.isnull().sum(),
        'count': numeric_df.count()
    })
    
    return summary

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 60],
        'C': [1.1, 2.2, np.nan, 4.4, 5.5, 6.6]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    df_clean = remove_outliers_iqr(df, 'A')
    print("DataFrame after removing outliers from column 'A':")
    print(df_clean)
    print()
    
    df_filled = clean_missing_values(df, strategy='mean')
    print("DataFrame after handling missing values:")
    print(df_filled)
    print()
    
    df_normalized = normalize_column(df_filled, 'B')
    print("DataFrame with normalized column 'B':")
    print(df_normalized)
    print()
    
    summary = get_data_summary(df)
    print("Data summary:")
    print(summary)