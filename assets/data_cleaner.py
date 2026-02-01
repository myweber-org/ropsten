import pandas as pd
import re

def clean_text_column(df, column_name):
    """
    Standardize text by converting to lowercase and removing extra whitespace.
    """
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str).str.lower()
        df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    return df

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def process_dataframe(df, text_columns=None, dedupe_subset=None):
    """
    Main function to clean text columns and remove duplicates.
    """
    if text_columns:
        for col in text_columns:
            df = clean_text_column(df, col)
    
    if dedupe_subset:
        df = remove_duplicates(df, subset=dedupe_subset)
    else:
        df = remove_duplicates(df)
    
    return df

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'bob'],
        'email': ['alice@test.com', 'bob@test.com', 'alice@test.com', 'charlie@test.com', 'Bob@Test.Com']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    processed_df = process_dataframe(
        df, 
        text_columns=['name', 'email'], 
        dedupe_subset=['email']
    )
    
    print("\nProcessed DataFrame:")
    print(processed_df)
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        drop_duplicates (bool): Whether to drop duplicate rows. Default is True.
        fill_missing (bool): Whether to fill missing values. Default is True.
        fill_value: Value to use for filling missing values. Default is 0.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing:
        cleaned_df = cleaned_df.fillna(fill_value)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        bool: True if validation passes, False otherwise.
    """
    if df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4],
        'value': [10, 20, 20, None, 40],
        'category': ['A', 'B', 'B', 'C', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataset(df)
    print(cleaned)
    print(f"\nData validation result: {validate_dataframe(cleaned, ['id', 'value'])}")
def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df

def normalize_minmax(dataframe, columns=None):
    """
    Normalize data using min-max scaling
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns
    
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and np.issubdtype(dataframe[col].dtype, np.number):
            min_val = dataframe[col].min()
            max_val = dataframe[col].max()
            
            if max_val > min_val:
                normalized_df[col] = (dataframe[col] - min_val) / (max_val - min_val)
    
    return normalized_df

def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Detect columns with significant skewness
    """
    skewed_cols = []
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = stats.skew(dataframe[col].dropna())
        if abs(skewness) > threshold:
            skewed_cols.append((col, skewness))
    
    return sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values with different strategies
    """
    if columns is None:
        columns = dataframe.columns
    
    processed_df = dataframe.copy()
    
    for col in columns:
        if col in dataframe.columns and dataframe[col].isnull().any():
            if strategy == 'mean' and np.issubdtype(dataframe[col].dtype, np.number):
                processed_df[col] = dataframe[col].fillna(dataframe[col].mean())
            elif strategy == 'median' and np.issubdtype(dataframe[col].dtype, np.number):
                processed_df[col] = dataframe[col].fillna(dataframe[col].median())
            elif strategy == 'mode':
                processed_df[col] = dataframe[col].fillna(dataframe[col].mode()[0])
            elif strategy == 'drop':
                processed_df = processed_df.dropna(subset=[col])
    
    return processed_df

def create_data_summary(dataframe):
    """
    Create comprehensive data summary
    """
    summary = {
        'shape': dataframe.shape,
        'missing_values': dataframe.isnull().sum().to_dict(),
        'data_types': dataframe.dtypes.to_dict(),
        'numeric_stats': {}
    }
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': dataframe[col].mean(),
            'std': dataframe[col].std(),
            'min': dataframe[col].min(),
            'max': dataframe[col].max(),
            'median': dataframe[col].median()
        }
    
    return summary