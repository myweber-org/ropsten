
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to process
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
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

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to normalize
    
    Returns:
        pd.DataFrame: DataFrame with normalized column
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    df_copy = df.copy()
    min_val = df_copy[column].min()
    max_val = df_copy[column].max()
    
    if max_val == min_val:
        df_copy[column] = 0.5
    else:
        df_copy[column] = (df_copy[column] - min_val) / (max_val - min_val)
    
    return df_copy

def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df_copy[col].isnull().any():
            if strategy == 'mean':
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
            elif strategy == 'median':
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif strategy == 'mode':
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_copy

def clean_dataset(df, numeric_columns=None, outlier_removal=True, normalization=True, missing_strategy='mean'):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of numeric columns to process
        outlier_removal (bool): Whether to remove outliers
        normalization (bool): Whether to normalize columns
        missing_strategy (str): Strategy for handling missing values
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_copy = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    df_copy = handle_missing_values(df_copy, strategy=missing_strategy)
    
    if outlier_removal:
        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy = remove_outliers_iqr(df_copy, col)
    
    if normalization:
        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy = normalize_column(df_copy, col)
    
    return df_copy
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__ == "__main__":
    sample_list = [1, 2, 2, 3, 4, 4, 5, 1, 6]
    cleaned_list = remove_duplicates(sample_list)
    print(f"Original list: {sample_list}")
    print(f"List after removing duplicates: {cleaned_list}")
import pandas as pd
import re

def clean_dataframe(df, columns_to_normalize=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df: pandas DataFrame to clean
        columns_to_normalize: list of column names to normalize (default: all object columns)
        remove_duplicates: whether to remove duplicate rows (default: True)
    
    Returns:
        Cleaned pandas DataFrame
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    if columns_to_normalize is None:
        columns_to_normalize = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in columns_to_normalize:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].apply(normalize_string)
            print(f"Normalized column: {col}")
    
    return df_clean

def normalize_string(text):
    """
    Normalize a string by converting to lowercase, removing extra whitespace,
    and stripping special characters.
    
    Args:
        text: string to normalize
    
    Returns:
        Normalized string
    """
    if not isinstance(text, str):
        return text
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[^\w\s-]', '', text)
    
    return text

def validate_email(email):
    """
    Validate email format using regex pattern.
    
    Args:
        email: email string to validate
    
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) if isinstance(email, str) else False

def sample_usage():
    """
    Demonstrate usage of the data cleaning functions.
    """
    data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson  '],
        'email': ['john@example.com', 'jane@example.com', 'john@example.com', 'invalid-email'],
        'age': [25, 30, 25, 35]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    df_clean = clean_dataframe(df)
    print("Cleaned DataFrame:")
    print(df_clean)
    print("\n")
    
    df_clean['email_valid'] = df_clean['email'].apply(validate_email)
    print("DataFrame with email validation:")
    print(df_clean)

if __name__ == "__main__":
    sample_usage()
import numpy as np
import pandas as pd
from scipy import stats

def detect_outliers_iqr(data, column, threshold=1.5):
    """
    Detect outliers using IQR method
    """
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method
    """
    z_scores = np.abs(stats.zscore(data[column]))
    filtered_data = data[z_scores < threshold]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    min_val = data[column].min()
    max_val = data[column].max()
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_data(data, column):
    """
    Standardize data using Z-score normalization
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns, outlier_method='zscore', normalize=False):
    """
    Main cleaning function for datasets
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if col not in cleaned_df.columns:
            continue
            
        # Handle outliers
        if outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        elif outlier_method == 'iqr':
            outliers = detect_outliers_iqr(cleaned_df, col)
            cleaned_df = cleaned_df.drop(outliers.index)
        
        # Normalize if requested
        if normalize:
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[f'{col}_standardized'] = standardize_data(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, min_rows=10):
    """
    Validate dataset structure and content
    """
    validation_results = {
        'has_required_columns': all(col in df.columns for col in required_columns),
        'has_sufficient_rows': len(df) >= min_rows,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_results

def example_usage():
    """
    Example usage of the data cleaning utilities
    """
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add some outliers
    df.loc[0, 'feature_a'] = 500
    df.loc[1, 'feature_b'] = 1000
    
    # Clean the data
    numeric_cols = ['feature_a', 'feature_b']
    cleaned_df = clean_dataset(df, numeric_cols, outlier_method='zscore', normalize=True)
    
    # Validate
    validation = validate_data(cleaned_df, numeric_cols)
    
    return cleaned_df, validation

if __name__ == "__main__":
    cleaned_data, validation_results = example_usage()
    print(f"Original shape: 100 rows")
    print(f"Cleaned shape: {len(cleaned_data)} rows")
    print(f"Validation: {validation_results}")