
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier (default: 1.5)
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    filtered_df = dataframe[(dataframe[column] >= lower_bound) & 
                           (dataframe[column] <= upper_bound)]
    
    return filtered_df.copy()

def z_score_normalize(dataframe, columns=None):
    """
    Apply z-score normalization to specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to normalize (default: all numeric columns)
    
    Returns:
    pd.DataFrame: DataFrame with normalized columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col in result_df.columns and np.issubdtype(result_df[col].dtype, np.number):
            mean_val = result_df[col].mean()
            std_val = result_df[col].std()
            
            if std_val > 0:
                result_df[f"{col}_normalized"] = (result_df[col] - mean_val) / std_val
            else:
                result_df[f"{col}_normalized"] = 0
    
    return result_df

def min_max_scale(dataframe, columns=None, feature_range=(0, 1)):
    """
    Apply min-max scaling to specified columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    columns (list): List of column names to scale (default: all numeric columns)
    feature_range (tuple): Desired range of transformed data (default: (0, 1))
    
    Returns:
    pd.DataFrame: DataFrame with scaled columns
    """
    if columns is None:
        columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    result_df = dataframe.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col in result_df.columns and np.issubdtype(result_df[col].dtype, np.number):
            col_min = result_df[col].min()
            col_max = result_df[col].max()
            
            if col_max > col_min:
                result_df[f"{col}_scaled"] = min_val + (result_df[col] - col_min) * (max_val - min_val) / (col_max - col_min)
            else:
                result_df[f"{col}_scaled"] = min_val
    
    return result_df

def handle_missing_values(dataframe, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'constant', 'drop')
    columns (list): List of column names to process (default: all columns)
    
    Returns:
    pd.DataFrame: DataFrame with handled missing values
    """
    if columns is None:
        columns = dataframe.columns.tolist()
    
    result_df = dataframe.copy()
    
    for col in columns:
        if col not in result_df.columns:
            continue
            
        if result_df[col].isnull().any():
            if strategy == 'drop':
                result_df = result_df.dropna(subset=[col])
            elif strategy == 'mean' and np.issubdtype(result_df[col].dtype, np.number):
                result_df[col] = result_df[col].fillna(result_df[col].mean())
            elif strategy == 'median' and np.issubdtype(result_df[col].dtype, np.number):
                result_df[col] = result_df[col].fillna(result_df[col].median())
            elif strategy == 'mode':
                result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else None)
            elif strategy == 'constant':
                result_df[col] = result_df[col].fillna(0)
    
    return result_df

def validate_dataframe(dataframe, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if len(dataframe) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def create_sample_data():
    """
    Create sample data for testing purposes.
    
    Returns:
    pd.DataFrame: Sample DataFrame with various data types
    """
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.uniform(0, 1, 100),
        'feature_c': np.random.randint(1, 50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    df.loc[np.random.choice(100, 5, replace=False), 'feature_a'] = np.nan
    df.loc[np.random.choice(100, 3, replace=False), 'feature_b'] = np.nan
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    print("Original DataFrame shape:", sample_df.shape)
    
    cleaned_df = handle_missing_values(sample_df, strategy='mean')
    print("After handling missing values:", cleaned_df.shape)
    
    normalized_df = z_score_normalize(cleaned_df, ['feature_a', 'feature_b', 'feature_c'])
    print("After normalization:", normalized_df.shape)
    
    scaled_df = min_max_scale(cleaned_df, ['feature_a', 'feature_b', 'feature_c'])
    print("After scaling:", scaled_df.shape)
    
    filtered_df = remove_outliers_iqr(cleaned_df, 'feature_a')
    print("After outlier removal:", filtered_df.shape)
    
    is_valid, message = validate_dataframe(filtered_df, min_rows=10)
    print(f"Validation result: {is_valid}, Message: {message}")
import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset (list, optional): Column labels to consider for duplicates
        keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def clean_numeric_columns(df, columns):
    """
    Clean numeric columns by converting to appropriate types.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    return True

def get_data_summary(df):
    """
    Generate summary statistics for the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict()
    }
    
    return summary