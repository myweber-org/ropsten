import pandas as pd
import numpy as np

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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    cleaned_file = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file, index=False)
    return cleaned_file

if __name__ == "__main__":
    data_file = "sample_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    try:
        result = clean_dataset(data_file, numeric_cols)
        print(f"Cleaned data saved to: {result}")
    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='drop', fill_value=None, remove_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        missing_strategy (str): Strategy for handling missing values. 
                               Options: 'drop', 'fill', 'interpolate'
        fill_value: Value to fill missing values with if strategy is 'fill'
        remove_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        df_clean = df_clean.dropna()
    elif missing_strategy == 'fill':
        if fill_value is not None:
            df_clean = df_clean.fillna(fill_value)
        else:
            df_clean = df_clean.fillna(0)
    elif missing_strategy == 'interpolate':
        df_clean = df_clean.interpolate()
    
    # Remove duplicates
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    # Reset index after cleaning
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
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

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to check for outliers
        factor (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for column in columns:
        if column in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[column]):
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def normalize_columns(df, columns, method='minmax'):
    """
    Normalize specified columns in DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to normalize
        method (str): Normalization method ('minmax' or 'zscore')
    
    Returns:
        pd.DataFrame: DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    for column in columns:
        if column in df_normalized.columns and pd.api.types.is_numeric_dtype(df_normalized[column]):
            if method == 'minmax':
                min_val = df_normalized[column].min()
                max_val = df_normalized[column].max()
                if max_val > min_val:
                    df_normalized[column] = (df_normalized[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = df_normalized[column].mean()
                std_val = df_normalized[column].std()
                if std_val > 0:
                    df_normalized[column] = (df_normalized[column] - mean_val) / std_val
    
    return df_normalized

def create_summary_report(df):
    """
    Create a summary report of DataFrame statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    report = {
        'shape': df.shape,
        'total_missing': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        report['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        report['categorical_stats'][col] = {
            'unique_count': df[col].nunique(),
            'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'top_count': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        }
    
    return report

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, np.nan, 4, 5, 5, 100],
        'B': [10, 20, 30, np.nan, 50, 50, 60],
        'C': ['a', 'b', 'a', 'b', 'c', 'c', 'd']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    # Clean the data
    df_clean = clean_dataset(df, missing_strategy='fill', fill_value=0)
    print("Cleaned DataFrame:")
    print(df_clean)
    
    # Create summary report
    report = create_summary_report(df_clean)
    print("\nSummary Report:")
    print(f"Shape: {report['shape']}")
    print(f"Missing values: {report['total_missing']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")