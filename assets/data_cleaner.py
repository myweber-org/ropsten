
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: pandas DataFrame
        column: column name to process
        factor: IQR multiplier (default 1.5)
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data.copy()

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling.
    
    Args:
        data: pandas DataFrame
        column: column name to normalize
    
    Returns:
        DataFrame with normalized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (data[column] - min_val) / (max_val - min_val)
    
    result = data.copy()
    result[f"{column}_normalized"] = normalized
    return result

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization.
    
    Args:
        data: pandas DataFrame
        column: column name to standardize
    
    Returns:
        DataFrame with standardized column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        standardized = 0
    else:
        standardized = (data[column] - mean_val) / std_val
    
    result = data.copy()
    result[f"{column}_standardized"] = standardized
    return result

def clean_dataset(data, numeric_columns=None, outlier_factor=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        data: pandas DataFrame
        numeric_columns: list of numeric columns to process
        outlier_factor: IQR multiplier for outlier removal
    
    Returns:
        Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in numeric_columns:
        if column in cleaned_data.columns:
            cleaned_data = remove_outliers_iqr(cleaned_data, column, outlier_factor)
            cleaned_data = normalize_minmax(cleaned_data, column)
            cleaned_data = standardize_zscore(cleaned_data, column)
    
    return cleaned_data

def validate_data(data, required_columns=None, allow_nan=False):
    """
    Validate dataset structure and content.
    
    Args:
        data: pandas DataFrame to validate
        required_columns: list of required column names
        allow_nan: whether NaN values are allowed
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input must be a pandas DataFrame"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    if not allow_nan and data.isnull().any().any():
        return False, "Dataset contains NaN values"
    
    return True, "Dataset is valid"

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    
    cleaned = clean_dataset(sample_data, ['feature1', 'feature2'])
    print("Cleaned data shape:", cleaned.shape)
    
    is_valid, message = validate_data(cleaned)
    print(f"Validation: {is_valid} - {message}")
    
    print("\nCleaned data columns:", cleaned.columns.tolist())
    print("\nFirst 5 rows:")
    print(cleaned.head())
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by handling duplicates and missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows. Default True.
    fill_missing (str): Strategy to fill missing values. 
                        Options: 'mean', 'median', 'mode', or 'drop'. Default 'mean'.
    
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
        print("Dropped rows with missing values.")
    elif fill_missing in ['mean', 'median', 'mode']:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' with {fill_missing}: {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for column in categorical_cols:
        if cleaned_df[column].isnull().any():
            most_frequent = cleaned_df[column].mode()[0]
            cleaned_df[column] = cleaned_df[column].fillna(most_frequent)
            print(f"Filled missing categorical values in '{column}' with mode: '{most_frequent}'")
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    min_rows (int): Minimum number of rows required.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame.")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows.")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    print("DataFrame validation passed.")
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, np.nan],
        'B': [5, np.nan, 7, 8, 9],
        'C': ['x', 'y', 'y', np.nan, 'z'],
        'D': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    if validate_dataframe(df, required_columns=['A', 'B', 'C', 'D']):
        cleaned = clean_dataset(df, drop_duplicates=True, fill_missing='median')
        print("\nCleaned DataFrame:")
        print(cleaned)
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    drop_duplicates (bool): Whether to drop duplicate rows
    fill_missing (bool): Whether to fill missing values
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        print(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values
    if fill_missing:
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            for column in cleaned_df.columns:
                if cleaned_df[column].isnull().any():
                    if cleaned_df[column].dtype in ['int64', 'float64']:
                        if fill_strategy == 'mean':
                            fill_value = cleaned_df[column].mean()
                        elif fill_strategy == 'median':
                            fill_value = cleaned_df[column].median()
                        elif fill_strategy == 'mode':
                            fill_value = cleaned_df[column].mode()[0]
                        elif fill_strategy == 'zero':
                            fill_value = 0
                        else:
                            fill_value = cleaned_df[column].mean()
                        
                        cleaned_df[column].fillna(fill_value, inplace=True)
                    else:
                        # For non-numeric columns, fill with most frequent value
                        fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 'Unknown'
                        cleaned_df[column].fillna(fill_value, inplace=True)
            
            print(f"Missing values filled using '{fill_strategy}' strategy")
    
    # Additional cleaning: remove columns with too many missing values
    threshold = 0.5  # Remove columns with more than 50% missing values
    columns_to_drop = []
    
    for column in cleaned_df.columns:
        missing_ratio = cleaned_df[column].isnull().sum() / len(cleaned_df)
        if missing_ratio > threshold:
            columns_to_drop.append(column)
    
    if columns_to_drop:
        cleaned_df = cleaned_df.drop(columns=columns_to_drop)
        print(f"Removed columns with >{threshold*100}% missing values: {columns_to_drop}")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Dataset cleaned. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate the dataset for basic integrity checks.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing required columns: {missing_columns}')
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Column {col} contains infinite values')
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0, 50.0],
        'category': ['A', 'B', 'B', 'C', np.nan, 'A', 'A'],
        'score': [100, 200, 200, 300, 400, 500, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean')
    
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned dataset
    validation = validate_dataset(cleaned_df, required_columns=['id', 'value', 'category'])
    print(f"\nValidation results: {validation}")
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from a DataFrame column using IQR method.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    threshold (float): IQR multiplier for outlier detection
    
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
    return filtered_df

def normalize_column_zscore(dataframe, column):
    """
    Normalize a column using Z-score normalization.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to normalize
    
    Returns:
    pd.DataFrame: DataFrame with normalized column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    normalized_df = dataframe.copy()
    mean_val = normalized_df[column].mean()
    std_val = normalized_df[column].std()
    
    if std_val > 0:
        normalized_df[f'{column}_normalized'] = (normalized_df[column] - mean_val) / std_val
    else:
        normalized_df[f'{column}_normalized'] = 0
    
    return normalized_df

def detect_skewed_columns(dataframe, skew_threshold=0.5):
    """
    Detect columns with significant skewness.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    skew_threshold (float): Absolute skewness threshold
    
    Returns:
    dict: Dictionary of column names and their skewness values
    """
    skewed_columns = {}
    
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        skewness = dataframe[col].skew()
        if abs(skewness) > skew_threshold:
            skewed_columns[col] = skewness
    
    return skewed_columns

def apply_log_transform(dataframe, column):
    """
    Apply log transformation to reduce skewness.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    column (str): Column name to transform
    
    Returns:
    pd.DataFrame: DataFrame with transformed column
    """
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    transformed_df = dataframe.copy()
    
    # Add small constant to handle zero or negative values
    min_val = transformed_df[column].min()
    if min_val <= 0:
        constant = abs(min_val) + 1
        transformed_df[f'{column}_log'] = np.log(transformed_df[column] + constant)
    else:
        transformed_df[f'{column}_log'] = np.log(transformed_df[column])
    
    return transformed_df

def clean_dataset(dataframe, numeric_columns=None, outlier_threshold=1.5):
    """
    Comprehensive data cleaning pipeline.
    
    Parameters:
    dataframe (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric columns to process
    outlier_threshold (float): IQR threshold for outlier removal
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if numeric_columns is None:
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = dataframe.copy()
    
    # Remove outliers from each numeric column
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
    
    # Normalize numeric columns
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = normalize_column_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_dataframe(dataframe, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    dataframe (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    tuple: (is_valid, validation_message)
    """
    if not isinstance(dataframe, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if dataframe.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    # Check for infinite values
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(dataframe[numeric_cols]).sum().sum()
    if inf_count > 0:
        return False, f"Found {inf_count} infinite values in numeric columns"
    
    return True, "DataFrame validation passed"

# Example usage demonstration
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'feature_a': np.random.normal(100, 15, 1000),
        'feature_b': np.random.exponential(50, 1000),
        'feature_c': np.random.uniform(0, 1, 1000)
    }
    
    # Add some outliers
    sample_data['feature_a'][:10] = 500
    sample_data['feature_b'][:5] = 1000
    
    df = pd.DataFrame(sample_data)
    
    # Validate data
    is_valid, message = validate_dataframe(df)
    print(f"Validation: {is_valid} - {message}")
    
    # Detect skewed columns
    skewed = detect_skewed_columns(df)
    print(f"Skewed columns: {skewed}")
    
    # Clean data
    cleaned_df = clean_dataset(df, ['feature_a', 'feature_b', 'feature_c'])
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Columns after cleaning: {cleaned_df.columns.tolist()}")