
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
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to analyze
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': len(df[column])
    }
    
    return stats

def process_dataframe(df, columns_to_clean):
    """
    Process multiple columns for outlier removal and return cleaned DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_clean (list): List of column names to clean
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 27, 28, 29, 30, 100],
        'humidity': [45, 46, 47, 48, 49, 50, 51, 52, 53, 200],
        'pressure': [1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary before cleaning:")
    for col in df.columns:
        stats = calculate_summary_statistics(df, col)
        print(f"{col}: {stats}")
    
    cleaned_df = process_dataframe(df, ['temperature', 'humidity', 'pressure'])
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nSummary after cleaning:")
    for col in cleaned_df.columns:
        stats = calculate_summary_statistics(cleaned_df, col)
        print(f"{col}: {stats}")
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', drop_duplicates=True):
    """
    Clean a pandas DataFrame by handling missing values and removing duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
    drop_duplicates (bool): Whether to remove duplicate rows
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    
    df_clean = df.copy()
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    for column in df_clean.columns:
        if df_clean[column].isnull().any():
            missing_count = df_clean[column].isnull().sum()
            print(f"Column '{column}' has {missing_count} missing values")
            
            if missing_strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[column]):
                fill_value = df_clean[column].mean()
                df_clean[column] = df_clean[column].fillna(fill_value)
                print(f"  Filled with mean: {fill_value:.2f}")
                
            elif missing_strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[column]):
                fill_value = df_clean[column].median()
                df_clean[column] = df_clean[column].fillna(fill_value)
                print(f"  Filled with median: {fill_value:.2f}")
                
            elif missing_strategy == 'mode':
                fill_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else None
                if fill_value is not None:
                    df_clean[column] = df_clean[column].fillna(fill_value)
                    print(f"  Filled with mode: {fill_value}")
                else:
                    df_clean[column] = df_clean[column].fillna('Unknown')
                    
            elif missing_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[column])
                print(f"  Dropped rows with missing values in column '{column}'")
                
            else:
                df_clean[column] = df_clean[column].fillna('Unknown')
                print(f"  Filled with 'Unknown'")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    if not isinstance(df, pd.DataFrame):
        validation_results['is_valid'] = False
        validation_results['issues'].append("Input is not a pandas DataFrame")
        return validation_results
    
    validation_results['summary']['total_rows'] = len(df)
    validation_results['summary']['total_columns'] = len(df.columns)
    validation_results['summary']['columns'] = list(df.columns)
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
    
    for column in df.columns:
        null_count = df[column].isnull().sum()
        unique_count = df[column].nunique()
        dtype = str(df[column].dtype)
        
        column_stats = {
            'null_count': null_count,
            'unique_count': unique_count,
            'dtype': dtype
        }
        
        if pd.api.types.is_numeric_dtype(df[column]):
            column_stats['min'] = float(df[column].min())
            column_stats['max'] = float(df[column].max())
            column_stats['mean'] = float(df[column].mean())
        
        validation_results['summary'][column] = column_stats
        
        if null_count > 0:
            validation_results['issues'].append(f"Column '{column}' has {null_count} null values")
    
    return validation_results

def example_usage():
    """Example usage of the data cleaning functions."""
    
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5, 6],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Eve', None],
        'age': [25, 30, None, 35, 40, 40, 45],
        'score': [85.5, 92.0, 78.5, None, 88.0, 88.0, 95.5],
        'department': ['HR', 'IT', 'IT', 'Finance', None, None, 'HR']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataframe(df, required_columns=['id', 'name', 'age'])
    print("Validation Results:")
    print(f"Is valid: {validation['is_valid']}")
    if validation['issues']:
        print("Issues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataframe(df, missing_strategy='mean', drop_duplicates=True)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = example_usage()
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_minmax(df, columns):
    normalized_df = df.copy()
    for col in columns:
        min_val = normalized_df[col].min()
        max_val = normalized_df[col].max()
        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    return normalized_df

def handle_missing_values(df, strategy='mean'):
    processed_df = df.copy()
    for col in processed_df.select_dtypes(include=[np.number]).columns:
        if strategy == 'mean':
            processed_df[col].fillna(processed_df[col].mean(), inplace=True)
        elif strategy == 'median':
            processed_df[col].fillna(processed_df[col].median(), inplace=True)
        elif strategy == 'mode':
            processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
    return processed_df

def clean_data(input_file, output_file, numeric_columns):
    df = load_dataset(input_file)
    df = handle_missing_values(df, strategy='median')
    df = remove_outliers_iqr(df, numeric_columns)
    df = normalize_minmax(df, numeric_columns)
    df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    numeric_cols = ['age', 'income', 'score']
    cleaned_data = clean_data('raw_data.csv', 'cleaned_data.csv', numeric_cols)
    print(f"Data cleaning completed. Shape: {cleaned_data.shape}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(data, column, multiplier=1.5):
    """
    Remove outliers using IQR method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize data using min-max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if min_val == max_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def clean_dataset(df, numeric_columns=None, outlier_multiplier=1.5):
    """
    Clean dataset by handling outliers and normalizing numeric columns
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    stats = {}
    
    for col in numeric_columns:
        if col in df.columns:
            # Remove outliers
            cleaned_df, outliers_removed = remove_outliers_iqr(cleaned_df, col, outlier_multiplier)
            stats[f'{col}_outliers_removed'] = outliers_removed
            
            # Normalize
            cleaned_df[f'{col}_normalized'] = normalize_minmax(cleaned_df, col)
            cleaned_df[f'{col}_standardized'] = standardize_zscore(cleaned_df, col)
    
    return cleaned_df, stats

def validate_data(df, required_columns=None, allow_nan_columns=None):
    """
    Validate data structure and content
    """
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    if allow_nan_columns is None:
        allow_nan_columns = []
    
    nan_columns = [col for col in df.columns 
                   if df[col].isnull().any() and col not in allow_nan_columns]
    
    if nan_columns:
        validation_results['columns_with_unexpected_nan'] = nan_columns
    
    return validation_results

def example_usage():
    """
    Example usage of the data cleaning utilities
    """
    # Create sample data
    np.random.seed(42)
    data = {
        'id': range(100),
        'feature_a': np.random.normal(100, 15, 100),
        'feature_b': np.random.exponential(50, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    
    # Add some outliers
    df.loc[10, 'feature_a'] = 500
    df.loc[20, 'feature_b'] = 1000
    
    # Clean the data
    cleaned_df, stats = clean_dataset(df, numeric_columns=['feature_a', 'feature_b'])
    
    # Validate
    validation = validate_data(cleaned_df, required_columns=['feature_a', 'feature_b'])
    
    return cleaned_df, stats, validation

if __name__ == "__main__":
    cleaned_data, cleaning_stats, validation_results = example_usage()
    print(f"Data cleaning completed:")
    print(f"Original shape: (100, 4)")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(f"Outliers removed: {cleaning_stats}")
    print(f"Validation results: {validation_results}")
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result