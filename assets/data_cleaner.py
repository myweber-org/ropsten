
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        if col in cleaned_df.columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    return cleaned_df

def normalize_data(df, columns, method='minmax'):
    normalized_df = df.copy()
    for col in columns:
        if col in normalized_df.columns:
            if method == 'minmax':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val != min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val != 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
    return normalized_df

def clean_dataset(df, numeric_columns):
    df_cleaned = df.dropna(subset=numeric_columns)
    df_no_outliers = remove_outliers_iqr(df_cleaned, numeric_columns)
    df_normalized = normalize_data(df_no_outliers, numeric_columns, method='zscore')
    return df_normalized

if __name__ == "__main__":
    sample_data = {
        'feature_a': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10],
        'feature_b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    }
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature_a', 'feature_b']
    
    cleaned_df = clean_dataset(df, numeric_cols)
    print("Original dataset shape:", df.shape)
    print("Cleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned data summary:")
    print(cleaned_df[numeric_cols].describe())
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                           Options: 'mean', 'median', 'drop', 'fill_zero'.
    outlier_threshold (float): Number of standard deviations to identify outliers.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'fill_zero':
        cleaned_df = cleaned_df.fillna(0)
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Handle outliers using z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        outliers = z_scores > outlier_threshold
        
        if outliers.any():
            # Replace outliers with column median
            col_median = cleaned_df[col].median()
            cleaned_df.loc[outliers, col] = col_median
    
    # Reset index if rows were dropped
    if missing_strategy == 'drop' and len(cleaned_df) != len(df):
        cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate that the DataFrame meets basic requirements.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # 100 is an outlier
        'B': [5, np.nan, 7, 8, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame info:")
    print(df.info())
    
    # Clean the data
    cleaned = clean_dataset(df, missing_strategy='median', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nData validation passed: {is_valid}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column in a dataset using the IQR method.
    
    Parameters:
    data (numpy.ndarray): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    numpy.ndarray: Dataset with outliers removed from the specified column.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    
    return data[mask]import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): Whether to drop duplicate rows.
    fill_method (str or None): Method to fill missing values. 
                               Options: 'mean', 'median', 'mode', or None.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is not None:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        
        if fill_method == 'mean':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
                cleaned_df[numeric_cols].mean()
            )
        elif fill_method == 'median':
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
                cleaned_df[numeric_cols].median()
            )
        elif fill_method == 'mode':
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].fillna(
                        cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else ''
                    )
    
    # Remove duplicates
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        if removed > 0:
            print(f"Removed {removed} duplicate row(s)")
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 3, 4, None],
        'name': ['Alice', 'Bob', 'Charlie', 'Charlie', 'David', None],
        'score': [85.5, 92.0, None, 78.5, 88.0, 90.5],
        'age': [25, 30, 35, 35, 28, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nShape:", df.shape)
    
    # Clean the data
    cleaned = clean_dataset(df, drop_duplicates=True, fill_method='mean')
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nShape after cleaning:", cleaned.shape)
    
    # Validate
    is_valid, message = validate_dataframe(cleaned, required_columns=['id', 'name', 'score'])
    print(f"\nValidation: {message}")import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_summary_statistics(data, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': data[column].mean(),
        'median': data[column].median(),
        'std': data[column].std(),
        'min': data[column].min(),
        'max': data[column].max(),
        'count': data[column].count()
    }
    
    return stats

def clean_dataset(data, columns_to_clean=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns_to_clean is None:
        columns_to_clean = data.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column in cleaned_data.columns and np.issubdtype(cleaned_data[column].dtype, np.number):
            original_count = len(cleaned_data)
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
            removed_count = original_count - len(cleaned_data)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_data

if __name__ == "__main__":
    import pandas as pd
    
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 1000),
        'B': np.random.exponential(50, 1000),
        'C': np.random.uniform(0, 200, 1000)
    })
    
    print("Original dataset shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['A', 'B'])
    print("Cleaned dataset shape:", cleaned.shape)
    
    stats = calculate_summary_statistics(cleaned, 'A')
    print("\nSummary statistics for column A:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")import numpy as np
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
    if max_val == min_val:
        return df[column].apply(lambda x: 0.0)
    return df[column].apply(lambda x: (x - min_val) / (max_val - min_val))

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df[col] = normalize_minmax(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV. If None, returns DataFrame
        missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop', 'zero')
    
    Returns:
        pd.DataFrame or None: Cleaned DataFrame if output_path is None, else None
    """
    
    try:
        df = pd.read_csv(file_path)
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if missing_strategy == 'mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif missing_strategy == 'median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif missing_strategy == 'zero':
            df = df.fillna(0)
        elif missing_strategy == 'drop':
            df = df.dropna()
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        print(f"Data cleaning completed:")
        print(f"  - Removed {duplicates_removed} duplicate rows")
        print(f"  - Missing values handled using '{missing_strategy}' strategy")
        print(f"  - Final dataset shape: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"  - Cleaned data saved to: {output_path}")
            return None
        else:
            return df
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list, optional): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append('DataFrame is empty or None')
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Missing required columns: {missing_cols}')
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            validation_results['warnings'].append(f'Column {col} contains infinite values')
    
    # Check data types
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                validation_results['warnings'].append(f'Column {col} has high cardinality ({unique_ratio:.1%} unique values)')
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10.5, None, 15.2, 20.1, None, 10.5],
        'category': ['A', 'B', 'A', 'C', 'B', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', missing_strategy='mean')
    
    if cleaned_df is not None:
        validation = validate_dataframe(cleaned_df, required_columns=['id', 'value'])
        print(f"Validation results: {validation}")