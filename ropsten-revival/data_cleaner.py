import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing=False, fill_value=0):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    drop_duplicates (bool): Whether to remove duplicate rows
    fill_missing (bool): Whether to fill missing values
    fill_value: Value to use for filling missing data
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        missing_before = cleaned_df.isnull().sum().sum()
        cleaned_df = cleaned_df.fillna(fill_value)
        missing_after = cleaned_df.isnull().sum().sum()
        print(f"Filled {missing_before - missing_after} missing values")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("Dataset is empty")
        return False
    
    return True

def get_dataset_summary(df):
    """
    Generate summary statistics for the dataset.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    }
    
    return summary
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None, subset=None):
    """
    Remove duplicate rows from a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str, optional): Path to save the cleaned CSV file.
                                     If None, overwrites the input file.
        subset (list, optional): List of column names to consider for duplicates.
    
    Returns:
        int: Number of duplicates removed.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        
        if subset:
            df_clean = df.drop_duplicates(subset=subset, keep='first')
        else:
            df_clean = df.drop_duplicates(keep='first')
        
        final_count = len(df_clean)
        duplicates_removed = initial_count - final_count
        
        if output_file is None:
            output_file = input_file
        
        df_clean.to_csv(output_file, index=False)
        
        print(f"Processed: {input_file}")
        print(f"Initial rows: {initial_count}")
        print(f"Final rows: {final_count}")
        print(f"Duplicates removed: {duplicates_removed}")
        
        return duplicates_removed
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return -1
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return -1
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return -1

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        print("Example: python data_cleaner.py data.csv cleaned_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_duplicates(input_file, output_file)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from scipy import stats

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

def clean_dataset(file_path, columns_to_clean):
    df = pd.read_csv(file_path)
    
    for column in columns_to_clean:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file_path, index=False)
    return cleaned_file_path

def calculate_statistics(df):
    stats_dict = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        stats_dict[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'skewness': stats.skew(df[col].dropna()),
            'kurtosis': stats.kurtosis(df[col].dropna())
        }
    
    return stats_dict

if __name__ == "__main__":
    sample_data = {
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(60, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000)
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_sensor_data.csv', index=False)
    
    cleaned_file = clean_dataset('sample_sensor_data.csv', ['temperature', 'humidity', 'pressure'])
    cleaned_df = pd.read_csv(cleaned_file)
    
    statistics = calculate_statistics(cleaned_df)
    for col, stats_values in statistics.items():
        print(f"Statistics for {col}: {stats_values}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, missing_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (bool): Whether to fill missing values
        missing_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'zero')
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if missing_strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif missing_strategy == 'median':
                    fill_value = cleaned_df[col].median()
                elif missing_strategy == 'mode':
                    fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
                elif missing_strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = cleaned_df[col].mean()
                
                missing_count = cleaned_df[col].isnull().sum()
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled {missing_count} missing values in column '{col}' with {fill_value:.2f}")
    
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if cleaned_df[col].isnull().any():
            missing_count = cleaned_df[col].isnull().sum()
            cleaned_df[col] = cleaned_df[col].fillna('Unknown')
            print(f"Filled {missing_count} missing values in categorical column '{col}' with 'Unknown'")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
    
    if df.isnull().sum().sum() > 0:
        total_missing = df.isnull().sum().sum()
        validation_results['warnings'].append(f"Found {total_missing} missing values in the dataset")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].nunique() == 1:
            validation_results['warnings'].append(f"Column '{col}' has only one unique value")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 6],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'Eve', None],
        'age': [25, 30, 30, None, 35, 40, 28],
        'score': [85.5, 92.0, 92.0, 78.5, None, 88.0, 91.5],
        'department': ['HR', 'IT', 'IT', 'Finance', 'Marketing', None, 'IT']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, missing_strategy='median')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'name', 'age'])
    print("\nValidation Results:")
    print(validation)import pandas as pd

def clean_dataset(df):
    """
    Clean a pandas DataFrame by removing null values and duplicate rows.
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def filter_by_column(df, column_name, threshold):
    """
    Filter DataFrame rows where column value is greater than threshold.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of column to filter by.
        threshold (float): Threshold value.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    filtered_df = df[df[column_name] > threshold]
    return filtered_df

def main():
    # Example usage
    data = {
        'A': [1, 2, None, 4, 5, 5],
        'B': [10, 20, 30, None, 50, 50],
        'C': [100, 200, 300, 400, 500, 500]
    }
    
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    filtered_df = filter_by_column(cleaned_df, 'C', 250)
    print("Filtered DataFrame (C > 250):")
    print(filtered_df)

if __name__ == "__main__":
    main()
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        remove_duplicates (bool): Whether to remove duplicate rows.
        fill_method (str or None): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or None to drop rows.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is None:
        cleaned_df = cleaned_df.dropna()
    elif fill_method == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
    elif fill_method == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
    elif fill_method == 'mode':
        cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
    
    # Remove duplicates
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        dict: Validation results with issues found.
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
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
            validation_results['issues'].append(f'Missing columns: {missing_columns}')
    
    # Calculate basic statistics
    validation_results['summary']['row_count'] = len(df)
    validation_results['summary']['column_count'] = len(df.columns)
    validation_results['summary']['null_count'] = df.isnull().sum().sum()
    validation_results['summary']['duplicate_count'] = df.duplicated().sum()
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 1],
        'B': [5, None, 7, 8, 5],
        'C': ['x', 'y', 'z', 'x', 'y']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Clean the data
    cleaned = clean_dataset(df, fill_method='mean')
    print("Cleaned DataFrame:")
    print(cleaned)
    print()
    
    # Validate the data
    validation = validate_dataset(cleaned, required_columns=['A', 'B', 'C'])
    print("Validation Results:")
    print(f"Is valid: {validation['is_valid']}")
    print(f"Issues: {validation['issues']}")
    print(f"Summary: {validation['summary']}")
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
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

def calculate_statistics(df, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
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

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    tuple: (cleaned DataFrame, statistics dictionary)
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            all_stats[column] = stats
    
    return cleaned_df, all_stats

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10],
        'humidity': [45, 46, 47, 48, 49, 200, 50, 51, 52, -5],
        'pressure': [1013, 1014, 1015, 1016, 1017, 2000, 1018, 1019, 1020, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50)
    
    columns_to_process = ['temperature', 'humidity', 'pressure']
    cleaned_df, statistics = clean_dataset(df, columns_to_process)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50)
    
    print("\nStatistics:")
    for col, stats in statistics.items():
        print(f"\n{col}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")