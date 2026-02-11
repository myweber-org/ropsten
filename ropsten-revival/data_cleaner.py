
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

def calculate_basic_stats(df, column):
    """
    Calculate basic statistics for a DataFrame column.
    
    Args:
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric data by removing outliers from specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to clean. If None, uses all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = numeric_cols
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{col}'")
    
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
    print("\nOriginal statistics for temperature:")
    print(calculate_basic_stats(df, 'temperature'))
    
    cleaned_df = clean_numeric_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    print("\nCleaned statistics for temperature:")
    print(calculate_basic_stats(cleaned_df, 'temperature'))
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to clean.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path, output_path):
    """
    Load a dataset, clean specified columns, and save the cleaned data.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str): Path to save the cleaned CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            df = remove_outliers_iqr(df, col)
        
        print(f"Cleaned dataset shape: {df.shape}")
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    clean_dataset(input_file, output_file)import pandas as pd

def clean_dataset(df, column_names=None):
    """
    Clean a pandas DataFrame by removing duplicates and normalizing string columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        column_names (list, optional): List of column names to apply string normalization.
                                      If None, applies to all object dtype columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates()
    removed_duplicates = initial_rows - len(cleaned_df)
    
    # Normalize string columns
    if column_names is None:
        # Select all object dtype columns
        string_cols = cleaned_df.select_dtypes(include=['object']).columns
    else:
        # Use specified columns that exist in DataFrame
        string_cols = [col for col in column_names if col in cleaned_df.columns]
    
    for col in string_cols:
        if cleaned_df[col].dtype == 'object':
            # Strip whitespace and convert to lowercase
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()
            # Replace multiple spaces with single space
            cleaned_df[col] = cleaned_df[col].str.replace(r'\s+', ' ', regex=True)
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    print(f"Removed {removed_duplicates} duplicate rows.")
    print(f"Normalized {len(string_cols)} string columns.")
    print(f"Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        email_column (str): Name of the column containing email addresses.
    
    Returns:
        pd.DataFrame: DataFrame with additional validation columns.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame.")
    
    validation_df = df.copy()
    
    # Basic email validation regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if emails match pattern
    validation_df['email_valid'] = validation_df[email_column].str.match(email_pattern)
    
    # Check for missing emails
    validation_df['email_missing'] = validation_df[email_column].isna() | (validation_df[email_column].str.strip() == '')
    
    # Count valid emails
    valid_count = validation_df['email_valid'].sum()
    total_count = len(validation_df)
    
    print(f"Email validation complete:")
    print(f"  Total emails: {total_count}")
    print(f"  Valid emails: {valid_count} ({valid_count/total_count*100:.1f}%)")
    print(f"  Invalid emails: {total_count - valid_count}")
    
    return validation_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'name': ['John Doe', 'Jane Smith', 'John Doe', 'Bob Johnson', 'alice brown'],
        'email': ['john@example.com', 'jane@test.org', 'invalid-email', 'bob@company.com', ''],
        'age': [25, 30, 25, 35, 28],
        'city': ['New York', 'Los Angeles', 'New York', 'Chicago', 'boston']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate emails
    print("\n" + "="*50 + "\n")
    validated = validate_email_column(cleaned, 'email')
    print("\nDataFrame with email validation:")
    print(validated[['name', 'email', 'email_valid', 'email_missing']])
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(dataframe, column, threshold=1.5):
    """
    Remove outliers from specified column using IQR method
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
    
    return filtered_df.copy()

def z_score_normalize(dataframe, columns):
    """
    Apply z-score normalization to specified columns
    """
    normalized_df = dataframe.copy()
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        
        if std_val > 0:
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        else:
            normalized_df[col] = 0
    
    return normalized_df

def min_max_normalize(dataframe, columns, feature_range=(0, 1)):
    """
    Apply min-max normalization to specified columns
    """
    normalized_df = dataframe.copy()
    min_val, max_val = feature_range
    
    for col in columns:
        if col not in normalized_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        col_range = col_max - col_min
        
        if col_range > 0:
            normalized_df[col] = ((normalized_df[col] - col_min) / col_range) * (max_val - min_val) + min_val
        else:
            normalized_df[col] = min_val
    
    return normalized_df

def detect_missing_patterns(dataframe, threshold=0.3):
    """
    Detect columns with high percentage of missing values
    """
    missing_stats = {}
    total_rows = len(dataframe)
    
    for col in dataframe.columns:
        missing_count = dataframe[col].isnull().sum()
        missing_percentage = (missing_count / total_rows) * 100
        
        missing_stats[col] = {
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'needs_attention': missing_percentage > threshold * 100
        }
    
    return pd.DataFrame.from_dict(missing_stats, orient='index')

def clean_dataset(dataframe, numeric_columns, outlier_threshold=1.5, normalize_method='zscore'):
    """
    Complete data cleaning pipeline
    """
    cleaned_df = dataframe.copy()
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col, outlier_threshold)
    
    if normalize_method == 'zscore':
        cleaned_df = z_score_normalize(cleaned_df, numeric_columns)
    elif normalize_method == 'minmax':
        cleaned_df = min_max_normalize(cleaned_df, numeric_columns)
    
    return cleaned_df
def remove_duplicates(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result