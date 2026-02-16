
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

if __name__ == "__main__":
    data_file = "sample_data.csv"
    target_columns = ['feature1', 'feature2', 'feature3']
    result = clean_dataset(data_file, target_columns)
    print(f"Cleaned data saved to: {result}")
import pandas as pd

def clean_dataset(df, remove_duplicates=True, fill_method=None):
    """
    Clean a pandas DataFrame by handling missing values and optionally removing duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        remove_duplicates (bool): If True, remove duplicate rows.
        fill_method (str or None): Method to fill missing values.
            Options: 'mean', 'median', 'mode', or None to drop rows with nulls.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if fill_method is None:
        cleaned_df = cleaned_df.dropna()
    else:
        if fill_method == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif fill_method == 'median':
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif fill_method == 'mode':
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        else:
            raise ValueError(f"Unsupported fill_method: {fill_method}")
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of column names that must be present.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 4],
        'B': [5, None, 7, 8, 8],
        'C': [9, 10, 11, 12, 12]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, remove_duplicates=True, fill_method='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")import pandas as pd

def remove_duplicates(input_file, output_file, subset_columns=None):
    """
    Load data from CSV file, remove duplicate rows based on specified columns,
    and save cleaned data to a new CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        subset_columns (list, optional): List of column names to consider for duplicates
    """
    try:
        df = pd.read_csv(input_file)
        
        original_count = len(df)
        
        if subset_columns:
            df_cleaned = df.drop_duplicates(subset=subset_columns, keep='first')
        else:
            df_cleaned = df.drop_duplicates(keep='first')
        
        removed_count = original_count - len(df_cleaned)
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Original records: {original_count}")
        print(f"Removed duplicates: {removed_count}")
        print(f"Cleaned records: {len(df_cleaned)}")
        print(f"Cleaned data saved to: {output_file}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return None

if __name__ == "__main__":
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    
    columns_to_check = ['id', 'email']
    
    cleaned_data = remove_duplicates(input_csv, output_csv, columns_to_check)
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    sample_data.loc[10:15, 'A'] = 500
    sample_data.loc[20:25, 'B'] = 1000
    
    numeric_cols = ['A', 'B']
    result = clean_dataset(sample_data, numeric_cols)
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {result.shape}")
    print("Outliers removed successfully")
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            df = remove_outliers_iqr(df, column)
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset shape: {df.shape}")
        print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    cleaned_df = clean_dataset(input_file, output_file)import pandas as pd
import numpy as np

def clean_csv_data(filepath, fill_strategy='mean', drop_threshold=0.5):
    """
    Load and clean CSV data by handling missing values.
    
    Args:
        filepath (str): Path to the CSV file.
        fill_strategy (str): Strategy for filling missing values.
                             Options: 'mean', 'median', 'mode', 'zero'.
        drop_threshold (float): Drop columns with missing ratio above this threshold.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Calculate missing ratio per column
    missing_ratio = df.isnull().sum() / len(df)
    
    # Drop columns with high missing ratio
    columns_to_drop = missing_ratio[missing_ratio > drop_threshold].index
    if len(columns_to_drop) > 0:
        print(f"Dropping columns with >{drop_threshold*100}% missing values: {list(columns_to_drop)}")
        df = df.drop(columns=columns_to_drop)
    
    # Fill missing values based on strategy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if fill_strategy == 'mean':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
    elif fill_strategy == 'median':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    elif fill_strategy == 'mode':
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    elif fill_strategy == 'zero':
        df = df.fillna(0)
    else:
        print(f"Warning: Unknown fill strategy '{fill_strategy}', using forward fill")
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # For categorical columns, fill with most frequent value
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            most_frequent = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(most_frequent)
    
    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate rows")
        df = df.drop_duplicates()
    
    final_shape = df.shape
    print(f"Cleaned data shape: {final_shape}")
    print(f"Removed {original_shape[0] - final_shape[0]} rows and {original_shape[1] - final_shape[1]} columns")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        output_path (str): Path to save the cleaned data.
    """
    if df is not None:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
    else:
        print("No data to save")

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, fill_strategy='median', drop_threshold=0.3)
    save_cleaned_data(cleaned_df, output_file)
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
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a DataFrame column.
    
    Parameters:
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
        'count': df[column].count(),
        'missing': df[column].isnull().sum()
    }
    
    return stats

def clean_numeric_data(df, columns=None):
    """
    Clean numeric columns by removing outliers and filling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list): List of column names to clean. If None, clean all numeric columns.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            # Fill missing values with median
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            
            # Remove outliers
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nSummary statistics for column 'A':")
    print(calculate_summary_stats(df, 'A'))
    
    cleaned_df = clean_numeric_data(df, columns=['A'])
    print("\nCleaned DataFrame (outliers removed from column 'A'):")
    print(cleaned_df)
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result