
import pandas as pd
import numpy as np

def clean_csv_data(file_path, missing_strategy='mean', columns_to_drop=None):
    """
    Load and clean CSV data by handling missing values and optionally dropping columns.
    
    Parameters:
    file_path (str): Path to the CSV file.
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop').
    columns_to_drop (list): List of column names to drop.
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")
    
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")
    
    if df.isnull().sum().any():
        print("Missing values detected:")
        print(df.isnull().sum())
        
        if missing_strategy == 'drop':
            df = df.dropna()
            print("Missing rows dropped.")
        elif missing_strategy in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    if missing_strategy == 'mean':
                        fill_value = df[col].mean()
                    else:
                        fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    print(f"Filled missing values in '{col}' with {missing_strategy}: {fill_value:.2f}")
        elif missing_strategy == 'mode':
            for col in df.columns:
                if df[col].isnull().any():
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else None
                    if fill_value is not None:
                        df[col] = df[col].fillna(fill_value)
                        print(f"Filled missing values in '{col}' with mode: {fill_value}")
        else:
            raise ValueError("Invalid missing_strategy. Choose from 'mean', 'median', 'mode', or 'drop'.")
    
    cleaned_shape = df.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_shape[0]}")
    print(f"Columns removed: {original_shape[1] - cleaned_shape[1]}")
    
    return df

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to a CSV file.
    
    Parameters:
    df (pandas.DataFrame): Cleaned DataFrame.
    output_path (str): Path to save the cleaned CSV.
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    except Exception as e:
        raise Exception(f"Error saving cleaned data: {e}")

if __name__ == "__main__":
    input_file = "sample_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        cleaned_df = clean_csv_data(
            file_path=input_file,
            missing_strategy='mean',
            columns_to_drop=['unnecessary_column']
        )
        save_cleaned_data(cleaned_df, output_file)
    except Exception as e:
        print(f"Error during data cleaning: {e}")import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    subset (list, optional): Column labels to consider for duplicates
    keep (str, optional): Which duplicates to keep ('first', 'last', False)
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate rows")
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    bool: True if validation passes, False otherwise
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    return True

def clean_numeric_columns(df, columns=None):
    """
    Clean numeric columns by converting to appropriate types and handling errors.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns (list, optional): Specific columns to clean
    
    Returns:
    pd.DataFrame: DataFrame with cleaned numeric columns
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    cleaned_df = df.copy()
    
    for column in columns:
        if column in df.columns:
            cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors='coerce')
    
    return cleaned_df

def get_data_summary(df):
    """
    Generate summary statistics for a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'column_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    return summary

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve'],
        'score': ['85', '90', '90', '78', '92', '92', '88']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    # Remove duplicates
    cleaned_df = remove_duplicates(df, subset=['id', 'name'])
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    # Get summary
    summary = get_data_summary(cleaned_df)
    print("Data Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from a DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def fill_missing_values(df, strategy='mean', columns=None):
    """
    Fill missing values in specified columns using a given strategy.
    """
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            df_filled[col].fillna(fill_value, inplace=True)
        else:
            df_filled[col].fillna('Unknown', inplace=True)
    return df_filled

def validate_data_types(df, schema):
    """
    Validate that DataFrame columns match expected data types.
    """
    errors = []
    for col, expected_type in schema.items():
        if col not in df.columns:
            errors.append(f"Column '{col}' not found in DataFrame")
        elif not np.issubdtype(df[col].dtype, expected_type):
            errors.append(f"Column '{col}' has type {df[col].dtype}, expected {expected_type}")
    return errors

def normalize_column(df, column):
    """
    Normalize a numeric column to range [0, 1].
    """
    if df[column].dtype in [np.float64, np.int64]:
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val - min_val > 0:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def get_summary_stats(df):
    """
    Generate summary statistics for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].describe()