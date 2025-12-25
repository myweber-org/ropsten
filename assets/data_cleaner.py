
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a dataset using the Interquartile Range method.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to check for outliers
    
    Returns:
    numpy.ndarray: Data with outliers removed
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1]:
        raise IndexError("Column index out of bounds")
    
    q1 = np.percentile(data[:, column], 25)
    q3 = np.percentile(data[:, column], 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    mask = (data[:, column] >= lower_bound) & (data[:, column] <= upper_bound)
    return data[mask]

def calculate_statistics(data, column):
    """
    Calculate basic statistics for a data column.
    
    Parameters:
    data (numpy.ndarray): Input data array
    column (int): Column index to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    column_data = data[:, column]
    
    stats = {
        'mean': np.mean(column_data),
        'median': np.median(column_data),
        'std': np.std(column_data),
        'min': np.min(column_data),
        'max': np.max(column_data),
        'q1': np.percentile(column_data, 25),
        'q3': np.percentile(column_data, 75)
    }
    
    return stats

def validate_data(data):
    """
    Validate data for cleaning operations.
    
    Parameters:
    data: Input data to validate
    
    Returns:
    bool: True if data is valid, False otherwise
    """
    if data is None:
        return False
    
    if not isinstance(data, np.ndarray):
        return False
    
    if data.size == 0:
        return False
    
    if np.any(np.isnan(data)):
        return False
    
    return True

def clean_dataset(data, columns_to_clean=None):
    """
    Clean dataset by removing outliers from specified columns.
    
    Parameters:
    data (numpy.ndarray): Input data array
    columns_to_clean (list): List of column indices to clean
    
    Returns:
    numpy.ndarray: Cleaned data array
    """
    if not validate_data(data):
        raise ValueError("Invalid input data")
    
    if columns_to_clean is None:
        columns_to_clean = list(range(data.shape[1]))
    
    cleaned_data = data.copy()
    
    for column in columns_to_clean:
        if column < data.shape[1]:
            cleaned_data = remove_outliers_iqr(cleaned_data, column)
    
    return cleaned_data
import pandas as pd

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_duplicates (bool): If True, remove duplicate rows.
    fill_missing (str): Method to fill missing values: 'mean', 'median', 'mode', or 'drop'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    if fill_missing == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif fill_missing in ['mean', 'median']:
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if fill_missing == 'mean':
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif fill_missing == 'mode':
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate a DataFrame for required columns and data types.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of column names that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
    
    if df.empty:
        print("DataFrame is empty")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, None],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataset(cleaned, required_columns=['A', 'B'])
    print(f"\nDataset validation: {is_valid}")import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    """
    Clean a CSV file by removing duplicates, handling missing values,
    and standardizing column names.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')
        
        # Remove rows where critical columns are still null
        critical_columns = ['id', 'date', 'value']
        existing_critical = [col for col in critical_columns if col in df.columns]
        if existing_critical:
            df = df.dropna(subset=existing_critical)
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaned successfully. Output saved to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    clean_data(input_csv, output_csv)