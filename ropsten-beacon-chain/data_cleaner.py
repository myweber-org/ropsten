import pandas as pd

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list, optional): Column labels to consider for duplicates.
    keep (str, optional): Which duplicates to keep.
    
    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    if df.empty:
        return df
    
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
    
    removed_count = len(df) - len(cleaned_df)
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate row(s)")
    
    return cleaned_df

def clean_numeric_column(df, column_name, fill_method='mean'):
    """
    Clean a numeric column by handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the column to clean.
    fill_method (str): Method to fill missing values ('mean', 'median', 'zero').
    
    Returns:
    pd.DataFrame: DataFrame with cleaned column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Column '{column_name}' is not numeric")
    
    df_clean = df.copy()
    
    missing_count = df_clean[column_name].isna().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing value(s) in column '{column_name}'")
        
        if fill_method == 'mean':
            fill_value = df_clean[column_name].mean()
        elif fill_method == 'median':
            fill_value = df_clean[column_name].median()
        elif fill_method == 'zero':
            fill_value = 0
        else:
            raise ValueError(f"Invalid fill_method: {fill_method}")
        
        df_clean[column_name] = df_clean[column_name].fillna(fill_value)
        print(f"Filled missing values with {fill_method}: {fill_value}")
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list, optional): List of columns that must be present.
    
    Returns:
    bool: True if validation passes, False otherwise.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input is not a pandas DataFrame")
        return False
    
    if df.empty:
        print("Warning: DataFrame is empty")
        return True
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
    
    print(f"DataFrame validation passed. Shape: {df.shape}")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 4, 5],
        'value': [10.5, None, 20.3, 30.1, 40.0, 40.0, None],
        'category': ['A', 'B', 'B', 'C', 'D', 'D', 'E']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    if validate_dataframe(df, required_columns=['id', 'value']):
        df_clean = remove_duplicates(df, subset=['id'], keep='first')
        df_clean = clean_numeric_column(df_clean, 'value', fill_method='mean')
        
        print("\nCleaned DataFrame:")
        print(df_clean)
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 100),
        'B': np.random.exponential(50, 100),
        'C': np.random.randint(1, 1000, 100)
    }
    df = pd.DataFrame(sample_data)
    numeric_cols = ['A', 'B', 'C']
    result = clean_dataset(df, numeric_cols)
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {result.shape}")
    print(result.head())
import pandas as pd
import numpy as np
import sys

def clean_csv_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        df = df.drop_duplicates()
        
        Q1 = df[numeric_columns].quantile(0.25)
        Q3 = df[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_condition = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
        df_cleaned = df[~outlier_condition]
        
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"Cleaned data shape: {df_cleaned.shape}")
        print(f"Removed {len(df) - len(df_cleaned)} outliers")
        print(f"Cleaned data saved to: {output_file}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return False
    except pd.errors.EmptyDataError:
        print("Error: Input file is empty.")
        return False
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file.csv> <output_file.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = clean_csv_data(input_file, output_file)
    sys.exit(0 if success else 1)