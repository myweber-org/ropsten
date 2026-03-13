
import numpy as np
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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def standardize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def handle_missing_values(df, strategy='mean'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            df[col].fillna(fill_value, inplace=True)
    return df

def clean_dataset(df, numeric_columns, outlier_removal=True, normalization='minmax'):
    df_clean = df.copy()
    
    if outlier_removal:
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean = remove_outliers_iqr(df_clean, col)
    
    df_clean = handle_missing_values(df_clean)
    
    for col in numeric_columns:
        if col in df_clean.columns:
            if normalization == 'minmax':
                df_clean = normalize_minmax(df_clean, col)
            elif normalization == 'zscore':
                df_clean = standardize_zscore(df_clean, col)
    
    return df_clean
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    outliers_removed = len(df) - len(filtered_df)
    
    return filtered_df, outliers_removed

def clean_dataset(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        total_outliers = 0
        for col in numeric_columns:
            df, outliers = remove_outliers_iqr(df, col)
            total_outliers += outliers
            print(f"Removed {outliers} outliers from column: {col}")
        
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to: {output_path}")
        print(f"Total outliers removed: {total_outliers}")
        print(f"Final dataset shape: {df.shape}")
        
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
    
    cleaned_data = clean_dataset(input_file, output_file)
    
    if cleaned_data is not None:
        print("Data cleaning completed successfully")
        print(cleaned_data.describe())
import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans a pandas DataFrame by removing duplicate rows and
    handling missing values in numeric columns.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # For numeric columns, fill missing values with the column median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].apply(
        lambda col: col.fillna(col.median())
    )

    # For non-numeric columns, fill missing values with 'Unknown'
    non_numeric_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
    df_cleaned[non_numeric_cols] = df_cleaned[non_numeric_cols].fillna('Unknown')

    return df_cleaned

def validate_data(df):
    """
    Performs basic validation on the cleaned DataFrame.
    """
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning.")

    # Check for any remaining NaN values
    if df.isnull().any().any():
        raise ValueError("DataFrame still contains NaN values.")

    print("Data validation passed.")
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, 2, np.nan, 5],
        'B': [10.5, np.nan, 10.5, 13.2, 15.0],
        'C': ['x', 'y', 'x', np.nan, 'z']
    }
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)

    cleaned_df = clean_data(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)

    try:
        validate_data(cleaned_df)
    except ValueError as e:
        print(f"Validation error: {e}")
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean dataset by handling missing values and removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    outlier_threshold (float): Z-score threshold for outlier detection
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    
    # Remove outliers using Z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned_df[numeric_cols] - cleaned_df[numeric_cols].mean()) / 
                     cleaned_df[numeric_cols].std())
    
    outlier_mask = (z_scores < outlier_threshold).all(axis=1)
    cleaned_df = cleaned_df[outlier_mask]
    
    # Reset index after cleaning
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_data(df, required_columns=None, min_rows=10):
    """
    Validate dataset structure and content.
    
    Parameters:
    df (pd.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    min_rows (int): Minimum number of rows required
    
    Returns:
    tuple: (is_valid, error_message)
    """
    if len(df) < min_rows:
        return False, f"Dataset has less than {min_rows} rows"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        if np.any(np.isinf(df[numeric_cols].values)):
            return False, "Dataset contains infinite values"
    
    return True, "Dataset validation passed"

def normalize_data(df, method='minmax'):
    """
    Normalize numerical columns in the dataset.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    method (str): Normalization method ('minmax' or 'standard')
    
    Returns:
    pd.DataFrame: Normalized dataframe
    """
    normalized_df = df.copy()
    numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
    
    if method == 'minmax':
        for col in numeric_cols:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max > col_min:  # Avoid division by zero
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    elif method == 'standard':
        for col in numeric_cols:
            col_mean = normalized_df[col].mean()
            col_std = normalized_df[col].std()
            if col_std > 0:  # Avoid division by zero
                normalized_df[col] = (normalized_df[col] - col_mean) / col_std
    
    return normalized_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'feature1': [1, 2, np.nan, 4, 100],
        'feature2': [5, 6, 7, np.nan, 8],
        'feature3': [9, 10, 11, 12, 13],
        'category': ['A', 'B', 'A', 'B', 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n")
    
    # Clean the dataset
    cleaned = clean_dataset(df, missing_strategy='mean', outlier_threshold=2)
    print("Cleaned dataset:")
    print(cleaned)
    print("\n")
    
    # Validate the cleaned dataset
    is_valid, message = validate_data(cleaned, required_columns=['feature1', 'feature2', 'feature3'])
    print(f"Validation: {is_valid} - {message}")
    print("\n")
    
    # Normalize the data
    normalized = normalize_data(cleaned, method='minmax')
    print("Normalized dataset:")
    print(normalized)import pandas as pd

def clean_dataset(df, remove_duplicates=True):
    """
    Clean a pandas DataFrame by removing null values and optionally duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        remove_duplicates (bool): Whether to remove duplicate rows.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    cleaned_df = cleaned_df.dropna()
    
    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Rows removed: {df.shape[0] - cleaned_df.shape[0]}")
    
    return cleaned_df

def validate_dataset(df, required_columns=None):
    """
    Validate dataset structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
    
    Returns:
        dict: Validation results.
    """
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('DataFrame is empty')
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing columns: {missing_columns}')
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        validation_results['issues'].append(f'Null values found: {null_counts.to_dict()}')
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Eve'],
        'age': [25, 30, 35, None, 40, 40],
        'score': [85.5, 92.0, 78.5, 88.0, 95.5, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Sample dataset:")
    print(df)
    print("\n" + "="*50)
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    validation = validate_dataset(cleaned_df, required_columns=['id', 'name', 'age'])
    print(f"\nValidation results: {validation}")
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original data shape: {df.shape}")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate rows")
            df = df.drop_duplicates()
        
        # Remove outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"Cleaned data shape: {df.shape}")
        
        # Save cleaned data
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        else:
            base_name = file_path.rsplit('.', 1)[0]
            output_path = f"{base_name}_cleaned.csv"
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_data(df, required_columns=None):
    """
    Validate data quality after cleaning.
    """
    if df is None or df.empty:
        print("Error: Invalid or empty DataFrame")
        return False
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    # Check for remaining missing values
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values still present")
    
    # Check data types
    print("Data types:")
    print(df.dtypes)
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    return True

if __name__ == "__main__":
    # Example usage
    input_file = "sample_data.csv"
    cleaned_data = clean_csv_data(input_file)
    
    if cleaned_data is not None:
        validate_data(cleaned_data)