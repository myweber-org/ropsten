
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean dataset by removing duplicates, standardizing column names,
    and handling missing values.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Fill missing numeric values with column median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Fill missing categorical values with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown')
    
    return df_clean

def validate_data(df):
    """
    Validate data by checking for remaining nulls and data types.
    """
    validation_report = {}
    
    # Check for null values
    null_counts = df.isnull().sum()
    validation_report['null_counts'] = null_counts[null_counts > 0].to_dict()
    
    # Check data types
    validation_report['dtypes'] = df.dtypes.to_dict()
    
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    negative_counts = {}
    for col in numeric_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            negative_counts[col] = negative_count
    validation_report['negative_counts'] = negative_counts
    
    return validation_report

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'Customer ID': [1, 2, 2, 3, 4],
        'Order Value': [100, 200, 200, np.nan, 300],
        'Product Category': ['A', 'B', 'B', None, 'C'],
        'Discount Applied': [0.1, -0.2, 0.15, 0.0, 0.05]
    })
    
    print("Original dataset:")
    print(sample_data)
    print("\nCleaned dataset:")
    cleaned_data = clean_dataset(sample_data)
    print(cleaned_data)
    
    print("\nValidation report:")
    report = validate_data(cleaned_data)
    for key, value in report.items():
        print(f"{key}: {value}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")

if __name__ == "__main__":
    clean_dataset("raw_data.csv", "cleaned_data.csv")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        drop_duplicates (bool): Whether to remove duplicate rows
        fill_missing (str): Strategy for filling missing values ('mean', 'median', 'mode', or 'drop')
    
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
    missing_count = cleaned_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values")
        
        if fill_missing == 'drop':
            cleaned_df = cleaned_df.dropna()
            print("Removed rows with missing values")
        elif fill_missing in ['mean', 'median']:
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if fill_missing == 'mean':
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                else:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            print(f"Filled missing numeric values with {fill_missing}")
        elif fill_missing == 'mode':
            for col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None)
            print("Filled missing values with mode")
    
    # Remove columns with all missing values
    cols_before = len(cleaned_df.columns)
    cleaned_df = cleaned_df.dropna(axis=1, how='all')
    cols_removed = cols_before - len(cleaned_df.columns)
    if cols_removed > 0:
        print(f"Removed {cols_removed} columns with all missing values")
    
    print(f"Cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
    
    Returns:
        dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            validation_results['warnings'].append(f"Column '{col}' contains infinite values")
    
    # Check data types
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if unique_count == len(df):
                validation_results['warnings'].append(f"Column '{col}' has all unique values (possible ID column)")
    
    return validation_results

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10.5, None, 15.2, 15.2, None, 20.1],
        'category': ['A', 'B', 'A', 'A', 'B', 'C'],
        'empty_col': [None, None, None, None, None, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned = clean_dataset(df, fill_missing='mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned, required_columns=['id', 'value'])
    print("\nValidation Results:")
    print(validation)