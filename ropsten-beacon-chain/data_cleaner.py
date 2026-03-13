
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