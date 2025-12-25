
import pandas as pd
import numpy as np

def clean_dataframe(df, drop_duplicates=True, fill_missing=True):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows")
    
    if fill_missing:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                print(f"Filled missing values in column '{col}' with median")
        
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if cleaned_df[col].isnull().any():
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
                print(f"Filled missing values in column '{col}' with 'Unknown'")
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate DataFrame for common data quality issues.
    """
    issues = []
    
    if df.empty:
        issues.append("DataFrame is empty")
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows")
    
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        issues.append(f"Found {missing_total} missing values")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].abs().max() > 1e10:
            issues.append(f"Column '{col}' contains extremely large values")
    
    return issues

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nData quality issues:")
    issues = validate_dataframe(df)
    for issue in issues:
        print(f"- {issue}")
    
    cleaned_df = clean_dataframe(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)