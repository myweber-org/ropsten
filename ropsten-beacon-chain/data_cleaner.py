
import pandas as pd
import numpy as np
from typing import List, Optional

def clean_dataset(df: pd.DataFrame, 
                  drop_duplicates: bool = True,
                  columns_to_standardize: Optional[List[str]] = None,
                  date_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates, standardizing text columns,
    and converting date columns to datetime format.
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_rows - len(df_clean)
        print(f"Removed {removed} duplicate rows")
    
    if columns_to_standardize:
        for col in columns_to_standardize:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
                df_clean[col] = df_clean[col].replace({'nan': np.nan, 'none': np.nan})
    
    if date_columns:
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    missing_values = df_clean.isnull().sum().sum()
    if missing_values > 0:
        print(f"Dataset contains {missing_values} missing values")
    
    return df_clean

def validate_email_column(df: pd.DataFrame, email_column: str) -> pd.Series:
    """
    Validate email addresses in a specified column and return a boolean series.
    """
    if email_column not in df.columns:
        raise ValueError(f"Column '{email_column}' not found in DataFrame")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].str.match(email_pattern, na=False)

def remove_outliers_iqr(df: pd.DataFrame, 
                       numeric_columns: List[str],
                       multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from numeric columns using the Interquartile Range method.
    """
    df_filtered = df.copy()
    
    for col in numeric_columns:
        if col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col]):
            Q1 = df_filtered[col].quantile(0.25)
            Q3 = df_filtered[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)
            outliers_removed = len(df_filtered) - mask.sum()
            
            if outliers_removed > 0:
                print(f"Removed {outliers_removed} outliers from column '{col}'")
                df_filtered = df_filtered[mask]
    
    return df_filtered

if __name__ == "__main__":
    sample_data = {
        'name': ['John', 'Jane', 'John', 'Bob', 'Alice', 'ALICE'],
        'email': ['john@example.com', 'jane@test.org', 'invalid', 'bob@sample.net', None, 'alice@demo.com'],
        'age': [25, 30, 25, 150, 28, 28],
        'join_date': ['2023-01-15', '2023-02-20', '2023-01-15', '2023-03-10', 'invalid', '2023-04-05']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(
        df,
        columns_to_standardize=['name'],
        date_columns=['join_date']
    )
    
    print("Cleaned dataset:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    email_valid = validate_email_column(cleaned_df, 'email')
    print("Valid email addresses:")
    print(email_valid)
    print("\n" + "="*50 + "\n")
    
    filtered_df = remove_outliers_iqr(cleaned_df, ['age'])
    print("Dataset after outlier removal:")
    print(filtered_df)