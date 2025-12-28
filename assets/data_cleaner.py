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

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        cleaned_df = remove_outliers_iqr(cleaned_df, col)
        cleaned_df = normalize_minmax(cleaned_df, col)
    return cleaned_df

def validate_data(df, required_columns):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    })
    
    print("Original data shape:", sample_data.shape)
    cleaned = clean_dataset(sample_data, ['A', 'B', 'C'])
    print("Cleaned data shape:", cleaned.shape)
    print("Cleaned data columns:", cleaned.columns.tolist())import pandas as pd
import re

def clean_dataframe(df, text_columns=None):
    """
    Remove duplicate rows and standardize text in specified columns.
    """
    df_clean = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    removed_duplicates = initial_rows - len(df_clean)
    
    # Standardize text in specified columns
    if text_columns:
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(_standardize_text)
    
    return df_clean, removed_duplicates

def _standardize_text(text):
    """
    Helper function to standardize text: lowercase, remove extra spaces.
    """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def validate_email_column(df, email_column):
    """
    Validate email addresses in a specified column.
    """
    if email_column not in df.columns:
        return pd.Series([], dtype=bool)
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return df[email_column].apply(lambda x: bool(re.match(email_pattern, str(x))))