import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean the dataset by removing duplicates and handling missing values.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Fill missing numeric values with column median
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
    
    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_cleaned[col].isnull().any():
            mode_value = df_cleaned[col].mode()[0]
            df_cleaned[col] = df_cleaned[col].fillna(mode_value)
    
    return df_cleaned

def remove_outliers(df, column, threshold=3):
    """
    Remove outliers from a specific column using z-score method.
    """
    from scipy import stats
    z_scores = np.abs(stats.zscore(df[column]))
    df_filtered = df[(z_scores < threshold)]
    return df_filtered

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    """
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 3, 4, 5, 5],
        'value': [10, 20, np.nan, 40, 50, 50],
        'category': ['A', 'B', 'A', np.nan, 'C', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)