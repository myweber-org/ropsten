
import pandas as pd

def clean_data(df):
    """
    Clean the input DataFrame by removing duplicate rows and
    filling missing numeric values with the column mean.
    """
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()

    # Fill missing numeric values with column mean
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())

    return df_cleaned

if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'A': [1, 2, 2, None, 5],
        'B': [5, None, 7, 8, 9],
        'C': ['x', 'y', 'y', 'z', 'x']
    })
    print("Original data:")
    print(data)
    cleaned = clean_data(data)
    print("\nCleaned data:")
    print(cleaned)
import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def normalize_data(df, columns, method='minmax'):
    df_norm = df.copy()
    for col in columns:
        if method == 'minmax':
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        elif method == 'zscore':
            df_norm[col] = (df_norm[col] - df_norm[col].mean()) / df_norm[col].std()
        elif method == 'robust':
            df_norm[col] = (df_norm[col] - df_norm[col].median()) / stats.iqr(df_norm[col])
    return df_norm

def clean_dataset(df, numeric_columns):
    df_no_outliers = remove_outliers_iqr(df, numeric_columns)
    df_normalized = normalize_data(df_no_outliers, numeric_columns, method='zscore')
    return df_normalized

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature_a': np.random.normal(100, 15, 200),
        'feature_b': np.random.exponential(50, 200),
        'feature_c': np.random.uniform(0, 1, 200)
    })
    cleaned_df = clean_dataset(sample_data, ['feature_a', 'feature_b', 'feature_c'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Cleaned data summary:\n{cleaned_df.describe()}")