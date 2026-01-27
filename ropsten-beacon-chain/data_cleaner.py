
def remove_duplicates(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return resultimport pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows.")
    
    if fill_missing:
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            if fill_strategy == 'mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_strategy == 'median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_strategy == 'mode':
                for col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
            elif fill_strategy == 'drop':
                df = df.dropna()
            print(f"Handled {missing_count} missing values using {fill_strategy} strategy.")
    
    print(f"Dataset cleaned. Original shape: {original_shape}, New shape: {df.shape}")
    return df

def validate_data(df, required_columns=None, min_rows=1):
    """
    Validate the dataset structure and content.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    if len(df) < min_rows:
        raise ValueError(f"Dataset has fewer than {min_rows} rows.")
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 3, None, 5],
        'B': [10, None, 10, 30, 40, 50],
        'C': ['x', 'y', 'x', 'y', 'z', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n")
    
    cleaned_df = clean_dataset(df.copy(), fill_strategy='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    try:
        validate_data(cleaned_df, required_columns=['A', 'B'], min_rows=3)
        print("Data validation passed.")
    except ValueError as e:
        print(f"Data validation failed: {e}")