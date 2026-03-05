import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    cleaned_df = df.copy()
    
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
        print(f"Removed {original_shape[0] - cleaned_df.shape[0]} duplicate rows.")
    
    if fill_missing:
        for column in cleaned_df.columns:
            if cleaned_df[column].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    fill_value = cleaned_df[column].mean()
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    fill_value = cleaned_df[column].median()
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    fill_value = 0
                
                missing_count = cleaned_df[column].isnull().sum()
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
                print(f"Filled {missing_count} missing values in column '{column}' with {fill_value}.")
    
    print(f"Dataset cleaned. Original shape: {original_shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df, required_columns=None, unique_constraints=None):
    """
    Validate dataset structure and constraints.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if unique_constraints:
        for constraint in unique_constraints:
            if df[constraint].duplicated().any():
                duplicates = df[df[constraint].duplicated(keep=False)]
                print(f"Warning: Duplicate values found in unique constraint column '{constraint}'")
                print(f"Duplicate entries:\n{duplicates[constraint].value_counts().head()}")
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 3, 4, 5],
        'value': [10, 20, np.nan, 30, 40, np.nan],
        'category': ['A', 'B', 'A', 'A', 'B', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    try:
        validate_dataset(cleaned_df, required_columns=['id', 'value'], unique_constraints=['id'])
        print("\nDataset validation passed.")
    except ValueError as e:
        print(f"\nDataset validation failed: {e}")