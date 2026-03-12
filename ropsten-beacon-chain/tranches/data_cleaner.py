
import pandas as pd
import numpy as np
from typing import Optional, List

def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    handle_nulls: str = 'drop',
    columns_to_clean: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Clean a pandas DataFrame by removing duplicates and handling null values.
    
    Parameters:
    df: Input DataFrame
    drop_duplicates: Whether to remove duplicate rows
    handle_nulls: Strategy for null handling - 'drop', 'fill_mean', 'fill_median', 'fill_mode'
    columns_to_clean: Specific columns to apply null handling to
    
    Returns:
    Cleaned DataFrame
    """
    
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - len(cleaned_df)
        print(f"Removed {removed} duplicate rows")
    
    if handle_nulls:
        if columns_to_clean is None:
            columns_to_clean = cleaned_df.columns.tolist()
        
        null_counts = cleaned_df[columns_to_clean].isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            print(f"Found {total_nulls} null values to handle")
            
            if handle_nulls == 'drop':
                cleaned_df = cleaned_df.dropna(subset=columns_to_clean)
                print(f"Dropped rows with nulls in specified columns")
            
            elif handle_nulls in ['fill_mean', 'fill_median', 'fill_mode']:
                for col in columns_to_clean:
                    if cleaned_df[col].isnull().any():
                        if handle_nulls == 'fill_mean' and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            fill_value = cleaned_df[col].mean()
                        elif handle_nulls == 'fill_median' and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            fill_value = cleaned_df[col].median()
                        elif handle_nulls == 'fill_mode':
                            fill_value = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else None
                        else:
                            fill_value = None
                        
                        if fill_value is not None:
                            cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                            print(f"Filled nulls in column '{col}' with {handle_nulls.split('_')[1]}")
    
    return cleaned_df

def validate_dataset(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains all required columns and has no nulls in them.
    
    Parameters:
    df: DataFrame to validate
    required_columns: List of column names that must be present and non-null
    
    Returns:
    Boolean indicating if validation passed
    """
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    null_check = df[required_columns].isnull().any()
    columns_with_nulls = null_check[null_check].index.tolist()
    
    if columns_with_nulls:
        print(f"Columns with null values: {columns_with_nulls}")
        return False
    
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', None, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_dataset(df, handle_nulls='fill_mean')
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    is_valid = validate_dataset(cleaned, ['id', 'value'])
    print(f"\nDataset validation: {'PASS' if is_valid else 'FAIL'}")