
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd

def clean_dataset(df, drop_na=True, column_case='lower'):
    """
    Clean a pandas DataFrame by handling missing values and standardizing column names.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    drop_na (bool): If True, drop rows with any null values. Default is True.
    column_case (str): Target case for column names ('lower', 'upper', 'title'). Default is 'lower'.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    if drop_na:
        df_clean = df_clean.dropna()
    
    if column_case == 'lower':
        df_clean.columns = df_clean.columns.str.lower()
    elif column_case == 'upper':
        df_clean.columns = df_clean.columns.str.upper()
    elif column_case == 'title':
        df_clean.columns = df_clean.columns.str.title()
    
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def validate_numeric_columns(df, numeric_columns):
    """
    Validate that specified columns contain only numeric data.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    numeric_columns (list): List of column names expected to be numeric.
    
    Returns:
    dict: Dictionary with validation results for each column.
    """
    validation_results = {}
    
    for col in numeric_columns:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
            validation_results[col] = {
                'total_non_numeric': int(non_numeric),
                'is_valid': non_numeric == 0
            }
        else:
            validation_results[col] = {
                'error': 'Column not found',
                'is_valid': False
            }
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 30, 35],
        'Score': ['90', '85', 'invalid', '95']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print()
    
    cleaned_df = clean_dataset(df, drop_na=True, column_case='lower')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print()
    
    validation = validate_numeric_columns(cleaned_df, ['age', 'score'])
    print("Numeric Validation Results:")
    print(validation)