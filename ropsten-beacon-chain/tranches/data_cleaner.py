import pandas as pd

def clean_dataframe(df, drop_duplicates=True, fill_missing=None):
    """
    Cleans a pandas DataFrame by removing duplicates and handling missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        drop_duplicates (bool): If True, drop duplicate rows.
        fill_missing (str or dict, optional): Method to fill missing values.
            Can be 'ffill', 'bfill', a scalar value, or a dict of column:value pairs.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    cleaned_df = df.copy()

    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if fill_missing is not None:
        if isinstance(fill_missing, dict):
            cleaned_df = cleaned_df.fillna(fill_missing)
        else:
            cleaned_df = cleaned_df.fillna(method=fill_missing)

    return cleaned_df

if __name__ == "__main__":
    # Example usage
    data = {
        'A': [1, 2, 2, None, 5],
        'B': [10, None, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', None]
    }
    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df)

    cleaned = clean_dataframe(df, fill_missing={'A': 0, 'B': 'bfill', 'C': 'unknown'})
    print("\nCleaned DataFrame:")
    print(cleaned)