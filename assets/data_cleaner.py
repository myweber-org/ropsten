
import pandas as pd

def clean_dataset(df):
    """
    Clean the dataset by removing null values and duplicate rows.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to be cleaned.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    # Remove rows with any null values
    df_cleaned = df.dropna()
    
    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Reset index after cleaning
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    return df_cleaned

def filter_by_threshold(df, column, threshold):
    """
    Filter rows where the specified column value is above a threshold.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame.
    column (str): Column name to apply the threshold.
    threshold (float): Threshold value.
    
    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    filtered_df = df[df[column] > threshold]
    return filtered_df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': [1, 2, None, 4, 4, 5],
        'B': [10, 20, 30, 40, 40, 50],
        'C': [100, 200, 300, 400, 400, 500]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    cleaned_df = clean_dataset(df)
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    filtered_df = filter_by_threshold(cleaned_df, 'B', 25)
    print("\nFiltered DataFrame (B > 25):")
    print(filtered_df)
def remove_duplicates_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result