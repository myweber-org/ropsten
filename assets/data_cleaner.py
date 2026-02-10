
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                           Options: 'mean', 'median', 'drop', 'fill_zero'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    if missing_strategy == 'mean':
        cleaned_df = cleaned_df.fillna(cleaned_df.mean())
    elif missing_strategy == 'median':
        cleaned_df = cleaned_df.fillna(cleaned_df.median())
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == 'fill_zero':
        cleaned_df = cleaned_df.fillna(0)
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Handle outliers using z-score method
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if cleaned_df[col].notna().any():
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            outlier_mask = z_scores > outlier_threshold
            
            if outlier_mask.any():
                # Replace outliers with column median
                col_median = cleaned_df[col].median()
                cleaned_df.loc[outlier_mask, col] = col_median
    
    # Reset index if rows were dropped
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

# Example usage function
def process_sample_data():
    """
    Demonstrate the data cleaning functions with sample data.
    """
    
    # Create sample data with missing values and outliers
    sample_data = {
        'temperature': [22.5, 23.1, np.nan, 24.3, 100.0, 22.8, 23.5],
        'humidity': [45, 48, 52, np.nan, 47, 200, 49],
        'pressure': [1013, 1012, 1014, 1013, 1015, 1012, np.nan]
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Validate the DataFrame
    is_valid, message = validate_dataframe(df)
    print(f"Validation: {message}")
    
    # Clean the DataFrame
    cleaned_df = clean_dataframe(df, missing_strategy='median', outlier_threshold=2.5)
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    return cleaned_df

if __name__ == "__main__":
    process_sample_data()