
import pandas as pd
import numpy as np

def clean_dataset(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Z-score threshold for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Handle missing values
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
            cleaned_df[numeric_cols].mean()
        )
    elif missing_strategy == 'median':
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(
            cleaned_df[numeric_cols].median()
        )
    elif missing_strategy == 'mode':
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(
                cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 0
            )
    elif missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna(subset=numeric_cols)
    
    # Handle outliers using Z-score method
    for col in numeric_cols:
        z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
        outliers = z_scores > outlier_threshold
        
        if outliers.any():
            # Cap outliers at threshold
            col_mean = cleaned_df[col].mean()
            col_std = cleaned_df[col].std()
            upper_bound = col_mean + outlier_threshold * col_std
            lower_bound = col_mean - outlier_threshold * col_std
            
            cleaned_df.loc[outliers, col] = np.where(
                cleaned_df.loc[outliers, col] > upper_bound,
                upper_bound,
                lower_bound
            )
    
    # Fill remaining non-numeric missing values
    non_numeric_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
    cleaned_df[non_numeric_cols] = cleaned_df[non_numeric_cols].fillna('Unknown')
    
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

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'age': [25, 30, np.nan, 35, 150, 28, 32],
        'salary': [50000, 60000, 55000, np.nan, 1000000, 52000, 58000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, missing_strategy='median', outlier_threshold=2.5)
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['age', 'salary'])
    print(f"Validation result: {is_valid}")
    print(f"Validation message: {message}")