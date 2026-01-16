
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    """
    Clean a dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    numeric_columns (list): List of column names to clean.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    return cleaned_df.reset_index(drop=True)

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'A': np.random.randn(100),
        'B': np.random.exponential(2, 100),
        'C': np.random.uniform(0, 10, 100)
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df.loc[10, 'A'] = 100  # Add an outlier
    sample_df.loc[20, 'B'] = 50   # Add an outlier
    
    print("Original shape:", sample_df.shape)
    cleaned = clean_dataset(sample_df, ['A', 'B'])
    print("Cleaned shape:", cleaned.shape)
    print("Outliers removed:", sample_df.shape[0] - cleaned.shape[0])