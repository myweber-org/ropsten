
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a specified column using the IQR method.
    
    Parameters:
    data (np.ndarray): The dataset.
    column (int): Index of the column to process.
    
    Returns:
    np.ndarray: Data with outliers removed.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if column >= data.shape[1] or column < 0:
        raise IndexError("Column index out of bounds")
    
    col_data = data[:, column]
    Q1 = np.percentile(col_data, 25)
    Q3 = np.percentile(col_data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)
    return data[mask]import pandas as pd
import numpy as np
from scipy import stats

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Initial shape: {df.shape}")
    
    df_cleaned = df.copy()
    
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df_cleaned[col].dropna()))
        outliers = z_scores > 3
        df_cleaned.loc[outliers, col] = np.nan
        print(f"Removed {outliers.sum()} outliers from {col}")
    
    df_cleaned = df_cleaned.dropna()
    print(f"Shape after outlier removal: {df_cleaned.shape}")
    
    for col in numeric_cols:
        if col in df_cleaned.columns:
            col_min = df_cleaned[col].min()
            col_max = df_cleaned[col].max()
            if col_max > col_min:
                df_cleaned[col] = (df_cleaned[col] - col_min) / (col_max - col_min)
    
    print(f"Final cleaned shape: {df_cleaned.shape}")
    return df_cleaned

if __name__ == "__main__":
    cleaned_df = load_and_clean_data("sample_data.csv")
    cleaned_df.to_csv("cleaned_data.csv", index=False)
    print("Data cleaning complete. Saved to cleaned_data.csv")