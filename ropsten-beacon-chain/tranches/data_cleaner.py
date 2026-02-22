
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            original_len = len(data)
            data = remove_outliers_iqr(data, col)
            removed = original_len - len(data)
            print(f"Removed {removed} outliers from column: {col}")
        
        cleaned_path = file_path.replace('.csv', '_cleaned.csv')
        data.to_csv(cleaned_path, index=False)
        return cleaned_path
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return None

if __name__ == "__main__":
    cleaned_file = clean_dataset('sample_data.csv')
    if cleaned_file:
        print(f"Cleaned data saved to: {cleaned_file}")
import numpy as np

def remove_outliers_iqr(data, column):
    """
    Remove outliers from a pandas DataFrame column using the IQR method.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    return filtered_data

def calculate_basic_stats(data, column):
    """
    Calculate basic statistics for a column after outlier removal.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing statistical measures
    """
    stats = {
        'mean': np.mean(data[column]),
        'median': np.median(data[column]),
        'std': np.std(data[column]),
        'min': np.min(data[column]),
        'max': np.max(data[column]),
        'count': len(data[column])
    }
    
    return stats

def process_dataset(data, column):
    """
    Complete pipeline for processing a dataset column.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
    Returns:
    tuple: (cleaned_data, original_stats, cleaned_stats)
    """
    original_stats = calculate_basic_stats(data, column)
    cleaned_data = remove_outliers_iqr(data, column)
    cleaned_stats = calculate_basic_stats(cleaned_data, column)
    
    return cleaned_data, original_stats, cleaned_stats