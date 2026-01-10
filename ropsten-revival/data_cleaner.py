
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    min_val = data[column].min()
    max_val = data[column].max()
    data[column] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, outlier_method='iqr', normalize_method='zscore'):
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif outlier_method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalize_method == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalize_method == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df.reset_index(drop=True)

def validate_cleaning(original_df, cleaned_df, numeric_columns):
    report = {}
    for col in numeric_columns:
        report[col] = {
            'original_mean': original_df[col].mean(),
            'cleaned_mean': cleaned_df[col].mean(),
            'original_std': original_df[col].std(),
            'cleaned_std': cleaned_df[col].std(),
            'rows_removed': len(original_df) - len(cleaned_df)
        }
    return pd.DataFrame(report).T

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000),
        'feature3': np.random.uniform(0, 200, 1000)
    })
    
    cleaned_data = clean_dataset(
        sample_data, 
        ['feature1', 'feature2', 'feature3'],
        outlier_method='zscore',
        normalize_method='minmax'
    )
    
    validation_report = validate_cleaning(sample_data, cleaned_data, ['feature1', 'feature2', 'feature3'])
    print("Data cleaning completed successfully.")
    print(f"Original rows: {len(sample_data)}")
    print(f"Cleaned rows: {len(cleaned_data)}")
    print("\nValidation Report:")
    print(validation_report)