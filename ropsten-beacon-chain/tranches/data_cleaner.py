
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using Interquartile Range method
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    outliers_removed = len(data) - len(filtered_data)
    
    return filtered_data, outliers_removed

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return data[column].apply(lambda x: 0.5)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return data[column].apply(lambda x: 0)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in numerical columns
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
    elif strategy == 'mode':
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
    elif strategy == 'drop':
        data = data.dropna(subset=numeric_cols)
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    
    return data

def detect_skewness(data, column, threshold=0.5):
    """
    Detect skewness in data column
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    skewness = stats.skew(data[column].dropna())
    is_skewed = abs(skewness) > threshold
    
    return {
        'skewness': skewness,
        'is_skewed': is_skewed,
        'interpretation': 'positively skewed' if skewness > 0 else 'negatively skewed' if skewness < 0 else 'symmetric'
    }

def log_transform(data, column):
    """
    Apply log transformation to reduce skewness
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if data[column].min() <= 0:
        shifted_data = data[column] - data[column].min() + 1
        transformed = np.log(shifted_data)
    else:
        transformed = np.log(data[column])
    
    return transformed

def create_summary_report(data):
    """
    Create a comprehensive data quality report
    """
    report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict(),
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    
    for col in numeric_cols:
        report['numeric_summary'][col] = {
            'mean': data[col].mean(),
            'median': data[col].median(),
            'std': data[col].std(),
            'min': data[col].min(),
            'max': data[col].max(),
            'skewness': stats.skew(data[col].dropna())
        }
    
    for col in categorical_cols:
        report['categorical_summary'][col] = {
            'unique_values': data[col].nunique(),
            'top_value': data[col].mode()[0] if len(data[col].mode()) > 0 else None,
            'top_count': data[col].value_counts().iloc[0] if len(data[col]) > 0 else 0
        }
    
    return report

def validate_dataframe(data):
    """
    Validate DataFrame structure and content
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("DataFrame is empty")
    
    if len(data.columns) == 0:
        raise ValueError("DataFrame has no columns")
    
    return True
import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_duplicates(df, column_name):
    df[column_name] = df[column_name].apply(clean_text)
    df = df.drop_duplicates(subset=[column_name], keep='first')
    return df

def process_data(input_file, output_file, column_to_clean):
    try:
        df = pd.read_csv(input_file)
        df_cleaned = remove_duplicates(df, column_to_clean)
        df_cleaned.to_csv(output_file, index=False)
        print(f"Data cleaned and saved to {output_file}")
        print(f"Removed {len(df) - len(df_cleaned)} duplicate entries")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    process_data("raw_data.csv", "cleaned_data.csv", "description")