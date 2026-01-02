
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    if max_val != min_val:
        df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_column(df, col)
    df = df.dropna()
    return df.reset_index(drop=True)

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    sample_data.loc[10, 'feature1'] = 500
    sample_data.loc[20, 'feature2'] = 1000
    
    cleaned_data = clean_dataset(sample_data, ['feature1', 'feature2'])
    print(f"Original shape: {sample_data.shape}")
    print(f"Cleaned shape: {cleaned_data.shape}")
    print(cleaned_data.describe())
import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to process
    factor (float): Multiplier for IQR
    
    Returns:
    pd.DataFrame: Dataframe with outliers removed
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to normalize
    
    Returns:
    pd.Series: Normalized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    min_val = data[column].min()
    max_val = data[column].max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    normalized = (data[column] - min_val) / (max_val - min_val)
    return normalized

def standardize_zscore(data, column):
    """
    Standardize data using Z-score normalization.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    column (str): Column name to standardize
    
    Returns:
    pd.Series: Standardized values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    mean_val = data[column].mean()
    std_val = data[column].std()
    
    if std_val == 0:
        return pd.Series([0] * len(data), index=data.index)
    
    standardized = (data[column] - mean_val) / std_val
    return standardized

def handle_missing_values(data, strategy='mean', columns=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    strategy (str): Imputation strategy ('mean', 'median', 'mode', 'constant')
    columns (list): List of columns to process, None for all numeric columns
    
    Returns:
    pd.DataFrame: Dataframe with missing values handled
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    result = data.copy()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if result[col].isnull().sum() == 0:
            continue
            
        if strategy == 'mean':
            fill_value = result[col].mean()
        elif strategy == 'median':
            fill_value = result[col].median()
        elif strategy == 'mode':
            fill_value = result[col].mode()[0] if not result[col].mode().empty else 0
        elif strategy == 'constant':
            fill_value = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        result[col] = result[col].fillna(fill_value)
    
    return result

def create_data_summary(data):
    """
    Create a comprehensive summary of the dataframe.
    
    Parameters:
    data (pd.DataFrame): Input dataframe
    
    Returns:
    dict: Summary statistics
    """
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'min': data[col].min(),
            '25%': data[col].quantile(0.25),
            '50%': data[col].median(),
            '75%': data[col].quantile(0.75),
            'max': data[col].max()
        }
    
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = {
            'unique_count': data[col].nunique(),
            'top_value': data[col].mode()[0] if not data[col].mode().empty else None,
            'top_count': data[col].value_counts().iloc[0] if not data[col].empty else 0
        }
    
    return summary