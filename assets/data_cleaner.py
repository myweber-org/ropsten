
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count()
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a dataset by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    cleaned_df = df.copy()
    all_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_stats(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            
            all_stats[column] = stats
    
    return cleaned_df, all_statsimport pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and standardizing columns.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV
    missing_strategy (str): Strategy for handling missing values
                           ('mean', 'median', 'drop', 'zero')
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    # Read input CSV
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Store original shape
    original_shape = df.shape
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Handle missing values based on strategy
    if missing_strategy == 'mean':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
    elif missing_strategy == 'median':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
    elif missing_strategy == 'zero':
        df.fillna(0, inplace=True)
    elif missing_strategy == 'drop':
        df.dropna(inplace=True)
    else:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Log cleaning statistics
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Rows removed: {original_shape[0] - df.shape[0]}")
    print(f"Columns processed: {len(numeric_cols)}")
    
    # Save cleaned data if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    required_columns (list): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    # Check if dataframe is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append('Dataframe is empty')
    
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing columns: {missing_cols}')
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f'Infinite values found: {inf_counts}')
    
    # Generate summary statistics
    validation_results['summary'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }
    
    return validation_results

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'ID': [1, 2, 3, 4, 5],
        'Value': [10.5, None, 15.2, None, 20.1],
        'Category': ['A', 'B', 'A', 'C', 'B']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_data.csv', index=False)
    
    # Clean the data
    cleaned_df = clean_csv_data('sample_data.csv', 
                               'cleaned_sample_data.csv',
                               missing_strategy='mean')
    
    # Validate the cleaned data
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
    print("\nValidation Results:")
    print(f"Is Valid: {validation['is_valid']}")
    print(f"Issues: {validation['issues']}")
    print(f"Summary: {validation['summary']}")import numpy as np
import pandas as pd
from scipy import stats

def remove_outliers_iqr(data, column, factor=1.5):
    """
    Remove outliers using the Interquartile Range method.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_zscore(data, column, threshold=3):
    """
    Remove outliers using Z-score method.
    """
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores < threshold]

def normalize_minmax(data, column):
    """
    Normalize data using Min-Max scaling.
    """
    min_val = data[column].min()
    max_val = data[column].max()
    data[column + '_normalized'] = (data[column] - min_val) / (max_val - min_val)
    return data

def normalize_zscore(data, column):
    """
    Normalize data using Z-score standardization.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    data[column + '_standardized'] = (data[column] - mean_val) / std_val
    return data

def clean_dataset(df, numeric_columns, method='iqr', normalization='minmax'):
    """
    Main cleaning function to process multiple numeric columns.
    """
    cleaned_df = df.copy()
    
    for col in numeric_columns:
        if method == 'iqr':
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
        elif method == 'zscore':
            cleaned_df = remove_outliers_zscore(cleaned_df, col)
        
        if normalization == 'minmax':
            cleaned_df = normalize_minmax(cleaned_df, col)
        elif normalization == 'zscore':
            cleaned_df = normalize_zscore(cleaned_df, col)
    
    return cleaned_df

def validate_data(df, required_columns, numeric_columns):
    """
    Validate dataset structure and content.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data