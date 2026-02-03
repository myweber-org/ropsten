
import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original data shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Remove rows where all values are NaN (if any remain)
        df = df.dropna(how='all')
        
        print(f"Final cleaned data shape: {df.shape}")
        print(f"Missing values after cleaning:\n{df.isnull().sum()}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df)
        print(f"Data validation result: {validation_result}")
import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return df[z_scores < threshold]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize_zscore(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()
    df[column + '_standardized'] = (df[column] - mean_val) / std_val
    return df

def clean_dataset(filepath, numeric_columns, outlier_method='iqr', normalize_method='minmax'):
    df = load_data(filepath)
    
    for col in numeric_columns:
        if outlier_method == 'iqr':
            df = remove_outliers_iqr(df, col)
        elif outlier_method == 'zscore':
            df = remove_outliers_zscore(df, col)
        
        if normalize_method == 'minmax':
            df = normalize_minmax(df, col)
        elif normalize_method == 'zscore':
            df = normalize_zscore(df, col)
    
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    cleaned_df = clean_dataset(
        input_file, 
        numeric_cols, 
        outlier_method='zscore', 
        normalize_method='zscore'
    )
    
    save_cleaned_data(cleaned_df, output_file)
    print(f"Original shape: {pd.read_csv(input_file).shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")