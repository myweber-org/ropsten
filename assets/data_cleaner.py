
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
    cleaned_df = cleaned_df.dropna()
    cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'A': np.random.normal(100, 15, 200),
        'B': np.random.exponential(50, 200),
        'C': np.random.uniform(0, 1, 200)
    }
    sample_data['A'][10] = 500
    sample_data['B'][20] = -100
    df = pd.DataFrame(sample_data)
    print("Original shape:", df.shape)
    cleaned = clean_dataset(df, ['A', 'B', 'C'])
    print("Cleaned shape:", cleaned.shape)
import pandas as pd
import numpy as np

def clean_csv_data(file_path, fill_strategy='mean', output_path=None):
    """
    Load a CSV file, clean missing values, and optionally save cleaned data.
    
    Parameters:
    file_path (str): Path to input CSV file
    fill_strategy (str): Strategy for filling missing values ('mean', 'median', 'mode', 'zero')
    output_path (str): Optional path to save cleaned CSV
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            
            if fill_strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif fill_strategy == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif fill_strategy == 'mode':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
            elif fill_strategy == 'zero':
                df[numeric_cols] = df[numeric_cols].fillna(0)
            
            df[categorical_cols] = df[categorical_cols].fillna('Unknown')
            print("Missing values filled")
        
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

def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method for a specific column.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    column (str): Column name to check for outliers
    
    Returns:
    tuple: (lower_bound, upper_bound, outlier_indices)
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame")
        return None, None, []
    
    if not np.issubdtype(df[column].dtype, np.number):
        print(f"Column '{column}' is not numeric")
        return None, None, []
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return lower_bound, upper_bound, outliers.index.tolist()

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': ['X', 'Y', 'Z', np.nan, 'W']
    })
    
    sample_data.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', fill_strategy='mean', output_path='cleaned_data.csv')
    
    if cleaned_df is not None:
        print("Data cleaning completed successfully")
        print("Cleaned DataFrame:")
        print(cleaned_df)