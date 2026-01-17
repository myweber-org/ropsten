
import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_column(df, column):
    """Normalize column using min-max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataframe(df, numeric_columns):
    """Apply cleaning operations to numeric columns."""
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_column(df, col)
    return df.reset_index(drop=True)

def save_cleaned_data(df, output_path):
    """Save cleaned dataframe to CSV."""
    df.to_csv(output_path, index=False)

def main():
    input_file = 'raw_data.csv'
    output_file = 'cleaned_data.csv'
    
    df = load_dataset(input_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    cleaned_df = clean_dataframe(df, numeric_cols)
    save_cleaned_data(cleaned_df, output_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
import pandas as pd
import sys

def remove_duplicates(input_file, output_file=None):
    """
    Read a CSV file, remove duplicate rows, and save the cleaned data.
    If no output file is specified, overwrite the input file.
    """
    try:
        df = pd.read_csv(input_file)
        initial_count = len(df)
        df_cleaned = df.drop_duplicates()
        final_count = len(df_cleaned)
        
        if output_file is None:
            output_file = input_file
        
        df_cleaned.to_csv(output_file, index=False)
        
        duplicates_removed = initial_count - final_count
        print(f"Successfully removed {duplicates_removed} duplicate row(s).")
        print(f"Cleaned data saved to: {output_file}")
        
        return duplicates_removed
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{input_file}' is empty.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    remove_duplicates(input_file, output_file)import numpy as np
import pandas as pd
from scipy import stats

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

def handle_missing_values(df, strategy='mean', columns=None):
    if columns is None:
        columns = df.columns
    
    df_filled = df.copy()
    for col in columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                fill_value = df[col].mean()
            elif strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mode':
                fill_value = df[col].mode()[0]
            else:
                fill_value = 0
            
            df_filled[col] = df[col].fillna(fill_value)
    
    return df_filled

def clean_dataset(df, outlier_method='iqr', normalize_method='minmax', missing_strategy='mean'):
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if outlier_method == 'iqr':
            df_clean = remove_outliers_iqr(df_clean, col)
        elif outlier_method == 'zscore':
            df_clean = remove_outliers_zscore(df_clean, col)
    
    for col in numeric_cols:
        if normalize_method == 'minmax':
            df_clean = normalize_minmax(df_clean, col)
        elif normalize_method == 'zscore':
            df_clean = normalize_zscore(df_clean, col)
    
    df_clean = handle_missing_values(df_clean, strategy=missing_strategy)
    
    return df_clean