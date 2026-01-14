
import csv
import sys

def remove_duplicates(input_file, output_file, key_columns):
    """
    Remove duplicate rows from a CSV file based on specified key columns.
    Keep the first occurrence of each duplicate.
    """
    seen = set()
    unique_rows = []
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # Create a tuple of values from the key columns
            key = tuple(row[col] for col in key_columns)
            
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)
    
    return len(unique_rows)

def main():
    if len(sys.argv) < 4:
        print("Usage: python data_cleaner.py <input_file> <output_file> <key_column1> [key_column2 ...]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    key_columns = sys.argv[3:]
    
    try:
        count = remove_duplicates(input_file, output_file, key_columns)
        print(f"Processed {count} unique rows. Output saved to {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except KeyError as e:
        print(f"Error: Key column {e} not found in CSV header.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np
from scipy import stats

def load_data(filepath):
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

def zscore_normalize(df, column):
    """Normalize column using Z-score."""
    df[column + '_normalized'] = stats.zscore(df[column])
    return df

def minmax_normalize(df, column):
    """Normalize column using Min-Max scaling."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_scaled'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(input_file, output_file):
    """Main cleaning pipeline."""
    df = load_data(input_file)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    for col in numeric_cols:
        df = zscore_normalize(df, col)
        df = minmax_normalize(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    return df

if __name__ == "__main__":
    clean_dataset('raw_data.csv', 'cleaned_data.csv')
import numpy as np
import pandas as pd
from scipy import stats

def normalize_data(data, method='zscore'):
    """
    Normalize data using specified method.
    
    Args:
        data: numpy array or pandas Series
        method: 'zscore', 'minmax', or 'robust'
    
    Returns:
        Normalized data
    """
    if method == 'zscore':
        return (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'robust':
        return (data - np.median(data)) / stats.iqr(data)
    else:
        raise ValueError("Method must be 'zscore', 'minmax', or 'robust'")

def remove_outliers_iqr(data, threshold=1.5):
    """
    Remove outliers using IQR method.
    
    Args:
        data: numpy array or pandas Series
        threshold: multiplier for IQR
    
    Returns:
        Data with outliers removed
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def clean_dataset(df, columns=None, normalize=True, remove_outliers=True):
    """
    Clean dataset by normalizing and removing outliers.
    
    Args:
        df: pandas DataFrame
        columns: list of columns to clean (default: all numeric columns)
        normalize: whether to normalize data
        remove_outliers: whether to remove outliers
    
    Returns:
        Cleaned DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    cleaned_df = df.copy()
    
    for col in columns:
        if remove_outliers:
            cleaned_df[col] = remove_outliers_iqr(cleaned_df[col])
        
        if normalize and not cleaned_df[col].empty:
            cleaned_df[col] = normalize_data(cleaned_df[col], method='zscore')
    
    return cleaned_df.dropna()

def calculate_statistics(data):
    """
    Calculate descriptive statistics.
    
    Args:
        data: numpy array or pandas Series
    
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75)
    }

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.exponential(50, 1000)
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:")
    print(sample_data.describe())
    
    cleaned_data = clean_dataset(sample_data)
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned statistics:")
    print(cleaned_data.describe())
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns
        
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        if columns is None:
            columns = self.numeric_columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                df_clean = df_clean[mask]
        
        return df_clean
    
    def remove_outliers_zscore(self, columns=None, threshold=3):
        if columns is None:
            columns = self.numeric_columns
            
        df_clean = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                mask = z_scores < threshold
                df_clean = df_clean[mask]
        
        return df_clean
    
    def normalize_minmax(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        return df_normalized
    
    def normalize_zscore(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        df_normalized = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
        
        return df_normalized
    
    def fill_missing_mean(self, columns=None):
        if columns is None:
            columns = self.numeric_columns
            
        df_filled = self.df.copy()
        for col in columns:
            if col in self.numeric_columns:
                mean_val = df_filled[col].mean()
                df_filled[col] = df_filled[col].fillna(mean_val)
        
        return df_filled
    
    def get_summary(self):
        summary = {
            'original_shape': self.df.shape,
            'numeric_columns': list(self.numeric_columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary

def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'date': dates,
        'value_a': np.random.normal(100, 15, 100),
        'value_b': np.random.exponential(50, 100),
        'value_c': np.random.uniform(0, 200, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 10), 'value_a'] = np.nan
    df.loc[np.random.choice(df.index, 5), 'value_b'] = 500
    
    return df

if __name__ == "__main__":
    sample_df = create_sample_data()
    cleaner = DataCleaner(sample_df)
    
    print("Original data shape:", cleaner.df.shape)
    print("\nSummary:")
    summary = cleaner.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cleaned_df = cleaner.remove_outliers_iqr(['value_a', 'value_b', 'value_c'])
    print(f"\nAfter IQR outlier removal: {cleaned_df.shape}")
    
    normalized_df = cleaner.normalize_minmax()
    print(f"After min-max normalization: {normalized_df.shape}")
    
    filled_df = cleaner.fill_missing_mean()
    print(f"After filling missing values: {filled_df.shape}")