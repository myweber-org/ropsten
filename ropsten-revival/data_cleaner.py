
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