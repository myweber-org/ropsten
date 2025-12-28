
def deduplicate_list(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
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
    
    return filtered_df

def calculate_summary_stats(df, column):
    """
    Calculate summary statistics for a column.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name
    
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

def clean_dataset(df, numeric_columns):
    """
    Clean a dataset by removing outliers from multiple numeric columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    numeric_columns (list): List of numeric column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            print(f"Removed {removed_count} outliers from column '{column}'")
    
    return cleaned_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    data = {
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 1, 100)
    }
    
    # Introduce some outliers
    data['value'][10] = 500
    data['value'][20] = -200
    data['score'][30] = 5.0
    
    df = pd.DataFrame(data)
    
    print("Original dataset shape:", df.shape)
    print("\nOriginal summary statistics for 'value':")
    print(calculate_summary_stats(df, 'value'))
    
    cleaned_df = clean_dataset(df, ['value', 'score'])
    
    print("\nCleaned dataset shape:", cleaned_df.shape)
    print("\nCleaned summary statistics for 'value':")
    print(calculate_summary_stats(cleaned_df, 'value'))import pandas as pd
import numpy as np

def load_data(filepath):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def remove_outliers(df, column, threshold=3):
    """Remove outliers using z-score method."""
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe.")
        return df
    
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    filtered_df = df[z_scores < threshold]
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} outliers from column '{column}'.")
    return filtered_df

def normalize_column(df, column):
    """Normalize column values to range [0, 1]."""
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe.")
        return df
    
    min_val = df[column].min()
    max_val = df[column].max()
    
    if max_val == min_val:
        print(f"Column '{column}' has constant values. Skipping normalization.")
        return df
    
    df[column] = (df[column] - min_val) / (max_val - min_val)
    print(f"Normalized column '{column}' to range [0, 1].")
    return df

def clean_dataset(df, numeric_columns):
    """Apply cleaning operations to dataset."""
    if df is None or df.empty:
        print("Dataframe is empty or None.")
        return df
    
    cleaned_df = df.copy()
    
    for column in numeric_columns:
        if column in cleaned_df.columns:
            cleaned_df = remove_outliers(cleaned_df, column)
            cleaned_df = normalize_column(cleaned_df, column)
    
    print(f"Data cleaning complete. Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def save_cleaned_data(df, output_path):
    """Save cleaned dataframe to CSV."""
    try:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    """Main execution function."""
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    numeric_cols = ["age", "income", "score"]
    
    raw_data = load_data(input_file)
    
    if raw_data is not None:
        cleaned_data = clean_dataset(raw_data, numeric_cols)
        save_cleaned_data(cleaned_data, output_file)

if __name__ == "__main__":
    main()