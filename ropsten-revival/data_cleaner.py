
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
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(df, numeric_columns):
    cleaned_df = df.copy()
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df = remove_outliers_iqr(cleaned_df, col)
            cleaned_df = normalize_column(cleaned_df, col)
    return cleaned_df

if __name__ == "__main__":
    sample_data = {
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.exponential(50, 200),
        'category': np.random.choice(['A', 'B', 'C'], 200)
    }
    df = pd.DataFrame(sample_data)
    numeric_cols = ['feature1', 'feature2']
    result_df = clean_dataset(df, numeric_cols)
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {result_df.shape}")
    print(result_df.head())
def remove_duplicates_preserve_order(sequence):
    seen = set()
    result = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset shape: {df.shape}")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, col)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column '{col}'")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to {output_file}")
        print(f"Final dataset shape: {df.shape}")
        
        return True
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    input_path = "raw_data.csv"
    output_path = "cleaned_data.csv"
    clean_dataset(input_path, output_path)
import pandas as pd

def clean_dataset(df, missing_strategy='drop', duplicate_strategy='first'):
    """
    Clean a pandas DataFrame by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean
        missing_strategy (str): Strategy for handling missing values. 
                               Options: 'drop', 'fill_mean', 'fill_median', 'fill_mode'
        duplicate_strategy (str): Strategy for handling duplicates.
                                 Options: 'first', 'last', 'drop_all'
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    if missing_strategy == 'drop':
        df_clean = df_clean.dropna()
    elif missing_strategy == 'fill_mean':
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif missing_strategy == 'fill_median':
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif missing_strategy == 'fill_mode':
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    
    # Handle duplicates
    if duplicate_strategy == 'first':
        df_clean = df_clean.drop_duplicates(keep='first')
    elif duplicate_strategy == 'last':
        df_clean = df_clean.drop_duplicates(keep='last')
    elif duplicate_strategy == 'drop_all':
        df_clean = df_clean.drop_duplicates(keep=False)
    
    return df_clean

def validate_dataframe(df, required_columns=None):
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of column names that must be present
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers. If None, uses all numeric columns.
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    df_clean = df.copy()
    
    for column in columns:
        if column in df_clean.columns:
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            mask = (df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'A': [1, 2, None, 4, 5, 5, 100],
        'B': [10, 20, 30, 40, 50, 50, 60],
        'C': ['x', 'y', 'z', 'x', 'y', 'y', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, missing_strategy='fill_mean', duplicate_strategy='first')
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n")
    
    # Validate
    is_valid, message = validate_dataframe(cleaned_df, required_columns=['A', 'B'])
    print(f"Validation: {is_valid} - {message}")
    print("\n")
    
    # Remove outliers
    outlier_free_df = remove_outliers_iqr(cleaned_df, columns=['A', 'B'])
    print("DataFrame after outlier removal:")
    print(outlier_free_df)