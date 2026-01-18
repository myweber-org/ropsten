
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to process
    
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

def process_dataset(file_path, column_name):
    """
    Complete pipeline to load data, remove outliers, and return cleaned data.
    
    Parameters:
    file_path (str): Path to CSV file
    column_name (str): Column to clean
    
    Returns:
    tuple: (cleaned_df, original_stats, cleaned_stats)
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    original_stats = calculate_summary_stats(df, column_name)
    cleaned_df = remove_outliers_iqr(df, column_name)
    cleaned_stats = calculate_summary_stats(cleaned_df, column_name)
    
    return cleaned_df, original_stats, cleaned_stats

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'values': np.concatenate([
            np.random.normal(100, 15, 90),
            np.random.normal(300, 50, 10)
        ])
    })
    
    print("Original data shape:", sample_data.shape)
    print("Original statistics:", calculate_summary_stats(sample_data, 'values'))
    
    cleaned_data = remove_outliers_iqr(sample_data, 'values')
    print("\nCleaned data shape:", cleaned_data.shape)
    print("Cleaned statistics:", calculate_summary_stats(cleaned_data, 'values'))
import pandas as pd
import re

def clean_dataframe(df, text_column='text'):
    """
    Clean a DataFrame by removing duplicates and normalizing text in a specified column.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates().reset_index(drop=True)
    
    # Normalize text: lowercase and remove extra whitespace
    def normalize_text(text):
        if pd.isna(text):
            return text
        text = str(text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df_clean[text_column] = df_clean[text_column].apply(normalize_text)
    
    return df_clean

def save_cleaned_data(df, input_path, suffix='_cleaned'):
    """
    Save the cleaned DataFrame to a new CSV file.
    """
    if input_path.endswith('.csv'):
        output_path = input_path.replace('.csv', f'{suffix}.csv')
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    else:
        print("Input path must be a CSV file.")

if __name__ == "__main__":
    # Example usage
    input_file = 'raw_data.csv'
    try:
        raw_df = pd.read_csv(input_file)
        cleaned_df = clean_dataframe(raw_df, text_column='review')
        save_cleaned_data(cleaned_df, input_file)
    except FileNotFoundError:
        print(f"File {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")