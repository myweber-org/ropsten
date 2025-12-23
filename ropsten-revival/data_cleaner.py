import csv
import sys

def clean_csv(input_file, output_file):
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            data = list(reader)
        
        cleaned_data = []
        for row in data:
            if len(row) == len(headers):
                cleaned_row = [cell.strip() for cell in row]
                cleaned_data.append(cleaned_row)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            writer.writerows(cleaned_data)
        
        print(f"Cleaned data saved to {output_file}")
        return True
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_csv(input_file, output_file)
import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values. 
                            Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Number of standard deviations for outlier detection.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    
    cleaned_df = df.copy()
    
    # Handle missing values
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        if cleaned_df[column].isnull().any():
            if missing_strategy == 'mean':
                cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
            elif missing_strategy == 'median':
                cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
            elif missing_strategy == 'mode':
                cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
            elif missing_strategy == 'drop':
                cleaned_df.dropna(subset=[column], inplace=True)
    
    # Handle outliers using Z-score method
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        z_scores = np.abs((cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std())
        outliers = z_scores > outlier_threshold
        if outliers.any():
            # Cap outliers at threshold * standard deviation
            upper_bound = cleaned_df[column].mean() + outlier_threshold * cleaned_df[column].std()
            lower_bound = cleaned_df[column].mean() - outlier_threshold * cleaned_df[column].std()
            cleaned_df[column] = cleaned_df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return cleaned_df

def validate_dataframe(df, required_columns=None):
    """
    Validate DataFrame structure and content.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate.
    required_columns (list): List of required column names.
    
    Returns:
    tuple: (is_valid, error_message)
    """
    
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
    
    return True, "DataFrame is valid"

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    sample_data = {
        'A': [1, 2, np.nan, 4, 100],  # Contains NaN and outlier
        'B': [5, 6, 7, np.nan, 9],
        'C': [10, 11, 12, 13, 14]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    
    # Clean the data
    cleaned = clean_dataframe(df, missing_strategy='mean', outlier_threshold=2)
    print("\nCleaned DataFrame:")
    print(cleaned)
    
    # Validate the cleaned data
    is_valid, message = validate_dataframe(cleaned, required_columns=['A', 'B', 'C'])
    print(f"\nValidation: {message}")