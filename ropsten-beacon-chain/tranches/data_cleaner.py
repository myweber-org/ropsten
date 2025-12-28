import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to process.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

def clean_dataset(file_path, output_path=None):
    """
    Load a CSV file, clean numeric columns by removing outliers, and save the result.
    
    Parameters:
    file_path (str): Path to the input CSV file.
    output_path (str, optional): Path to save the cleaned CSV. If None, returns DataFrame.
    
    Returns:
    pd.DataFrame or None: Cleaned DataFrame if output_path is None, else saves to file.
    """
    df = pd.read_csv(file_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        df = remove_outliers_iqr(df, col)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return None
    else:
        return df

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    try:
        clean_dataset(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")