import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def normalize_minmax(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + '_normalized'] = (df[column] - min_val) / (max_val - min_val)
    return df

def clean_dataset(file_path, numeric_columns):
    df = pd.read_csv(file_path)
    
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
            df = normalize_minmax(df, col)
    
    cleaned_file = file_path.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_file, index=False)
    return cleaned_file

if __name__ == "__main__":
    data_file = "sample_data.csv"
    numeric_cols = ['age', 'income', 'score']
    
    try:
        result = clean_dataset(data_file, numeric_cols)
        print(f"Cleaned data saved to: {result}")
    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")