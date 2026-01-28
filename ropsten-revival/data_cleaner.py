
import pandas as pd
import numpy as np
from scipy import stats

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

def clean_dataset(filepath, columns_to_clean):
    df = pd.read_csv(filepath)
    
    for column in columns_to_clean:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
            df = normalize_minmax(df, column)
    
    cleaned_filepath = filepath.replace('.csv', '_cleaned.csv')
    df.to_csv(cleaned_filepath, index=False)
    return cleaned_filepath

if __name__ == "__main__":
    data_file = "raw_data.csv"
    columns = ["temperature", "pressure", "humidity"]
    result = clean_dataset(data_file, columns)
    print(f"Cleaned data saved to: {result}")