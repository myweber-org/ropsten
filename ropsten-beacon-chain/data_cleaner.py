
import numpy as np

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Original dataset shape: {data.shape}")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            original_len = len(data)
            data = remove_outliers_iqr(data, col)
            removed = original_len - len(data)
            if removed > 0:
                print(f"Removed {removed} outliers from column: {col}")
        
        print(f"Cleaned dataset shape: {data.shape}")
        return data
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None

if __name__ == "__main__":
    cleaned_data = clean_dataset("sample_data.csv")
    if cleaned_data is not None:
        cleaned_data.to_csv("cleaned_data.csv", index=False)
        print("Cleaned data saved to cleaned_data.csv")