
import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(df, numeric_columns):
    original_shape = df.shape
    for col in numeric_columns:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
    cleaned_shape = df.shape
    removed_count = original_shape[0] - cleaned_shape[0]
    print(f"Removed {removed_count} outliers from dataset")
    return df

def main():
    data = {
        'id': range(1, 101),
        'value': np.concatenate([
            np.random.normal(100, 10, 90),
            np.random.normal(300, 50, 10)
        ])
    }
    df = pd.DataFrame(data)
    print(f"Original dataset shape: {df.shape}")
    cleaned_df = clean_dataset(df, ['value'])
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Cleaned data statistics:")
    print(cleaned_df['value'].describe())

if __name__ == "__main__":
    main()