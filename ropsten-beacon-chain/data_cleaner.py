
import pandas as pd
import numpy as np

def clean_dataset(df):
    """
    Clean dataset by removing duplicates, standardizing column names,
    and handling missing values.
    """
    # Remove duplicate rows
    df_clean = df.drop_duplicates()
    
    # Standardize column names: lowercase and replace spaces with underscores
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
    
    # Fill missing numeric values with column median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Fill missing categorical values with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'unknown')
    
    return df_clean

def validate_data(df):
    """
    Validate data by checking for remaining nulls and data types.
    """
    validation_report = {}
    
    # Check for null values
    null_counts = df.isnull().sum()
    validation_report['null_counts'] = null_counts[null_counts > 0].to_dict()
    
    # Check data types
    validation_report['dtypes'] = df.dtypes.to_dict()
    
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    negative_counts = {}
    for col in numeric_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            negative_counts[col] = negative_count
    validation_report['negative_counts'] = negative_counts
    
    return validation_report

if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'Customer ID': [1, 2, 2, 3, 4],
        'Order Value': [100, 200, 200, np.nan, 300],
        'Product Category': ['A', 'B', 'B', None, 'C'],
        'Discount Applied': [0.1, -0.2, 0.15, 0.0, 0.05]
    })
    
    print("Original dataset:")
    print(sample_data)
    print("\nCleaned dataset:")
    cleaned_data = clean_dataset(sample_data)
    print(cleaned_data)
    
    print("\nValidation report:")
    report = validate_data(cleaned_data)
    for key, value in report.items():
        print(f"{key}: {value}")import pandas as pd
import numpy as np

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def clean_dataset(input_file, output_file):
    df = pd.read_csv(input_file)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        df = remove_outliers_iqr(df, col)
    
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    print(f"Original rows: {len(pd.read_csv(input_file))}, Cleaned rows: {len(df)}")

if __name__ == "__main__":
    clean_dataset("raw_data.csv", "cleaned_data.csv")