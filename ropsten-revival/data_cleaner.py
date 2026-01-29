
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    original_shape = df.shape
    
    if drop_duplicates:
        df = df.drop_duplicates()
        print(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
    
    if fill_missing:
        for column in df.columns:
            if df[column].isnull().any():
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                    fill_value = df[column].mean()
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                    fill_value = df[column].median()
                elif fill_missing == 'mode':
                    fill_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
                else:
                    fill_value = 0 if pd.api.types.is_numeric_dtype(df[column]) else 'Unknown'
                
                df[column] = df[column].fillna(fill_value)
                print(f"Filled missing values in column '{column}' with {fill_missing}: {fill_value}")
    
    print(f"Dataset cleaned: {original_shape} -> {df.shape}")
    return df

def validate_data(df, required_columns=None):
    """
    Validate the DataFrame for required columns and data types.
    """
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict()
    }
    
    return validation_report

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', None, 'Eve', 'Frank'],
        'age': [25, 30, 30, 35, None, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, 95.5]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    
    report = validate_data(cleaned_df, required_columns=['id', 'name', 'age', 'score'])
    print("\nValidation report:")
    for key, value in report.items():
        print(f"{key}: {value}")
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
        
        for column in numeric_columns:
            original_count = len(df)
            df = remove_outliers_iqr(df, column)
            removed_count = original_count - len(df)
            if removed_count > 0:
                print(f"Removed {removed_count} outliers from column: {column}")
        
        df.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")
        print(f"Final dataset shape: {df.shape}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return False

if __name__ == "__main__":
    clean_dataset("raw_data.csv", "cleaned_data.csv")