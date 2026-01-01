import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Args:
        file_path (str): Path to input CSV file
        output_path (str, optional): Path for cleaned output CSV
        missing_strategy (str): Strategy for handling missing values
                               ('mean', 'median', 'drop', 'zero')
    
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    
    try:
        df = pd.read_csv(file_path)
        
        print(f"Original data shape: {df.shape}")
        print(f"Missing values per column:\n{df.isnull().sum()}")
        
        df_cleaned = df.copy()
        
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"After removing duplicates: {df_cleaned.shape}")
        
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_cleaned[col].isnull().any():
                if missing_strategy == 'mean':
                    fill_value = df_cleaned[col].mean()
                elif missing_strategy == 'median':
                    fill_value = df_cleaned[col].median()
                elif missing_strategy == 'zero':
                    fill_value = 0
                elif missing_strategy == 'drop':
                    df_cleaned = df_cleaned.dropna(subset=[col])
                    continue
                else:
                    raise ValueError(f"Unknown strategy: {missing_strategy}")
                
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
                print(f"Filled missing values in '{col}' with {missing_strategy}: {fill_value}")
        
        if output_path:
            df_cleaned.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        print(f"Final data shape: {df_cleaned.shape}")
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Args:
        df (pandas.DataFrame): Dataframe to validate
        required_columns (list): List of required column names
    
    Returns:
        bool: True if validation passes
    """
    if df is None or df.empty:
        print("Error: Dataframe is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    print("Data validation passed")
    return True

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'value': [10.5, None, 15.2, 10.5, None, 18.7],
        'category': ['A', 'B', 'A', 'C', 'B', 'A']
    }
    
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_data.csv', index=False)
    
    cleaned_df = clean_csv_data('sample_data.csv', 'cleaned_sample.csv', 'mean')
    
    if cleaned_df is not None:
        validation_result = validate_dataframe(cleaned_df, ['id', 'value', 'category'])
        print(f"Validation result: {validation_result}")
import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if cleaned_df.isnull().sum().any():
        print("Handling missing values...")
        for column in cleaned_df.columns:
            if cleaned_df[column].isnull().sum() > 0:
                if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    fill_value = cleaned_df[column].mean()
                    cleaned_df[column].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in '{column}' with mean: {fill_value:.2f}")
                elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(cleaned_df[column]):
                    fill_value = cleaned_df[column].median()
                    cleaned_df[column].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in '{column}' with median: {fill_value:.2f}")
                elif fill_missing == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                    cleaned_df[column].fillna(fill_value, inplace=True)
                    print(f"Filled missing values in '{column}' with mode: {fill_value}")
                else:
                    cleaned_df[column].fillna(0, inplace=True)
                    print(f"Filled missing values in '{column}' with 0")
    
    print(f"Data cleaning complete. Final shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df):
    """
    Validate the cleaned dataset for basic integrity.
    """
    validation_results = {
        'has_duplicates': df.duplicated().any(),
        'has_missing_values': df.isnull().sum().any(),
        'total_rows': df.shape[0],
        'total_columns': df.shape[1],
        'data_types': df.dtypes.to_dict()
    }
    
    print("Dataset Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value}")
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'A': [1, 2, 2, 4, 5, None, 7],
        'B': [10, 20, 20, None, 50, 60, 70],
        'C': ['x', 'y', 'y', 'z', None, 'x', 'z']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, drop_duplicates=True, fill_missing='mean')
    print("\nCleaned dataset:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    validation = validate_dataset(cleaned_df)