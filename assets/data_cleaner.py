
import pandas as pd
import numpy as np

def clean_dataframe(df):
    """
    Clean a pandas DataFrame by handling missing values and standardizing text.
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Fill numeric nulls with column mean
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    
    # Fill categorical nulls with 'Unknown'
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    # Standardize text columns: trim whitespace and convert to lowercase
    for col in categorical_cols:
        cleaned_df[col] = cleaned_df[col].str.strip().str.lower()
    
    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    
    return cleaned_df

def validate_dataframe(df):
    """
    Validate that DataFrame meets basic quality requirements.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for remaining null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        problematic_cols = null_counts[null_counts > 0].index.tolist()
        print(f"Warning: Columns with null values: {problematic_cols}")
    
    return True

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'name': ['Alice', 'Bob', None, 'Charlie', 'Alice'],
        'age': [25, None, 30, 35, 25],
        'city': ['New York', '  los angeles  ', 'Chicago', None, 'New York']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\nCleaned DataFrame:")
    cleaned = clean_dataframe(df)
    print(cleaned)
    
    try:
        validate_dataframe(cleaned)
        print("\nData validation passed")
    except ValueError as e:
        print(f"\nData validation failed: {e}")import pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape}")
        
        # Fill missing numeric values with column median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing values in {col} with median: {median_val}")
        
        # Fill missing categorical values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
                print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Remove columns with more than 50% missing values
        threshold = len(df) * 0.5
        cols_to_drop = [col for col in df.columns if df[col].isnull().sum() > threshold]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped columns with >50% missing values: {cols_to_drop}")
        
        print(f"Final cleaned shape: {df.shape}")
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to: {output_path}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

def validate_dataframe(df, required_columns=None):
    """
    Validate the structure and content of a DataFrame.
    """
    if df is None or df.empty:
        print("DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            return False
    
    # Check for any remaining NaN values
    if df.isnull().any().any():
        print("Warning: DataFrame still contains NaN values")
    
    return True

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    
    if cleaned_df is not None:
        is_valid = validate_dataframe(cleaned_df)
        if is_valid:
            print("Data cleaning completed successfully")
        else:
            print("Data validation failed")