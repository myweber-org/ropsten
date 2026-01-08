
import pandas as pd
import numpy as np

def clean_dataset(df, duplicate_threshold=0.8, missing_strategy='median'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Args:
        df: Input DataFrame
        duplicate_threshold: Similarity threshold for duplicate detection (0-1)
        missing_strategy: Strategy for handling missing values ('median', 'mean', 'mode', 'drop')
    
    Returns:
        Cleaned DataFrame
    """
    original_shape = df.shape
    
    # Remove exact duplicates
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    
    # Remove approximate duplicates based on similarity threshold
    if duplicate_threshold < 1.0:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Calculate similarity matrix for numeric columns
            numeric_data = df_cleaned[numeric_cols].fillna(0)
            similarity_matrix = cosine_similarity(numeric_data)
            
            # Find duplicates above threshold
            duplicates = set()
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i, j] > duplicate_threshold:
                        duplicates.add(j)
            
            # Remove duplicate indices
            df_cleaned = df_cleaned.drop(index=list(duplicates)).reset_index(drop=True)
    
    # Handle missing values
    if missing_strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
    elif missing_strategy in ['median', 'mean']:
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if missing_strategy == 'median':
                fill_value = df_cleaned[col].median()
            else:  # mean
                fill_value = df_cleaned[col].mean()
            df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    elif missing_strategy == 'mode':
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                fill_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else ''
                df_cleaned[col] = df_cleaned[col].fillna(fill_value)
    
    # Log cleaning statistics
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Rows removed: {original_shape[0] - df_cleaned.shape[0]}")
    print(f"Columns: {df_cleaned.shape[1]}")
    
    return df_cleaned

def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
    
    Returns:
        Boolean indicating if validation passed
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None")
        return False
    
    if len(df) < min_rows:
        print(f"Error: DataFrame has fewer than {min_rows} rows")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return False
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            print(f"Warning: Column '{col}' contains infinite values")
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'value_a': [10.5, 20.3, 10.5, 15.7, np.nan, 20.3, 25.1, 30.0],
        'value_b': [100, 200, 100, 150, 250, 200, 300, 350],
        'category': ['A', 'B', 'A', 'C', 'B', 'B', 'A', 'C']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the data
    cleaned_df = clean_dataset(df, duplicate_threshold=0.9, missing_strategy='median')
    
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    # Validate the cleaned data
    is_valid = validate_dataframe(cleaned_df, required_columns=['id', 'value_a', 'value_b'], min_rows=1)
    print(f"\nData validation passed: {is_valid}")