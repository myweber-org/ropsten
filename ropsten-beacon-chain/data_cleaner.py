import pandas as pd
import numpy as np

def clean_dataset(df, drop_duplicates=True, fill_missing=True, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    """
    cleaned_df = df.copy()
    
    if drop_duplicates:
        initial_rows = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates()
        removed = initial_rows - cleaned_df.shape[0]
        print(f"Removed {removed} duplicate rows.")
    
    if fill_missing and cleaned_df.isnull().sum().any():
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[col].mean()
                elif fill_strategy == 'median':
                    fill_value = cleaned_df[col].median()
                elif fill_strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = cleaned_df[col].mean()
                
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                print(f"Filled missing values in '{col}' with {fill_strategy}: {fill_value}")
        
        for col in categorical_cols:
            if cleaned_df[col].isnull().any():
                mode_value = cleaned_df[col].mode()
                if not mode_value.empty:
                    fill_value = mode_value[0]
                    cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                    print(f"Filled missing values in '{col}' with mode: {fill_value}")
    
    print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def validate_dataset(df, required_columns=None, unique_constraints=None):
    """
    Validate dataset structure and constraints.
    """
    validation_results = {}
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_cols
    
    if unique_constraints:
        for constraint in unique_constraints:
            if constraint in df.columns:
                duplicates = df[constraint].duplicated().sum()
                validation_results[f'{constraint}_duplicates'] = duplicates
    
    validation_results['total_rows'] = df.shape[0]
    validation_results['total_columns'] = df.shape[1]
    validation_results['missing_values'] = df.isnull().sum().sum()
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5],
        'value': [10.5, 20.3, 20.3, np.nan, 40.1, 50.0],
        'category': ['A', 'B', 'B', 'C', np.nan, 'A']
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    cleaned_df = clean_dataset(df, fill_strategy='mean')
    print("\nCleaned DataFrame:")
    print(cleaned_df)
    
    validation = validate_dataset(cleaned_df, 
                                 required_columns=['id', 'value', 'category'],
                                 unique_constraints=['id'])
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"{key}: {value}")