
import pandas as pd
import numpy as np
from pathlib import Path

def clean_csv_data(input_path, output_path=None, missing_strategy='mean'):
    """
    Clean CSV data by handling missing values and removing duplicates.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str, optional): Path for cleaned output CSV. If None, adds '_cleaned' suffix
    missing_strategy (str): Strategy for handling missing values: 'mean', 'median', 'drop', or 'zero'
    
    Returns:
    pandas.DataFrame: Cleaned dataframe
    """
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    original_rows = len(df)
    print(f"Original data shape: {df.shape}")
    
    df = df.drop_duplicates()
    print(f"Removed {original_rows - len(df)} duplicate rows")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == 'mean':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
    elif missing_strategy == 'median':
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
    elif missing_strategy == 'zero':
        df.fillna(0, inplace=True)
    elif missing_strategy == 'drop':
        df.dropna(inplace=True)
    else:
        raise ValueError("Invalid missing_strategy. Use 'mean', 'median', 'drop', or 'zero'")
    
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = input_path_obj.parent / f"{input_path_obj.stem}_cleaned.csv"
    
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    print(f"Final data shape: {df.shape}")
    
    return df

def validate_dataframe(df, required_columns=None):
    """
    Validate dataframe structure and content.
    
    Parameters:
    df (pandas.DataFrame): Dataframe to validate
    required_columns (list, optional): List of required column names
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary': {}
    }
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
    
    validation_results['summary']['total_rows'] = len(df)
    validation_results['summary']['total_columns'] = len(df.columns)
    validation_results['summary']['missing_values'] = df.isnull().sum().sum()
    validation_results['summary']['duplicate_rows'] = df.duplicated().sum()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        validation_results['summary']['numeric_stats'] = {}
        for col in numeric_cols:
            validation_results['summary']['numeric_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    return validation_results

if __name__ == "__main__":
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6],
        'value': [10.5, 20.3, None, 15.7, 20.3, None],
        'category': ['A', 'B', 'A', 'C', 'B', 'A']
    }
    
    test_df = pd.DataFrame(sample_data)
    test_df.to_csv('test_data.csv', index=False)
    
    cleaned_df = clean_csv_data('test_data.csv', missing_strategy='mean')
    
    validation = validate_dataframe(cleaned_df, required_columns=['id', 'value', 'category'])
    print("\nValidation Results:")
    print(validation)