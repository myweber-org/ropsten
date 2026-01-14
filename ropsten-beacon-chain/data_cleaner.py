import pandas as pd
import numpy as np

def clean_csv_data(input_file, output_file):
    """
    Load a CSV file, clean missing values, and save cleaned data.
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        print("Missing values per column:")
        print(missing_counts[missing_counts > 0])
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"Filled missing values in {col} with median: {median_val}")
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_rows - len(df)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Save cleaned data
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        print(f"Final data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return None
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    input_csv = "raw_data.csv"
    output_csv = "cleaned_data.csv"
    cleaned_df = clean_csv_data(input_csv, output_csv)
import numpy as np
import pandas as pd

def remove_outliers_iqr(df, column):
    """
    Remove outliers from a DataFrame column using the Interquartile Range method.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to clean
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df.reset_index(drop=True)

def calculate_summary_statistics(df, column):
    """
    Calculate summary statistics for a column after outlier removal.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    column (str): Column name to analyze
    
    Returns:
    dict: Dictionary containing summary statistics
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'count': df[column].count(),
        'q1': df[column].quantile(0.25),
        'q3': df[column].quantile(0.75)
    }
    
    return stats

def clean_dataset(df, columns_to_clean):
    """
    Clean multiple columns in a DataFrame by removing outliers.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_clean (list): List of column names to clean
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    dict: Dictionary of summary statistics for each cleaned column
    """
    cleaned_df = df.copy()
    summary_stats = {}
    
    for column in columns_to_clean:
        if column in cleaned_df.columns:
            original_count = len(cleaned_df)
            cleaned_df = remove_outliers_iqr(cleaned_df, column)
            removed_count = original_count - len(cleaned_df)
            
            stats = calculate_summary_statistics(cleaned_df, column)
            stats['outliers_removed'] = removed_count
            summary_stats[column] = stats
    
    return cleaned_df, summary_stats

if __name__ == "__main__":
    sample_data = {
        'temperature': [22, 23, 24, 25, 26, 100, 27, 28, 29, -10, 30, 31, 32],
        'humidity': [45, 46, 47, 48, 49, 200, 50, 51, 52, -5, 53, 54, 55],
        'pressure': [1013, 1014, 1015, 1016, 1017, 2000, 1018, 1019, 1020, 500, 1021, 1022, 1023]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original DataFrame:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    columns_to_clean = ['temperature', 'humidity', 'pressure']
    cleaned_df, stats = clean_dataset(df, columns_to_clean)
    
    print("Cleaned DataFrame:")
    print(cleaned_df)
    print("\n" + "="*50 + "\n")
    
    print("Summary Statistics:")
    for column, column_stats in stats.items():
        print(f"\n{column}:")
        for stat_name, stat_value in column_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}" if isinstance(stat_value, float) else f"  {stat_name}: {stat_value}")
import pandas as pd
import numpy as np

def clean_dataset(df, columns_to_check=None, fill_strategy='mean'):
    """
    Clean a pandas DataFrame by removing duplicates and handling missing values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame to clean
    columns_to_check (list): List of columns to check for duplicates, 
                             if None uses all columns
    fill_strategy (str): Strategy for filling missing values: 
                         'mean', 'median', 'mode', or 'drop'
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    # Create a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Remove duplicates
    if columns_to_check is None:
        columns_to_check = cleaned_df.columns.tolist()
    
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=columns_to_check, keep='first')
    duplicates_removed = initial_rows - len(cleaned_df)
    
    # Handle missing values
    missing_before = cleaned_df.isnull().sum().sum()
    
    if fill_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        for column in cleaned_df.columns:
            if cleaned_df[column].dtype in ['int64', 'float64']:
                if fill_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif fill_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif fill_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0] if not cleaned_df[column].mode().empty else 0
                else:
                    fill_value = 0
                
                cleaned_df[column] = cleaned_df[column].fillna(fill_value)
            else:
                # For non-numeric columns, fill with the most frequent value
                if not cleaned_df[column].mode().empty:
                    cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode()[0])
                else:
                    cleaned_df[column] = cleaned_df[column].fillna('Unknown')
    
    missing_after = cleaned_df.isnull().sum().sum()
    missing_filled = missing_before - missing_after
    
    # Print cleaning summary
    print(f"Data cleaning completed:")
    print(f"  - Duplicates removed: {duplicates_removed}")
    print(f"  - Missing values filled: {missing_filled}")
    print(f"  - Remaining missing values: {missing_after}")
    print(f"  - Final dataset shape: {cleaned_df.shape}")
    
    return cleaned_df

def validate_data(df, required_columns=None, numeric_ranges=None):
    """
    Validate data quality after cleaning.
    
    Parameters:
    df (pd.DataFrame): DataFrame to validate
    required_columns (list): List of columns that must be present
    numeric_ranges (dict): Dictionary with column names as keys and 
                          (min, max) tuples as values for numeric validation
    
    Returns:
    dict: Validation results
    """
    validation_results = {
        'has_required_columns': True,
        'numeric_ranges_valid': True,
        'no_missing_values': True,
        'validation_errors': []
    }
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['has_required_columns'] = False
            validation_results['validation_errors'].append(
                f"Missing required columns: {missing_columns}"
            )
    
    # Check numeric ranges
    if numeric_ranges:
        for column, (min_val, max_val) in numeric_ranges.items():
            if column in df.columns and df[column].dtype in ['int64', 'float64']:
                out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
                if len(out_of_range) > 0:
                    validation_results['numeric_ranges_valid'] = False
                    validation_results['validation_errors'].append(
                        f"Column '{column}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]"
                    )
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        validation_results['no_missing_values'] = False
        validation_results['validation_errors'].append(
            f"Dataset contains {missing_values} missing values"
        )
    
    return validation_results

# Example usage (commented out for production)
if __name__ == "__main__":
    # Create sample data for demonstration
    sample_data = {
        'id': [1, 2, 2, 3, 4, 5, 5],
        'name': ['Alice', 'Bob', 'Bob', 'Charlie', None, 'Eve', 'Eve'],
        'age': [25, 30, 30, None, 35, 40, 40],
        'score': [85.5, 92.0, 92.0, 78.5, 88.0, None, None]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original dataset:")
    print(df)
    print("\n" + "="*50 + "\n")
    
    # Clean the dataset
    cleaned_df = clean_dataset(df, fill_strategy='mean')
    
    print("\n" + "="*50 + "\n")
    print("Cleaned dataset:")
    print(cleaned_df)
    
    # Validate the cleaned data
    validation = validate_data(
        cleaned_df,
        required_columns=['id', 'name', 'age', 'score'],
        numeric_ranges={'age': (0, 150), 'score': (0, 100)}
    )
    
    print("\nValidation results:")
    for key, value in validation.items():
        if key != 'validation_errors':
            print(f"  {key}: {value}")
    
    if validation['validation_errors']:
        print("\nValidation errors:")
        for error in validation['validation_errors']:
            print(f"  - {error}")