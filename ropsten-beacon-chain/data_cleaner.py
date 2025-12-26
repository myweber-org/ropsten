import pandas as pd
import numpy as np

def clean_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
        columns (list): List of columns to clean, if None clean all columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    if columns is None:
        columns = df.columns
    
    df_clean = df.copy()
    
    for col in columns:
        if df[col].isnull().any():
            if strategy == 'mean':
                df_clean[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df_clean[col].fillna(df[col].median(), inplace=True)
            elif strategy == 'mode':
                df_clean[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
    
    return df_clean

def remove_outliers_iqr(df, columns=None, threshold=1.5):
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers
        threshold (float): IQR multiplier threshold
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            df_clean = df_clean[mask]
    
    return df_clean.reset_index(drop=True)

def standardize_columns(df, columns=None):
    """
    Standardize numerical columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    from sklearn.preprocessing import StandardScaler
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_standardized = df.copy()
    scaler = StandardScaler()
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df_standardized[col] = scaler.fit_transform(df_standardized[[col]])
    
    return df_standardized

def clean_dataset(df, missing_strategy='mean', outlier_threshold=1.5, standardize=False):
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        missing_strategy (str): Strategy for handling missing values
        outlier_threshold (float): Threshold for IQR outlier detection
        standardize (bool): Whether to standardize numerical columns
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print(f"Original shape: {df.shape}")
    
    # Handle missing values
    df_clean = clean_missing_values(df, strategy=missing_strategy)
    print(f"After missing value handling: {df_clean.shape}")
    
    # Remove outliers
    df_clean = remove_outliers_iqr(df_clean, threshold=outlier_threshold)
    print(f"After outlier removal: {df_clean.shape}")
    
    # Standardize if requested
    if standardize:
        df_clean = standardize_columns(df_clean)
        print("Columns standardized")
    
    return df_clean

# Example usage
if __name__ == "__main__":
    # Create sample data with missing values and outliers
    np.random.seed(42)
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100) * 2 + 5,
        'feature3': np.random.randn(100) * 0.5 + 10
    }
    
    # Add some missing values
    for col in data:
        mask = np.random.rand(100) < 0.1
        data[col][mask] = np.nan
    
    # Add some outliers
    data['feature1'][0] = 100
    data['feature2'][1] = -50
    
    df = pd.DataFrame(data)
    
    # Clean the dataset
    cleaned_df = clean_dataset(
        df, 
        missing_strategy='median',
        outlier_threshold=1.5,
        standardize=True
    )
    
    print(f"Final cleaned shape: {cleaned_df.shape}")
    print("Data cleaning completed successfully.")