import pandas as pd
import numpy as np

def clean_missing_data(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        strategy (str): Strategy for handling missing values.
                       Options: 'mean', 'median', 'mode', 'drop', 'fill_zero'
        columns (list): List of columns to apply cleaning to. If None, applies to all numeric columns.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        columns = list(numeric_cols)
    
    if strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'median':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mode':
        for col in columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0)
    elif strategy == 'fill_zero':
        df_clean[columns] = df_clean[columns].fillna(0)
    
    return df_clean

def remove_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to check for outliers
        multiplier (float): IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def standardize_columns(df, columns=None):
    """
    Standardize numeric columns to have zero mean and unit variance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of columns to standardize
    
    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    from sklearn.preprocessing import StandardScaler
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    scaler = StandardScaler()
    
    df_clean[columns] = scaler.fit_transform(df_clean[columns])
    
    return df_clean, scaler

def save_cleaned_data(df, output_path):
    """
    Save cleaned DataFrame to CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        output_path (str): Path to save the CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")import pandas as pd
import numpy as np

def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from DataFrame.
    """
    return df.drop_duplicates(subset=subset, keep='first')

def convert_column_types(df, column_types):
    """
    Convert DataFrame columns to specified types.
    """
    for column, dtype in column_types.items():
        if column in df.columns:
            df[column] = df[column].astype(dtype)
    return df

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in DataFrame columns.
    """
    if columns is None:
        columns = df.columns
    
    for column in columns:
        if column in df.columns:
            if strategy == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif strategy == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif strategy == 'mode':
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif strategy == 'drop':
                df.dropna(subset=[column], inplace=True)
    
    return df

def normalize_column(df, column):
    """
    Normalize a column to range [0, 1].
    """
    if column in df.columns:
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max != col_min:
            df[column] = (df[column] - col_min) / (col_max - col_min)
    return df

def clean_dataframe(df, config):
    """
    Apply multiple cleaning operations based on configuration.
    """
    if config.get('remove_duplicates'):
        df = remove_duplicates(df, config.get('duplicate_subset'))
    
    if config.get('column_types'):
        df = convert_column_types(df, config['column_types'])
    
    if config.get('missing_values'):
        missing_config = config['missing_values']
        df = handle_missing_values(
            df, 
            strategy=missing_config.get('strategy', 'mean'),
            columns=missing_config.get('columns')
        )
    
    if config.get('normalize_columns'):
        for column in config['normalize_columns']:
            df = normalize_column(df, column)
    
    return df

def validate_dataframe(df, rules):
    """
    Validate DataFrame against specified rules.
    """
    violations = []
    
    for rule in rules:
        column = rule.get('column')
        rule_type = rule.get('type')
        
        if column not in df.columns:
            violations.append(f"Column '{column}' not found")
            continue
        
        if rule_type == 'not_null':
            null_count = df[column].isnull().sum()
            if null_count > 0:
                violations.append(f"Column '{column}' has {null_count} null values")
        
        elif rule_type == 'unique':
            duplicate_count = df[column].duplicated().sum()
            if duplicate_count > 0:
                violations.append(f"Column '{column}' has {duplicate_count} duplicate values")
        
        elif rule_type == 'range':
            min_val = rule.get('min')
            max_val = rule.get('max')
            out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
            if len(out_of_range) > 0:
                violations.append(f"Column '{column}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]")
    
    return violations