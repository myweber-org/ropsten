import pandas as pd
import numpy as np

def clean_dataframe(df, missing_strategy='mean', outlier_threshold=3):
    """
    Clean a pandas DataFrame by handling missing values and outliers.

    Parameters:
    df (pd.DataFrame): Input DataFrame to clean.
    missing_strategy (str): Strategy for handling missing values.
        Options: 'mean', 'median', 'mode', 'drop'.
    outlier_threshold (float): Number of standard deviations to consider a point an outlier.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()

    # Handle missing values
    if missing_strategy == 'drop':
        cleaned_df = cleaned_df.dropna()
    else:
        for column in cleaned_df.select_dtypes(include=[np.number]).columns:
            if cleaned_df[column].isnull().any():
                if missing_strategy == 'mean':
                    fill_value = cleaned_df[column].mean()
                elif missing_strategy == 'median':
                    fill_value = cleaned_df[column].median()
                elif missing_strategy == 'mode':
                    fill_value = cleaned_df[column].mode()[0]
                else:
                    raise ValueError(f"Unsupported missing_strategy: {missing_strategy}")
                cleaned_df[column].fillna(fill_value, inplace=True)

    # Handle outliers for numeric columns
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        mean = cleaned_df[column].mean()
        std = cleaned_df[column].std()
        if std > 0:  # Avoid division by zero
            z_scores = np.abs((cleaned_df[column] - mean) / std)
            cleaned_df = cleaned_df[z_scores < outlier_threshold]

    return cleaned_df.reset_index(drop=True)

def remove_duplicates(df, subset=None, keep='first'):
    """
    Remove duplicate rows from DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    subset (list): Columns to consider for identifying duplicates.
    keep (str): Which duplicates to keep.

    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

def normalize_columns(df, columns=None, method='minmax'):
    """
    Normalize specified columns in DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns (list): Columns to normalize. If None, normalize all numeric columns.
    method (str): Normalization method. Options: 'minmax', 'zscore'.

    Returns:
    pd.DataFrame: DataFrame with normalized columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    normalized_df = df.copy()
    for column in columns:
        if column in normalized_df.columns and pd.api.types.is_numeric_dtype(normalized_df[column]):
            if method == 'minmax':
                min_val = normalized_df[column].min()
                max_val = normalized_df[column].max()
                if max_val > min_val:
                    normalized_df[column] = (normalized_df[column] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean = normalized_df[column].mean()
                std = normalized_df[column].std()
                if std > 0:
                    normalized_df[column] = (normalized_df[column] - mean) / std
            else:
                raise ValueError(f"Unsupported normalization method: {method}")

    return normalized_dfimport pandas as pd
import numpy as np

def clean_csv_data(file_path, output_path=None):
    """
    Load a CSV file, perform basic cleaning operations,
    and optionally save the cleaned data.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None

    original_shape = df.shape
    print(f"Original data shape: {original_shape}")

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)

    # Remove columns with more than 50% missing values
    threshold = len(df) * 0.5
    cols_to_drop = [col for col in df.columns if df[col].isnull().sum() > threshold]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped columns with >50% missing values: {cols_to_drop}")

    cleaned_shape = df.shape
    print(f"Cleaned data shape: {cleaned_shape}")
    print(f"Rows removed: {original_shape[0] - cleaned_shape[0]}")
    print(f"Columns removed: {original_shape[1] - cleaned_shape[1]}")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")

    return df

if __name__ == "__main__":
    # Example usage
    input_file = "raw_data.csv"
    output_file = "cleaned_data.csv"
    
    cleaned_df = clean_csv_data(input_file, output_file)
    if cleaned_df is not None:
        print("Data cleaning completed successfully.")
        print(cleaned_df.head())