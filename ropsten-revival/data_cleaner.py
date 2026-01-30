
import pandas as pd

def clean_dataframe(df, column_name, condition_func):
    """
    Filters a pandas DataFrame based on a condition applied to a specific column.
    Removes rows where the condition function returns False.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        column_name (str): The name of the column to apply the condition to.
        condition_func (function): A function that takes a single value from the
                                   specified column and returns a boolean.

    Returns:
        pd.DataFrame: A new DataFrame with rows filtered based on the condition.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    filtered_df = df[df[column_name].apply(condition_func)].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def remove_outliers_iqr(df, column_name):
    """
    Removes outliers from a specified column in a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the numeric column to process.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed from the specified column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df