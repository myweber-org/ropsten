
import pandas as pd
import numpy as np
from typing import List, Optional

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        
    def remove_duplicates(self, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.
        
        Args:
            subset: Column labels to consider for identifying duplicates.
            keep: Which duplicates to keep ('first', 'last', or False for none).
            
        Returns:
            Cleaned DataFrame with duplicates removed.
        """
        cleaned_df = self.df.drop_duplicates(subset=subset, keep=keep)
        removed_count = len(self.df) - len(cleaned_df)
        
        print(f"Removed {removed_count} duplicate rows")
        print(f"Original shape: {self.original_shape}")
        print(f"New shape: {cleaned_df.shape}")
        
        return cleaned_df
    
    def remove_outliers_iqr(self, column: str, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using the Interquartile Range method.
        
        Args:
            column: Name of the column to check for outliers.
            multiplier: IQR multiplier for outlier detection.
            
        Returns:
            DataFrame with outliers removed.
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)
        cleaned_df = self.df[mask]
        
        removed_count = len(self.df) - len(cleaned_df)
        print(f"Removed {removed_count} outliers from column '{column}'")
        
        return cleaned_df
    
    def fill_missing_values(self, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fill missing values in specified columns.
        
        Args:
            strategy: Method for filling ('mean', 'median', 'mode', or 'constant').
            columns: List of columns to fill. If None, fills all numeric columns.
            
        Returns:
            DataFrame with missing values filled.
        """
        df_copy = self.df.copy()
        
        if columns is None:
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            columns = list(numeric_cols)
        
        for col in columns:
            if col not in df_copy.columns:
                continue
                
            if strategy == 'mean':
                fill_value = df_copy[col].mean()
            elif strategy == 'median':
                fill_value = df_copy[col].median()
            elif strategy == 'mode':
                fill_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else 0
            elif strategy == 'constant':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            missing_count = df_copy[col].isna().sum()
            df_copy[col] = df_copy[col].fillna(fill_value)
            
            if missing_count > 0:
                print(f"Filled {missing_count} missing values in column '{col}' with {strategy}: {fill_value}")
        
        return df_copy

def clean_dataset(file_path: str, output_path: str) -> None:
    """
    Complete data cleaning pipeline for a CSV file.
    
    Args:
        file_path: Path to input CSV file.
        output_path: Path to save cleaned CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        cleaner = DataCleaner(df)
        
        cleaned_df = cleaner.remove_duplicates()
        cleaned_df = cleaner.fill_missing_values(strategy='median')
        
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            cleaned_df = cleaner.remove_outliers_iqr(col)
        
        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5, 6],
        'value': [10, 20, 30, 40, 50, 50, 1000],
        'category': ['A', 'B', 'A', 'B', 'A', 'A', 'C'],
        'score': [85, 90, None, 88, 92, 92, 95]
    })
    
    cleaner = DataCleaner(sample_data)
    result = cleaner.remove_duplicates(subset=['id'])
    result = cleaner.fill_missing_values(strategy='mean', columns=['score'])
    result = cleaner.remove_outliers_iqr('value')
    
    print("\nFinal cleaned data:")
    print(result)