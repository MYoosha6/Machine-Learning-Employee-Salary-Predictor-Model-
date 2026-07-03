import pandas as pd
import numpy as np
from pathlib import Path

class DataInspector:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Loads the dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {self.data_path}: {e}")

    def generate_summary(self) -> str:
        """Generates a comprehensive summary report of the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        report_sections = []
        
        # 1. First 5 rows
        report_sections.append("=== First 5 Rows ===")
        report_sections.append(self.df.head().to_string())
        
        # 2. Dataset shape
        report_sections.append("\n=== Dataset Shape ===")
        report_sections.append(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        # 3. Column names
        report_sections.append("\n=== Column Names ===")
        report_sections.append(", ".join(self.df.columns.tolist()))
        
        # 4. Data types
        report_sections.append("\n=== Data Types ===")
        report_sections.append(self.df.dtypes.to_string())
        
        # 5. Missing values count
        report_sections.append("\n=== Missing Values Count ===")
        report_sections.append(self.df.isnull().sum().to_string())
        
        # 6. Duplicate rows count
        report_sections.append("\n=== Duplicate Rows Count ===")
        report_sections.append(str(self.df.duplicated().sum()))
        
        # 7. Summary statistics for numerical columns
        report_sections.append("\n=== Summary Statistics (Numerical) ===")
        # select_dtypes includes np.number to get only numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number])
        if not numeric_cols.empty:
            report_sections.append(numeric_cols.describe().to_string())
        else:
            report_sections.append("No numerical columns found.")
            
        return "\n".join(report_sections)
