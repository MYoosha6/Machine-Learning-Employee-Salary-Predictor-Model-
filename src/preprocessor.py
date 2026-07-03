import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessor:
    def __init__(self, data_path: Path, output_dir: Path, models_dir: Path):
        self.data_path = data_path
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.df = None
        self.preprocessor = None

    def load_data(self):
        """Loads the dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {self.data_path}: {e}")

    def preprocess(self):
        """Builds and applies the preprocessing pipeline."""
        if self.df is None:
            self.load_data()

        # 1. Remove duplicate rows
        initial_shape = self.df.shape
        self.df.drop_duplicates(inplace=True)
        dropped_duplicates = initial_shape[0] - self.df.shape[0]

        # 3. Separate features and target variable
        # 4. Target variable: salary
        X = self.df.drop(columns=['salary'])
        y = self.df['salary']

        # Identify numerical and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # 2. Handle missing values & 5. Encode & 6. Scale
        # 7. Build preprocessing pipeline using ColumnTransformer and Pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Fit the preprocessor
        self.preprocessor.fit(X)

        # 8. Save preprocessing object
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, self.models_dir / 'preprocessor.pkl')

        # 9. Save cleaned dataset
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.output_dir / 'cleaned_dataset.csv', index=False)

        # 10. Print preprocessing summary
        summary = []
        summary.append("=== Preprocessing Summary ===")
        summary.append(f"Dropped duplicate rows: {dropped_duplicates}")
        summary.append(f"Target variable separated: 'salary'")
        summary.append(f"Numerical features ({len(numeric_features)}): {', '.join(numeric_features)}")
        summary.append("  - Imputation: median")
        summary.append("  - Scaling: StandardScaler")
        summary.append(f"Categorical features ({len(categorical_features)}): {', '.join(categorical_features)}")
        summary.append("  - Imputation: most_frequent")
        summary.append("  - Encoding: OneHotEncoder")
        summary.append(f"\nArtifacts Saved:")
        summary.append(f"- Preprocessor object: {self.models_dir / 'preprocessor.pkl'}")
        summary.append(f"- Cleaned dataset: {self.output_dir / 'cleaned_dataset.csv'}")

        return "\n".join(summary)
