import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self, data_path: Path, preprocessor_path: Path, output_dir: Path, models_dir: Path):
        self.data_path = data_path
        self.preprocessor_path = preprocessor_path
        self.output_dir = output_dir
        self.models_dir = models_dir

    def train_and_evaluate(self):
        print("Loading data and preprocessor...")
        df = pd.read_csv(self.data_path)
        preprocessor = joblib.load(self.preprocessor_path)

        # Separate features and target
        X = df.drop(columns=['salary'])
        y = df['salary']

        # Train test split (20% test size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
            # Setting n_estimators to 50 and using n_jobs=-1 to maintain decent performance and speed
            'Random Forest Regressor': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }

        results = []
        best_model = None
        best_r2 = -float('inf')
        best_model_name = ""
        best_pipeline = None

        for name, model in models.items():
            print(f"Training {name}...")
            # Create a full pipeline with preprocessor and model
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Train the pipeline
            pipeline.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            results.append({
                'Model': name,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2 Score': r2
            })

            # Check if best model based on R2 score
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_pipeline = pipeline

        # Create comparison table
        results_df = pd.DataFrame(results)
        
        # Save evaluation metrics
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.output_dir / 'model_comparison.csv', index=False)

        # Save best model
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, self.models_dir / 'best_model.pkl')

        summary = []
        summary.append("=== Model Comparison ===")
        summary.append(results_df.to_string(index=False))
        summary.append(f"\nBest Performing Model: {best_model_name}")
        summary.append(f"Best Model R2 Score: {best_r2:.4f}")
        summary.append(f"Saved best model to: {self.models_dir / 'best_model.pkl'}")
        summary.append(f"Saved model comparison to: {self.output_dir / 'model_comparison.csv'}")

        return "\n".join(summary)
