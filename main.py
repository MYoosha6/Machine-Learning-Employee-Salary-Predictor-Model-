import sys
from pathlib import Path
from src.data_inspector import DataInspector
from src.preprocessor import DataPreprocessor
from src.eda import ExploratoryDataAnalyzer
from src.model_trainer import ModelTrainer

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "data" / "job_salary_prediction_dataset.csv"
    output_dir = base_dir / "outputs"
    models_dir = base_dir / "models"
    output_file = output_dir / "dataset_summary.txt"
    
    print(f"Initializing project and inspecting data...\n")
    print(f"Looking for dataset at: {data_file}")
    
    if not data_file.exists():
        print(f"Error: Dataset not found at {data_file}")
        sys.exit(1)
        
    try:
        # Initialize and run inspector
        inspector = DataInspector(data_file)
        inspector.load_data()
        
        # Generate report
        report = inspector.generate_summary()
        
        # Print clean report to terminal
        print("\n" + "="*50)
        print("DATASET INSPECTION REPORT")
        print("="*50 + "\n")
        print(report)
        print("\n" + "="*50)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DATASET INSPECTION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(report)
            
        print(f"\nSuccess! Report successfully saved to {output_file}")
        
        # Initialize and run EDA
        print("\n" + "="*50)
        print("RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*50)
        eda = ExploratoryDataAnalyzer(data_file, output_dir)
        eda_summary = eda.run_eda()
        print("\n" + eda_summary)
        print("\n" + "="*50)
        
        # Initialize and run preprocessor
        print("\n" + "="*50)
        print("RUNNING DATA PREPROCESSING")
        print("="*50)
        preprocessor = DataPreprocessor(data_file, output_dir, models_dir)
        preprocessing_summary = preprocessor.preprocess()
        print("\n" + preprocessing_summary)
        print("\n" + "="*50)
        print("Data Preprocessing Completed Successfully!")
        
        # Initialize and run model trainer
        print("\n" + "="*50)
        print("RUNNING MODEL TRAINING AND EVALUATION")
        print("="*50)
        cleaned_data_file = output_dir / "cleaned_dataset.csv"
        preprocessor_file = models_dir / "preprocessor.pkl"
        trainer = ModelTrainer(cleaned_data_file, preprocessor_file, output_dir, models_dir)
        training_summary = trainer.train_and_evaluate()
        print("\n" + training_summary)
        print("\n" + "="*50)
        print("Model Training Completed Successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
