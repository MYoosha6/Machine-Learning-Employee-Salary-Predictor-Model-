import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class ExploratoryDataAnalyzer:
    def __init__(self, data_path: Path, output_dir: Path):
        self.data_path = data_path
        self.output_dir = output_dir
        self.plots_dir = self.output_dir / 'plots'
        self.summary_file = self.output_dir / 'eda_summary.txt'
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)

    def run_eda(self):
        if self.df is None:
            self.load_data()

        # Ensure plots directory exists
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        sns.set_theme(style="whitegrid")
        findings = []
        findings.append("=== Exploratory Data Analysis Summary ===")

        # 1. Salary distribution histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['salary'], bins=50, kde=True, color='blue')
        plt.title('Salary Distribution')
        plt.xlabel('Salary')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(self.plots_dir / '1_salary_distribution.png')
        plt.close()
        findings.append(f"- Salary distribution is plotted. Median salary is ~{self.df['salary'].median():.2f}, ranging from {self.df['salary'].min()} to {self.df['salary'].max()}.")

        # 2. Salary vs years of experience scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='experience_years', y='salary', data=self.df, alpha=0.3, color='green')
        plt.title('Salary vs Years of Experience')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.tight_layout()
        plt.savefig(self.plots_dir / '2_salary_vs_experience.png')
        plt.close()
        exp_corr = self.df['salary'].corr(self.df['experience_years'])
        findings.append(f"- Correlation between Salary and Years of Experience is {exp_corr:.2f}.")

        # 3. Salary by education level boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='education_level', y='salary', data=self.df, palette='Set2')
        plt.title('Salary by Education Level')
        plt.xlabel('Education Level')
        plt.ylabel('Salary')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / '3_salary_by_education.png')
        plt.close()
        findings.append("- Plot shows salary distribution variations across different education levels.")

        # 4. Salary by industry boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='industry', y='salary', data=self.df, palette='Set3')
        plt.title('Salary by Industry')
        plt.xlabel('Industry')
        plt.ylabel('Salary')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / '4_salary_by_industry.png')
        plt.close()
        findings.append("- Plot highlights which industries tend to pay more on average.")

        # 5. Correlation heatmap for numerical features
        plt.figure(figsize=(8, 6))
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(self.plots_dir / '5_correlation_heatmap.png')
        plt.close()
        
        highest_corr_val = None
        highest_corr_pair = None
        # get upper triangle without diagonal
        corr_triu = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        if not corr_triu.empty and not corr_triu.isna().all().all():
            unstacked = corr_triu.unstack().dropna()
            if not unstacked.empty:
                max_idx = unstacked.abs().idxmax()
                highest_corr_val = unstacked[max_idx]
                highest_corr_pair = max_idx
                findings.append(f"- Highest correlation between distinct numerical features is {highest_corr_val:.2f} ({highest_corr_pair[0]} & {highest_corr_pair[1]}).")

        # 6. Average salary by company size
        plt.figure(figsize=(10, 6))
        comp_salary = self.df.groupby('company_size')['salary'].mean().sort_values(ascending=False)
        sns.barplot(x=comp_salary.index, y=comp_salary.values, palette='viridis')
        plt.title('Average Salary by Company Size')
        plt.xlabel('Company Size')
        plt.ylabel('Average Salary')
        plt.tight_layout()
        plt.savefig(self.plots_dir / '6_average_salary_by_company_size.png')
        plt.close()
        findings.append(f"- Average salary by company size:\n{comp_salary.to_string()}")

        # 7. Average salary by remote work status
        plt.figure(figsize=(8, 6))
        remote_salary = self.df.groupby('remote_work')['salary'].mean().sort_values(ascending=False)
        sns.barplot(x=remote_salary.index, y=remote_salary.values, palette='magma')
        plt.title('Average Salary by Remote Work Status')
        plt.xlabel('Remote Work Status')
        plt.ylabel('Average Salary')
        plt.tight_layout()
        plt.savefig(self.plots_dir / '7_average_salary_by_remote_work.png')
        plt.close()
        findings.append(f"- Average salary by remote work status:\n{remote_salary.to_string()}")

        # Save findings
        findings_text = "\n\n".join(findings)
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(findings_text)

        summary_return = f"EDA completed successfully. \nGenerated 7 plots in '{self.plots_dir}'. \nSummary saved in '{self.summary_file}'."
        return summary_return
