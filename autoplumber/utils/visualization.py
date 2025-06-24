import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Optional

class Visualizer:
    """Visualization utilities for AutoPlumber results and data analysis."""
    
    def __init__(self, style='seaborn-v0_8', figsize=(10, 6)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 8)
    
    def plot_search_history(self, search_history: List[Dict], title="AutoPlumber Search History"):
        """Plot the search history showing score improvements over iterations."""
        if not search_history:
            print("No search history to plot.")
            return
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        iterations = [h['iteration'] for h in search_history]
        scores = [h['score'] for h in search_history]
        improved = [h['improved'] for h in search_history]
        
        # Plot all scores
        ax.plot(iterations, scores, 'o-', color=self.colors[0], alpha=0.7, label='Score')
        
        # Highlight improvements
        improved_iterations = [i for i, imp in zip(iterations, improved) if imp]
        improved_scores = [s for s, imp in zip(scores, improved) if imp]
        
        if improved_iterations:
            ax.scatter(improved_iterations, improved_scores, 
                      color=self.colors[1], s=100, alpha=0.8, 
                      label='Improvements', zorder=5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_column_distributions(self, data: pd.DataFrame, columns: Optional[List[str]] = None):
        """Plot distributions of columns in the dataset."""
        if columns is None:
            columns = data.columns[:6]  # Limit to first 6 columns
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            ax = axes[i] if len(columns) > 1 else axes[0]
            
            if pd.api.types.is_numeric_dtype(data[col]):
                # Numeric column - histogram
                data[col].hist(bins=30, alpha=0.7, ax=ax, color=self.colors[i % len(self.colors)])
                ax.set_title(f'{col} (Numeric)')
            else:
                # Categorical column - bar plot
                value_counts = data[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax, color=self.colors[i % len(self.colors)])
                ax.set_title(f'{col} (Categorical)')
                ax.tick_params(axis='x', rotation=45)
            
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_missing_values(self, data: pd.DataFrame):
        """Plot missing value patterns in the dataset."""
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if missing_data.empty:
            print("No missing values in the dataset.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Missing values count
        missing_data.plot(kind='bar', ax=ax1, color=self.colors[0])
        ax1.set_title('Missing Values Count')
        ax1.set_xlabel('Columns')
        ax1.set_ylabel('Missing Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Missing values percentage
        missing_percentage = (missing_data / len(data)) * 100
        missing_percentage.plot(kind='bar', ax=ax2, color=self.colors[1])
        ax2.set_title('Missing Values Percentage')
        ax2.set_xlabel('Columns')
        ax2.set_ylabel('Missing Percentage (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_pipeline_summary(self, pipeline_summary: Dict[str, Any]):
        """Plot a summary of the best pipeline found."""
        if 'column_pipelines' not in pipeline_summary:
            print("No pipeline summary to plot.")
            return
        
        column_pipelines = pipeline_summary['column_pipelines']
        
        if not column_pipelines:
            print("No column pipelines to visualize.")
            return
        
        # Count transformer types
        transformer_counts = {}
        for col, transformers in column_pipelines.items():
            for transformer in transformers:
                t_type = transformer['type']
                transformer_counts[t_type] = transformer_counts.get(t_type, 0) + 1
        
        # Plot transformer usage
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Transformer type counts
        if transformer_counts:
            types = list(transformer_counts.keys())
            counts = list(transformer_counts.values())
            
            ax1.bar(types, counts, color=self.colors[:len(types)])
            ax1.set_title('Transformer Usage Count')
            ax1.set_xlabel('Transformer Type')
            ax1.set_ylabel('Usage Count')
            ax1.tick_params(axis='x', rotation=45)
        
        # Pipeline complexity (number of transformers per column)
        complexities = []
        column_names = []
        for col, transformers in column_pipelines.items():
            complexities.append(len(transformers))
            column_names.append(col)
        
        ax2.bar(column_names, complexities, color=self.colors[2])
        ax2.set_title('Pipeline Complexity per Column')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Number of Transformers')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add best score as text
        if 'best_score' in pipeline_summary:
            fig.suptitle(f'Best Score: {pipeline_summary["best_score"]:.4f}', fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    def plot_before_after_comparison(self, original_data: pd.DataFrame, 
                                   transformed_data: pd.DataFrame, 
                                   column: str):
        """Plot before and after comparison for a specific column."""
        if column not in original_data.columns:
            print(f"Column '{column}' not found in original data.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original data
        if pd.api.types.is_numeric_dtype(original_data[column]):
            original_data[column].hist(bins=30, alpha=0.7, ax=ax1, color=self.colors[0])
        else:
            original_data[column].value_counts().head(10).plot(kind='bar', ax=ax1, color=self.colors[0])
        
        ax1.set_title(f'Original: {column}')
        ax1.set_xlabel(column)
        ax1.set_ylabel('Frequency')
        
        # Transformed data (check if column exists after transformation)
        transformed_columns = [col for col in transformed_data.columns if column in col]
        
        if transformed_columns:
            # Use the first matching column
            trans_col = transformed_columns[0]
            
            if pd.api.types.is_numeric_dtype(transformed_data[trans_col]):
                transformed_data[trans_col].hist(bins=30, alpha=0.7, ax=ax2, color=self.colors[1])
            else:
                transformed_data[trans_col].value_counts().head(10).plot(kind='bar', ax=ax2, color=self.colors[1])
            
            ax2.set_title(f'Transformed: {trans_col}')
            ax2.set_xlabel(trans_col)
            ax2.set_ylabel('Frequency')
        else:
            ax2.text(0.5, 0.5, f'Column {column}\\nremoved or transformed\\ninto multiple columns', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title(f'Transformed: {column}')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_analysis: Dict[str, Any], title="Feature Importance Analysis"):
        """Plot feature importance from AutoPlumber analysis."""
        if 'feature_contributions' not in importance_analysis:
            print("No feature contributions found in analysis.")
            return
        
        contributions = importance_analysis['feature_contributions']
        if not contributions:
            print("No feature contributions to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature contributions bar plot
        features = list(contributions.keys())
        scores = list(contributions.values())
        
        bars = ax1.bar(features, scores, color=self.colors[0], alpha=0.7)
        ax1.set_title('Individual Feature Contributions')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('CV Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Preprocessing complexity plot
        if 'preprocessing_impact' in importance_analysis:
            preprocessing_data = importance_analysis['preprocessing_impact']
            complexity = [data['num_transformers'] for data in preprocessing_data.values()]
            
            ax2.scatter(scores, complexity, c=self.colors[1], alpha=0.7, s=100)
            for i, feature in enumerate(features):
                ax2.annotate(feature, (scores[i], complexity[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax2.set_xlabel('CV Score')
            ax2.set_ylabel('Number of Preprocessing Steps')
            ax2.set_title('Score vs. Preprocessing Complexity')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_improvement_rate(self, improvement_analysis: Dict[str, Any], title="Improvement Rate Analysis"):
        """Plot improvement rate analysis over search iterations."""
        if 'improvement_trend' not in improvement_analysis:
            print("No improvement trend found in analysis.")
            return
        
        improvements = improvement_analysis['improvement_trend']
        if not improvements:
            print("No improvement data to plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Improvement per iteration
        iterations = range(2, len(improvements) + 2)  # Start from iteration 2
        ax1.bar(iterations, improvements, alpha=0.7, color=self.colors[2])
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Score Improvement per Iteration')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Score Improvement')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative improvement
        cumulative = np.cumsum(improvements)
        ax2.plot(iterations, cumulative, 'o-', color=self.colors[3], linewidth=2)
        ax2.set_title('Cumulative Score Improvement')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Cumulative Improvement')
        ax2.grid(True, alpha=0.3)
        
        # Add summary statistics
        stats_text = f"""
        Total Improvement: {improvement_analysis.get('total_improvement', 0):.4f}
        Avg per Iteration: {improvement_analysis.get('avg_improvement_per_iteration', 0):.4f}
        Iterations with Improvement: {improvement_analysis.get('iterations_with_improvement', 0)}
        Max Single Improvement: {improvement_analysis.get('max_single_improvement', 0):.4f}
        """
        ax2.text(0.02, 0.98, stats_text.strip(), transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()