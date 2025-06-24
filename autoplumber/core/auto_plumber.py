import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, clone
import warnings
from itertools import product
from copy import deepcopy

from ..preprocessor.imputer import Imputer
from ..preprocessor.outlier_remover import ZScoreOutlierRemover, IQROutlierRemover
from ..preprocessor.encoder import LabelEncoder, OneHotEncoder, TargetEncoder
from ..preprocessor.scaler import (
    NoScaler, LogScaler, SeriesAdapter,
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)
from ..core.pipeline import ColumnPipeline, DataFramePipeline
from ..utils.logger import Logger


class AutoPlumber:
    """
    Automated preprocessing pipeline optimizer using greedy search.
    
    This class automatically finds the best preprocessing pipeline for each column
    by trying different combinations of transformations and selecting the best
    performing ones based on model cross-validation scores.
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        scoring: str = 'accuracy',
        cv: int = 3,
        max_iterations: int = 10,
        early_stopping_rounds: int = 3,
        random_state: int = 42,
        verbose: bool = True,
        n_jobs: int = 1
    ):
        """
        Initialize AutoPlumber.
        
        Args:
            model: Scikit-learn compatible model
            scoring: Scoring metric for evaluation
            cv: Number of cross-validation folds
            max_iterations: Maximum iterations for greedy search
            early_stopping_rounds: Stop if no improvement for this many rounds
            random_state: Random state for reproducibility
            verbose: Whether to print progress information
            n_jobs: Number of parallel jobs (-1 for all processors)
        """
        self.model = model
        self.scoring = scoring
        self.cv = cv
        self.max_iterations = max_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        
        self.best_pipeline_ = None
        self.best_score_ = -np.inf
        self.search_history_ = []
        self.is_fitted = False
        
        if verbose:
            self.logger = Logger()
        
        # Define preprocessing options for different data types
        self._define_preprocessing_options()
    
    def _define_preprocessing_options(self):
        """Define available preprocessing options for different data types."""
        
        # Imputation strategies
        self.imputation_options = [
            None,  # No imputation
            Imputer(strategy='mean'),
            Imputer(strategy='median'),
            Imputer(strategy='mode'),
        ]
          # Outlier removal options
        self.outlier_options = [
            None,  # No outlier removal
            ZScoreOutlierRemover(threshold=2, capped=True),  # Only use capped versions
            ZScoreOutlierRemover(threshold=3, capped=True),
            IQROutlierRemover(threshold=1.5, capped=True),
        ]
        
        # Encoding options for categorical data
        self.encoding_options = [
            LabelEncoder(),
            OneHotEncoder(drop_first=True),
            OneHotEncoder(drop_first=False),
            TargetEncoder(smoothing=1.0),
        ]        # Scaling options for numerical data
        self.scaling_options = [
            NoScaler(),
            SeriesAdapter(StandardScaler()),
            SeriesAdapter(MinMaxScaler()),
            SeriesAdapter(RobustScaler()),
            LogScaler(base='natural'),
            LogScaler(base='log10'),
            SeriesAdapter(PowerTransformer(method='yeo-johnson')),
        ]
    
    def _detect_column_type(self, column: pd.Series) -> str:
        """Detect if column is numerical or categorical."""
        if pd.api.types.is_numeric_dtype(column):
            return 'numerical'
        else:
            return 'categorical'
    
    def _get_applicable_transformers(self, column: pd.Series, column_type: str) -> List[List[Any]]:
        """Get applicable transformer combinations for a column."""
        transformers = []
        
        # Always consider imputation first if there are missing values
        if column.isnull().any():
            if column_type == 'numerical':
                imputation_options = self.imputation_options
            else:  # categorical
                # For categorical data, only use mode and constant imputation
                imputation_options = [
                    None,
                    Imputer(strategy='mode'),
                ]
        else:
            imputation_options = [None]
        
        if column_type == 'numerical':
            # For numerical: imputation -> outlier removal -> scaling
            for imputer in imputation_options:
                for outlier_remover in self.outlier_options:
                    for scaler in self.scaling_options:
                        transformer_list = []
                        if imputer is not None:
                            transformer_list.append(deepcopy(imputer))
                        if outlier_remover is not None:
                            transformer_list.append(deepcopy(outlier_remover))
                        if scaler is not None:
                            transformer_list.append(deepcopy(scaler))
                        transformers.append(transformer_list)
        
        else:  # categorical
            # For categorical: imputation -> encoding
            for imputer in imputation_options:
                for encoder in self.encoding_options:
                    transformer_list = []
                    if imputer is not None:
                        transformer_list.append(deepcopy(imputer))
                    transformer_list.append(deepcopy(encoder))
                    transformers.append(transformer_list)
        
        return transformers
    
    def _evaluate_pipeline(self, pipeline: DataFramePipeline, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate a pipeline using cross-validation."""
        try:
            # Transform data
            X_transformed = pipeline.transform(X)
            
            # Handle empty DataFrame
            if X_transformed.empty or X_transformed.shape[1] == 0:
                return -np.inf
            
            # Handle infinite or very large values
            if not np.isfinite(X_transformed.values).all():
                return -np.inf
            
            # Evaluate with cross-validation
            scores = cross_val_score(
                clone(self.model), 
                X_transformed, 
                y, 
                cv=self.cv, 
                scoring=self.scoring,
                n_jobs=1
            )
            
            return np.mean(scores)
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Pipeline evaluation failed: {str(e)}")
            return -np.inf
    
    def _greedy_search_column(
        self, 
        column_name: str, 
        X: pd.DataFrame, 
        y: pd.Series, 
        current_pipeline: DataFramePipeline
    ) -> Tuple[ColumnPipeline, float]:
        """
        Perform greedy search for the best preprocessing pipeline for a single column.
        """
        column = X[column_name]
        column_type = self._detect_column_type(column)
        
        if self.verbose:
            self.logger.info(f"Optimizing column '{column_name}' (type: {column_type})")
        
        # Get all possible transformer combinations for this column
        transformer_combinations = self._get_applicable_transformers(column, column_type)
        
        best_score = -np.inf
        best_column_pipeline = None
        
        # Try each transformer combination
        for i, transformers in enumerate(transformer_combinations):
            try:
                # Create column pipeline
                column_pipeline = ColumnPipeline(column_name, transformers)
                
                # Create test pipeline
                test_pipeline = current_pipeline.copy()
                test_pipeline.add_column_pipeline(column_name, column_pipeline)
                
                # Fit and evaluate
                test_pipeline.fit(X, y)
                score = self._evaluate_pipeline(test_pipeline, X, y)
                
                if score > best_score:
                    best_score = score
                    best_column_pipeline = column_pipeline
                
                if self.verbose and i % 5 == 0:
                    self.logger.debug(f"  Tested {i+1}/{len(transformer_combinations)} combinations")
                    
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"  Combination {i} failed: {str(e)}")
                continue
        
        if self.verbose:
            self.logger.info(f"  Best score for '{column_name}': {best_score:.4f}")
        
        return best_column_pipeline, best_score
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AutoPlumber':
        """
        Fit AutoPlumber to find the best preprocessing pipeline.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self
        """
        if self.verbose:
            self.logger.info("Starting AutoPlumber optimization...")
            self.logger.info(f"Dataset shape: {X.shape}")
            self.logger.info(f"Columns: {list(X.columns)}")
        
        # Initialize with empty pipeline
        current_pipeline = DataFramePipeline()
        current_score = -np.inf
        
        # Keep track of search history
        self.search_history_ = []
        no_improvement_count = 0
        
        columns_to_optimize = list(X.columns)
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                self.logger.info(f"\\n--- Iteration {iteration + 1} ---")
            
            improved = False
            best_iteration_score = current_score
            best_iteration_pipeline = None
            
            # Try optimizing each column
            for column_name in columns_to_optimize:
                if self.verbose:
                    self.logger.info(f"Testing column: {column_name}")
                
                # Find best pipeline for this column
                best_column_pipeline, column_score = self._greedy_search_column(
                    column_name, X, y, current_pipeline
                )
                
                if best_column_pipeline is not None:
                    # Create new pipeline with this column optimization
                    test_pipeline = current_pipeline.copy()
                    test_pipeline.add_column_pipeline(column_name, best_column_pipeline)
                    
                    try:
                        test_pipeline.fit(X, y)
                        test_score = self._evaluate_pipeline(test_pipeline, X, y)
                        
                        if test_score > best_iteration_score:
                            best_iteration_score = test_score
                            best_iteration_pipeline = test_pipeline
                            improved = True
                            
                            if self.verbose:
                                self.logger.info(f"  Improvement found! Score: {test_score:.4f}")
                    
                    except Exception as e:
                        if self.verbose:
                            self.logger.warning(f"  Pipeline failed: {str(e)}")
            
            # Update current best if improved
            if improved and best_iteration_pipeline is not None:
                current_pipeline = best_iteration_pipeline
                current_score = best_iteration_score
                no_improvement_count = 0
                
                if self.verbose:
                    self.logger.info(f"New best score: {current_score:.4f}")
            else:
                no_improvement_count += 1
                if self.verbose:
                    self.logger.info("No improvement in this iteration")
            
            # Record iteration
            self.search_history_.append({
                'iteration': iteration + 1,
                'score': current_score,
                'improved': improved
            })
            
            # Early stopping
            if no_improvement_count >= self.early_stopping_rounds:
                if self.verbose:
                    self.logger.info(f"Early stopping after {no_improvement_count} iterations without improvement")
                break
        
        # Store best results
        self.best_pipeline_ = current_pipeline
        self.best_score_ = current_score
        self.is_fitted = True
        
        if self.verbose:
            self.logger.info(f"\\nOptimization complete!")
            self.logger.info(f"Best score: {self.best_score_:.4f}")
            self.logger.info(f"Total iterations: {len(self.search_history_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the best found pipeline."""
        if not self.is_fitted:
            raise ValueError("AutoPlumber must be fitted before transforming.")
        
        return self.best_pipeline_.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit AutoPlumber and transform data."""
        return self.fit(X, y).transform(X)
    
    def get_best_pipeline(self) -> DataFramePipeline:
        """Get the best pipeline found."""
        if not self.is_fitted:
            raise ValueError("AutoPlumber must be fitted first.")
        return self.best_pipeline_
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the best pipeline."""
        if not self.is_fitted:
            raise ValueError("AutoPlumber must be fitted first.")
        
        summary = {
            'best_score': self.best_score_,
            'total_iterations': len(self.search_history_),
            'column_pipelines': {}
        }
        
        for column_name, pipeline in self.best_pipeline_.column_pipelines.items():
            transformers = []
            for transformer in pipeline.transformers:
                transformers.append({
                    'type': type(transformer).__name__,
                    'params': getattr(transformer, '__dict__', {})
                })
            summary['column_pipelines'][column_name] = transformers
        
        return summary
    
    def get_feature_importance_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Analyze the importance of each selected feature and preprocessing step.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Dictionary containing feature importance analysis
        """
        if not self.is_fitted:
            raise ValueError("AutoPlumber must be fitted before analyzing feature importance.")
        
        analysis = {
            'selected_columns': [],
            'feature_contributions': {},
            'preprocessing_impact': {},
            'search_progression': self.search_history_
        }
        
        # Get selected columns and their pipelines
        for col_name, col_pipeline in self.best_pipeline_.column_pipelines.items():
            analysis['selected_columns'].append(col_name)
            
            # Calculate individual column contribution
            # Create pipeline with just this column
            temp_pipeline = DataFramePipeline()
            temp_pipeline.add_column_pipeline(col_name, col_pipeline)
            temp_pipeline.fit(X[[col_name]], y)
            
            # Evaluate individual contribution
            temp_score = self._evaluate_pipeline(temp_pipeline, X[[col_name]], y)
            analysis['feature_contributions'][col_name] = temp_score
            
            # Analyze preprocessing impact
            analysis['preprocessing_impact'][col_name] = {
                'transformers': [str(t) for t in col_pipeline.transformers],
                'num_transformers': len(col_pipeline.transformers)
            }
        
        return analysis
    
    def get_improvement_rate_analysis(self) -> Dict[str, Any]:
        """
        Analyze the rate of improvement throughout the search process.
        
        Returns:
            Dictionary containing improvement rate analysis
        """
        if not self.search_history_:
            return {}
        
        scores = [entry['score'] for entry in self.search_history_]
        improvements = []
        
        for i in range(1, len(scores)):
            improvement = scores[i] - scores[i-1]
            improvements.append(improvement)
        
        return {
            'total_improvement': scores[-1] - scores[0] if len(scores) > 1 else 0,
            'avg_improvement_per_iteration': np.mean(improvements) if improvements else 0,
            'improvement_trend': improvements,
            'iterations_with_improvement': len([imp for imp in improvements if imp > 0]),
            'max_single_improvement': max(improvements) if improvements else 0,
            'convergence_rate': np.std(improvements) if improvements else 0
        }
