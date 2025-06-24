import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from copy import deepcopy

class ColumnPipeline:
    """Pipeline for preprocessing a single column."""
    
    def __init__(self, column_name: str, transformers: List[Any] = None):
        self.column_name = column_name
        self.transformers = transformers or []
        self.is_fitted = False
    
    def add_transformer(self, transformer: Any):
        """Add a transformer to the pipeline."""
        self.transformers.append(transformer)
    
    def fit(self, X: pd.Series, y: pd.Series = None):
        """Fit all transformers in the pipeline."""
        current_data = X.copy()
        
        for transformer in self.transformers:
            if hasattr(transformer, 'fit'):
                # Check if transformer needs target variable (like TargetEncoder)
                if 'y' in transformer.fit.__code__.co_varnames and y is not None:
                    transformer.fit(current_data, y)
                else:
                    transformer.fit(current_data)
                
                # Transform data for next step
                if hasattr(transformer, 'transform'):
                    current_data = transformer.transform(current_data)
                    # Handle case where transform returns DataFrame (like OneHotEncoder)
                    if isinstance(current_data, pd.DataFrame):
                        # For subsequent transformers, use the first column or combine
                        current_data = current_data.iloc[:, 0] if current_data.shape[1] == 1 else current_data.sum(axis=1)
        
        self.is_fitted = True
    
    def transform(self, X: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        """Transform data through all transformers."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming.")
        
        current_data = X.copy()
        
        for transformer in self.transformers:
            if hasattr(transformer, 'transform'):
                current_data = transformer.transform(current_data)
                # Handle DataFrame output from transformers like OneHotEncoder
                if isinstance(current_data, pd.DataFrame) and len(self.transformers) > self.transformers.index(transformer) + 1:
                    # If more transformers follow, convert back to Series
                    current_data = current_data.iloc[:, 0] if current_data.shape[1] == 1 else current_data.sum(axis=1)
        
        return current_data
    
    def fit_transform(self, X: pd.Series, y: pd.Series = None) -> Union[pd.Series, pd.DataFrame]:
        """Fit and transform data."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get output feature names."""
        # Check if any transformer changes the output dimension
        for transformer in reversed(self.transformers):
            if hasattr(transformer, 'feature_names') and transformer.feature_names:
                return transformer.feature_names
        
        return [self.column_name]
    
    def copy(self):
        """Create a deep copy of the pipeline."""
        return ColumnPipeline(
            column_name=self.column_name,
            transformers=[deepcopy(t) for t in self.transformers]
        )


class DataFramePipeline:
    """Pipeline for preprocessing entire DataFrame with column-specific transformations."""
    
    def __init__(self):
        self.column_pipelines: Dict[str, ColumnPipeline] = {}
        self.is_fitted = False
        self.output_columns = []
    
    def add_column_pipeline(self, column_name: str, pipeline: ColumnPipeline):
        """Add a pipeline for a specific column."""
        self.column_pipelines[column_name] = pipeline
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit all column pipelines."""
        self.output_columns = []
        
        for column_name, pipeline in self.column_pipelines.items():
            if column_name in X.columns:
                pipeline.fit(X[column_name], y)
                self.output_columns.extend(pipeline.get_feature_names())
        
        self.is_fitted = True
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame using fitted pipelines."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming.")
        
        transformed_parts = []
        
        for column_name, pipeline in self.column_pipelines.items():
            if column_name in X.columns:
                transformed_data = pipeline.transform(X[column_name])
                
                if isinstance(transformed_data, pd.DataFrame):
                    transformed_parts.append(transformed_data)
                else:
                    # Convert Series to DataFrame
                    transformed_parts.append(pd.DataFrame(
                        {column_name: transformed_data}
                    ))
        
        if transformed_parts:
            result = pd.concat(transformed_parts, axis=1)
            return result
        else:
            return pd.DataFrame(index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform DataFrame."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_column_pipeline(self, column_name: str) -> Optional[ColumnPipeline]:
        """Get pipeline for a specific column."""
        return self.column_pipelines.get(column_name)
    
    def copy(self):
        """Create a deep copy of the pipeline."""
        new_pipeline = DataFramePipeline()
        new_pipeline.column_pipelines = {
            name: pipeline.copy() 
            for name, pipeline in self.column_pipelines.items()
        }
        return new_pipeline
