"""Test pipeline components."""

import pytest
import pandas as pd
import numpy as np

from autoplumber.core.pipeline import ColumnPipeline, DataFramePipeline
from autoplumber.preprocessor.imputer import Imputer
from autoplumber.preprocessor.scaler import StandardScaler


class TestColumnPipeline:
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.Series([1, 2, np.nan, 4, 5], name='test_column')
        self.target = pd.Series([0, 1, 0, 1, 0])
    
    def test_init(self):
        """Test ColumnPipeline initialization."""
        pipeline = ColumnPipeline('test_column')
        
        assert pipeline.column_name == 'test_column'
        assert pipeline.transformers == []
        assert not pipeline.is_fitted
    
    def test_add_transformer(self):
        """Test adding transformers."""
        pipeline = ColumnPipeline('test_column')
        imputer = Imputer(strategy='mean')
        
        pipeline.add_transformer(imputer)
        
        assert len(pipeline.transformers) == 1
        assert pipeline.transformers[0] == imputer
    
    def test_fit_transform(self):
        """Test fitting and transforming data."""
        pipeline = ColumnPipeline('test_column')
        pipeline.add_transformer(Imputer(strategy='mean'))
        pipeline.add_transformer(StandardScaler())
        
        result = pipeline.fit_transform(self.data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.data)
        assert not result.isnull().any()  # No missing values after imputation
        assert pipeline.is_fitted
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        pipeline = ColumnPipeline('test_column')
        
        feature_names = pipeline.get_feature_names()
        
        assert feature_names == ['test_column']
    
    def test_copy(self):
        """Test copying pipeline."""
        pipeline = ColumnPipeline('test_column')
        pipeline.add_transformer(Imputer(strategy='mean'))
        
        copied = pipeline.copy()
        
        assert copied.column_name == pipeline.column_name
        assert len(copied.transformers) == len(pipeline.transformers)
        assert copied.transformers[0] is not pipeline.transformers[0]  # Deep copy


class TestDataFramePipeline:
    
    def setup_method(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
        self.target = pd.Series([0, 1, 0, 1, 0])
    
    def test_init(self):
        """Test DataFramePipeline initialization."""
        pipeline = DataFramePipeline()
        
        assert pipeline.column_pipelines == {}
        assert not pipeline.is_fitted
    
    def test_add_column_pipeline(self):
        """Test adding column pipelines."""
        df_pipeline = DataFramePipeline()
        col_pipeline = ColumnPipeline('numeric')
        
        df_pipeline.add_column_pipeline('numeric', col_pipeline)
        
        assert 'numeric' in df_pipeline.column_pipelines
        assert df_pipeline.column_pipelines['numeric'] == col_pipeline
    
    def test_fit_transform(self):
        """Test fitting and transforming DataFrame."""
        df_pipeline = DataFramePipeline()
        
        # Add pipeline for numeric column
        numeric_pipeline = ColumnPipeline('numeric')
        numeric_pipeline.add_transformer(Imputer(strategy='mean'))
        df_pipeline.add_column_pipeline('numeric', numeric_pipeline)
        
        result = df_pipeline.fit_transform(self.data, self.target)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.data)
        assert df_pipeline.is_fitted
    
    def test_get_column_pipeline(self):
        """Test getting specific column pipeline."""
        df_pipeline = DataFramePipeline()
        col_pipeline = ColumnPipeline('numeric')
        df_pipeline.add_column_pipeline('numeric', col_pipeline)
        
        retrieved = df_pipeline.get_column_pipeline('numeric')
        
        assert retrieved == col_pipeline
        assert df_pipeline.get_column_pipeline('nonexistent') is None
    
    def test_copy(self):
        """Test copying DataFrame pipeline."""
        df_pipeline = DataFramePipeline()
        col_pipeline = ColumnPipeline('numeric')
        df_pipeline.add_column_pipeline('numeric', col_pipeline)
        
        copied = df_pipeline.copy()
        
        assert len(copied.column_pipelines) == len(df_pipeline.column_pipelines)
        assert 'numeric' in copied.column_pipelines
        assert copied.column_pipelines['numeric'] is not df_pipeline.column_pipelines['numeric']
