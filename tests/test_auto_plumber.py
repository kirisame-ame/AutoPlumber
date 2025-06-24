"""Test AutoPlumber main functionality."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from autoplumber import AutoPlumber


class TestAutoPlumber:
    
    def setup_method(self):
        """Set up test data."""
        # Create a simple dataset
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42
        )
        
        self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = pd.Series(y, name='target')
        
        # Add some missing values
        self.df.loc[0:5, 'feature_0'] = np.nan
        
        # Add a categorical column
        self.df['category'] = pd.Categorical(['A', 'B', 'C'] * (len(self.df) // 3) + ['A'] * (len(self.df) % 3))
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    def test_init(self):
        """Test AutoPlumber initialization."""
        auto_plumber = AutoPlumber(
            model=self.model,
            scoring='accuracy',
            cv=2,
            max_iterations=2,
            verbose=False
        )
        
        assert auto_plumber.model == self.model
        assert auto_plumber.scoring == 'accuracy'
        assert auto_plumber.cv == 2
        assert auto_plumber.max_iterations == 2
        assert not auto_plumber.is_fitted
    
    def test_detect_column_type(self):
        """Test column type detection."""
        auto_plumber = AutoPlumber(self.model, verbose=False)
        
        assert auto_plumber._detect_column_type(self.df['feature_0']) == 'numerical'
        assert auto_plumber._detect_column_type(self.df['category']) == 'categorical'
    
    def test_fit_basic(self):
        """Test basic fitting functionality."""
        auto_plumber = AutoPlumber(
            model=self.model,
            cv=2,
            max_iterations=1,
            verbose=False
        )
        
        # Should not raise an exception
        auto_plumber.fit(self.df, self.y)
        
        assert auto_plumber.is_fitted
        assert auto_plumber.best_pipeline_ is not None
        assert auto_plumber.best_score_ is not None
    
    def test_transform_before_fit(self):
        """Test that transform raises error before fit."""
        auto_plumber = AutoPlumber(self.model, verbose=False)
        
        with pytest.raises(ValueError, match="must be fitted"):
            auto_plumber.transform(self.df)
    
    def test_fit_transform(self):
        """Test fit_transform functionality."""
        auto_plumber = AutoPlumber(
            model=self.model,
            cv=2,
            max_iterations=1,
            verbose=False
        )
        
        transformed = auto_plumber.fit_transform(self.df, self.y)
        
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(self.df)
        assert auto_plumber.is_fitted
    
    def test_get_pipeline_summary(self):
        """Test pipeline summary generation."""
        auto_plumber = AutoPlumber(
            model=self.model,
            cv=2,
            max_iterations=1,
            verbose=False
        )
        
        auto_plumber.fit(self.df, self.y)
        summary = auto_plumber.get_pipeline_summary()
        
        assert 'best_score' in summary
        assert 'total_iterations' in summary
        assert 'column_pipelines' in summary
        assert isinstance(summary['column_pipelines'], dict)
