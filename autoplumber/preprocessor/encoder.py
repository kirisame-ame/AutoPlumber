import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder

class LabelEncoder:
    """Simple label encoder for categorical variables."""
    
    def __init__(self):
        self.encoder = SkLabelEncoder()
        self.is_fitted = False
    
    def fit(self, X: pd.Series):
        """Fit the encoder to the data."""
        self.encoder.fit(X.dropna())
        self.is_fitted = True
    
    def transform(self, X: pd.Series) -> pd.Series:
        """Transform the data using label encoding."""
        if not self.is_fitted:
            raise ValueError("Must fit the encoder before transforming.")
        
        # Handle NaN values by creating a mask
        mask = X.notna()
        result = pd.Series(index=X.index, dtype=int)
        
        # Transform only non-null values
        if mask.any():
            result[mask] = self.encoder.transform(X[mask])
        
        return result
    
    def fit_transform(self, X: pd.Series) -> pd.Series:
        """Fit the encoder and transform the data."""
        self.fit(X)
        return self.transform(X)


class OneHotEncoder:
    """One-hot encoder for categorical variables."""
    
    def __init__(self, drop_first=False, max_categories=10):
        self.drop_first = drop_first
        self.max_categories = max_categories
        self.encoder = None
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: pd.Series):
        """Fit the encoder to the data."""
        # Limit to most frequent categories if too many
        value_counts = X.value_counts()
        if len(value_counts) > self.max_categories:
            top_categories = value_counts.head(self.max_categories).index
            X_fit = X[X.isin(top_categories)]
        else:
            X_fit = X
        
        self.encoder = SkOneHotEncoder(
            drop='first' if self.drop_first else None,
            sparse_output=False,
            handle_unknown='ignore'
        )
        self.encoder.fit(X_fit.values.reshape(-1, 1))
        
        # Generate feature names
        categories = self.encoder.categories_[0]
        if self.drop_first and len(categories) > 1:
            self.feature_names = [f"{X.name}_{cat}" for cat in categories[1:]]
        else:
            self.feature_names = [f"{X.name}_{cat}" for cat in categories]
        
        self.is_fitted = True
    
    def transform(self, X: pd.Series) -> pd.DataFrame:
        """Transform the data using one-hot encoding."""
        if not self.is_fitted:
            raise ValueError("Must fit the encoder before transforming.")
        
        encoded = self.encoder.transform(X.values.reshape(-1, 1))
        return pd.DataFrame(encoded, columns=self.feature_names, index=X.index)
    
    def fit_transform(self, X: pd.Series) -> pd.DataFrame:
        """Fit the encoder and transform the data."""
        self.fit(X)
        return self.transform(X)


class TargetEncoder:
    """Target encoder for categorical variables (mean encoding)."""
    
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.global_mean = None
        self.category_means = None
        self.is_fitted = False
    
    def fit(self, X: pd.Series, y: pd.Series):
        """Fit the encoder to the data."""
        self.global_mean = y.mean()
          # Calculate category means with smoothing
        category_stats = pd.DataFrame({'category': X, 'target': y}).groupby('category', observed=True).agg({
            'target': ['mean', 'count']
        }).round(6)
        
        category_stats.columns = ['mean', 'count']
        
        # Apply smoothing
        self.category_means = (
            (category_stats['mean'] * category_stats['count'] + 
             self.global_mean * self.smoothing) /
            (category_stats['count'] + self.smoothing)
        ).to_dict()
        
        self.is_fitted = True
    
    def transform(self, X: pd.Series) -> pd.Series:
        """Transform the data using target encoding."""
        if not self.is_fitted:
            raise ValueError("Must fit the encoder before transforming.")
        
        return X.map(self.category_means).fillna(self.global_mean)
    
    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        """Fit the encoder and transform the data."""
        self.fit(X, y)
        return self.transform(X)
