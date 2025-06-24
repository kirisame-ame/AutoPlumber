import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer


class NoScaler:
    """No-op scaler that returns data unchanged."""
    
    def __init__(self):
        self.is_fitted = False
    
    def fit(self, X):
        """Fit the scaler (no-op)."""
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform the data (return unchanged)."""
        if isinstance(X, pd.Series):
            return X.copy()
        elif isinstance(X, np.ndarray):
            return X.copy()
        else:
            return X
    
    def fit_transform(self, X):
        """Fit the scaler and transform the data."""
        self.fit(X)
        return self.transform(X)
    
    def __repr__(self):
        return "NoScaler()"


class LogScaler:
    """Custom log scaler that applies log(1 + x) transformation."""
    
    def __init__(self, base='natural'):
        """
        Initialize LogScaler.
        
        Args:
            base: 'natural' for ln, 'log10' for log10, or number for custom base
        """
        self.base = base
        self.is_fitted = False
        self.min_value = None
        self.offset = None
    
    def fit(self, X):
        """Fit the scaler by determining the minimum value and offset."""
        if isinstance(X, pd.Series):
            values = X.values
        else:
            values = X.flatten() if len(X.shape) > 1 else X
            
        self.min_value = np.min(values)
        # Add offset to make all values positive if needed
        self.offset = max(0, -self.min_value + 1e-8)
        self.is_fitted = True
        return self
        
    def transform(self, X):
        """Transform the data using log scaling."""
        if not self.is_fitted:
            raise ValueError("Must fit the scaler before transforming.")
        
        if isinstance(X, pd.Series):
            # Apply offset to ensure positive values
            X_positive = X + self.offset
            
            # Ensure all values are positive and finite
            X_positive = np.maximum(X_positive, 1e-8)
            
            # Check for invalid values and handle them
            if not np.isfinite(X_positive).all():
                X_positive = np.nan_to_num(X_positive, nan=1e-8, posinf=1e8, neginf=1e-8)
            
            if self.base == 'natural':
                scaled = np.log1p(X_positive)  # log(1 + x)
            elif self.base == 'log10':
                scaled = np.log10(1 + X_positive)
            else:
                scaled = np.log(1 + X_positive) / np.log(self.base)
            
            # Handle any remaining invalid values
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=10.0, neginf=-10.0)
            
            return pd.Series(scaled, index=X.index, name=X.name)
        else:
            # Handle numpy arrays
            X_flat = X.flatten() if len(X.shape) > 1 else X
            X_positive = X_flat + self.offset
            
            # Ensure all values are positive and finite
            X_positive = np.maximum(X_positive, 1e-8)
            
            # Check for invalid values and handle them
            if not np.isfinite(X_positive).all():
                X_positive = np.nan_to_num(X_positive, nan=1e-8, posinf=1e8, neginf=1e-8)
            
            if self.base == 'natural':
                scaled = np.log1p(X_positive)
            elif self.base == 'log10':
                scaled = np.log10(1 + X_positive)
            else:
                scaled = np.log(1 + X_positive) / np.log(self.base)
            
            # Handle any remaining invalid values
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=10.0, neginf=-10.0)
            
            return scaled.reshape(X.shape)
    
    def fit_transform(self, X):
        """Fit the scaler and transform the data."""
        self.fit(X)
        return self.transform(X)
    
    def __repr__(self):
        return f"LogScaler(base={self.base})"


class SeriesAdapter:
    """
    Adapter to make sklearn scalers work with pandas Series.
    This is just for internal use in the pipeline - users don't need to use this.
    """
    
    def __init__(self, scaler):
        self.scaler = scaler
        self.is_fitted = False
    
    def fit(self, X):
        if isinstance(X, pd.Series):
            self.scaler.fit(X.values.reshape(-1, 1))
        else:
            self.scaler.fit(X.reshape(-1, 1) if len(X.shape) == 1 else X)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Must fit the scaler before transforming.")
        
        if isinstance(X, pd.Series):
            scaled = self.scaler.transform(X.values.reshape(-1, 1)).flatten()
            return pd.Series(scaled, index=X.index, name=X.name)
        else:
            input_shape = X.shape
            X_reshaped = X.reshape(-1, 1) if len(X.shape) == 1 else X
            scaled = self.scaler.transform(X_reshaped)
            return scaled.reshape(input_shape)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def __repr__(self):
        return f"SeriesAdapter({self.scaler})"
