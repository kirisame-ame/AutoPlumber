import pandas as pd
class ZScoreOutlierRemover:
    """Outlier remover using Z-Score method. Default threshold is 3 standard deviations."""
    def __init__(self, threshold=3,capped=False):
        self.threshold = threshold
        self.capped = capped

    def fit(self, X:pd.Series):
        self.mean = X.mean()
        self.std = X.std()
        self.lower_bound = self.mean - self.threshold * self.std
        self.upper_bound = self.mean + self.threshold * self.std
    
    def transform(self, X:pd.Series)-> pd.Series:
        """Transform the data by removing outliers or capping them."""
        if self.capped:
            return X.clip(lower=self.lower_bound, upper=self.upper_bound)
        else:
            return X[(X >= self.lower_bound) & (X <= self.upper_bound)]
    
    def fit_transform(self, X) -> pd.Series:
        """Fit the model and transform the data."""
        self.fit(X)
        return self.transform(X)
    
class IQROutlierRemover:
    """Outlier remover using Interquartile Range (IQR) method. Default threshold is 1.5 times the IQR."""
    def __init__(self, threshold=1.5,capped=False):
        self.threshold = threshold
        self.capped = capped

    def fit(self, X:pd.Series):
        self.q1 = X.quantile(0.25)
        self.q3 = X.quantile(0.75)
        self.iqr = self.q3 - self.q1
        self.lower_bound = self.q1 - self.threshold * self.iqr
        self.upper_bound = self.q3 + self.threshold * self.iqr

    def transform(self, X:pd.Series) -> pd.Series:
        """Transform the data by removing outliers or capping them."""
        if self.capped:
            return X.clip(lower=self.lower_bound, upper=self.upper_bound)
        else:
            return X[(X >= self.lower_bound) & (X <= self.upper_bound)]

    def fit_transform(self, X:pd.Series) -> pd.Series:
        """Fit the model and transform the data."""
        self.fit(X)
        return self.transform(X)