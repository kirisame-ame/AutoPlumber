import pandas as pd
class Imputer:
    def __init__(self, strategy='mean', constant=None):
        """
        Initializes the Imputer with a specified strategy.

        :param strategy: The strategy to use for imputing missing values.
                         Options are 'mean', 'median',or 'mode'.
                         Or 'constant' to fill with a constant value.
        :type strategy: str
        """
        self.strategy = strategy
        self.constant = constant
        if strategy not in ['mean', 'median', 'mode', 'constant']:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'constant'.")

    def fit(self, X:pd.Series):
        """
        Fits the imputer to the data.

        :param X: The input data.
        :type X: pd.Series
        """
        if self.strategy == 'mean':
            self.fill_value = X.mean()
        elif self.strategy == 'median':
            self.fill_value = X.median()
        elif self.strategy == 'mode':
            self.fill_value = X.mode()[0]
        elif self.strategy == 'constant':
            if self.constant is None:
                raise ValueError("Constant value must be provided for 'constant' strategy.")
            self.fill_value = self.constant

    def transform(self, X:pd.Series) -> pd.Series:
        """
        Transforms the data by imputing missing values.

        :param X: The input data.
        :return: The transformed data with missing values imputed.
        """
        if self.fill_value is None:
            raise ValueError("Imputer has not been fitted yet. Call fit() before transform().")
        return X.fillna(self.fill_value)
    
    def fit_transform(self, X:pd.Series) -> pd.Series:
        """
        Fits the imputer to the data and transforms it.
        
        :param X: The input data.
        :return: The transformed data with missing values imputed.
        """
        self.fit(X)
        return self.transform(X)
        