from autoplumber.preprocessor import outlier_remover
import pandas as pd

def test_zscore_outlier_remover():
    data = pd.Series([1, 2, 3, 4, 5, 100])
    remover = outlier_remover.ZScoreOutlierRemover(threshold=2)
    remover.fit(data)
    transformed_data = remover.transform(data)
    assert len(transformed_data) == 5  # 100 should be removed

def test_zscore_outlier_remover_capped():
    data = pd.Series([1, 2, 3, 4, 5, 100])
    remover = outlier_remover.ZScoreOutlierRemover(threshold=2, capped=True)
    remover.fit(data)
    transformed_data = remover.transform(data)
    assert len(transformed_data) == 6  # All values should remain, but 100 should be capped
    assert transformed_data == pd.Series([1, 2, 3, 4, 5, 5])  # 100 should be capped to the upper bound

def test_iqr_outlier_remover():
    data = pd.Series([1, 2, 3, 4, 5, 100])
    remover = outlier_remover.IQROutlierRemover(threshold=1.5)
    remover.fit(data)
    transformed_data = remover.transform(data)
    assert len(transformed_data) == 5  # 100 should be removed
    
def test_iqr_outlier_remover_capped():
    data = pd.Series([1, 2, 3, 4, 5, 100])
    remover = outlier_remover.IQROutlierRemover(threshold=1.5, capped=True)
    remover.fit(data)
    transformed_data = remover.transform(data)
    assert len(transformed_data) == 6  # All values should remain, but 100 should be capped
    assert transformed_data == pd.Series([1, 2, 3, 4, 5, 5])  # 100 should be capped to the upper bound