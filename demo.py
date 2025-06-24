"""
AutoPlumber Demo - Automated Preprocessing Pipeline Optimization

This script demonstrates how to use AutoPlumber to automatically find
the best preprocessing pipeline for your dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Import AutoPlumber
from autoplumber import AutoPlumber

def create_sample_data():
    """Create a sample dataset with mixed data types and preprocessing challenges."""
    
    # Generate base classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f'num_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Add missing values
    np.random.seed(42)
    for col in df.columns[:4]:
        missing_idx = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    # Add outliers
    outlier_cols = ['num_0', 'num_1']
    for col in outlier_cols:
        outlier_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[outlier_idx, col] = df[col].mean() + 5 * df[col].std()
    
    # Add categorical columns
    categories_1 = ['A', 'B', 'C', 'D']
    categories_2 = ['Type1', 'Type2', 'Type3']
    
    df['cat_1'] = np.random.choice(categories_1, size=len(df))
    df['cat_2'] = np.random.choice(categories_2, size=len(df))
    
    # Add missing values to categorical columns
    cat_missing_idx = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
    df.loc[cat_missing_idx, 'cat_1'] = np.nan
    
    return df

def main():
    """Main demo function."""
    
    print("🔧 AutoPlumber Demo - Automated Preprocessing Optimization")
    print("=" * 60)
    
    # Check if running in demo mode (for CI)
    import os
    demo_mode = os.environ.get('AUTOPLUMBER_DEMO_MODE', '0') == '1'
    
    # Create sample data
    print("\\n📊 Creating sample dataset...")
    data = create_sample_data()
    
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print("\\nFirst few rows:")
    print(data.head())
    
    print("\\nMissing values:")
    print(data.isnull().sum())
    
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    
    # Initialize AutoPlumber - reduce iterations for demo mode
    print("\\n🤖 Initializing AutoPlumber...")
    auto_plumber = AutoPlumber(
        model=model,
        scoring='accuracy',
        cv=2 if demo_mode else 3,
        max_iterations=2 if demo_mode else 5,  # Reduced for demo
        early_stopping_rounds=1 if demo_mode else 2,
        random_state=42,
        verbose=not demo_mode  # Reduce verbosity in CI
    )
    
    # Fit AutoPlumber
    print("\\n🔍 Starting preprocessing optimization...")
    print("This may take a few minutes as AutoPlumber tests different preprocessing combinations...")
    
    auto_plumber.fit(X_train, y_train)
    
    # Get results
    print("\\n✅ Optimization complete!")
    print(f"Best cross-validation score: {auto_plumber.best_score_:.4f}")
    
    # Transform data
    print("\\n🔄 Transforming data...")
    X_train_transformed = auto_plumber.transform(X_train)
    X_test_transformed = auto_plumber.transform(X_test)
    
    print(f"Original training shape: {X_train.shape}")
    print(f"Transformed training shape: {X_train_transformed.shape}")
    print(f"Transformed columns: {list(X_train_transformed.columns)}")
    
    # Train final model
    print("\\n🎯 Training final model...")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X_train_transformed, y_train)
    
    # Evaluate
    train_score = final_model.score(X_train_transformed, y_train)
    test_score = final_model.score(X_test_transformed, y_test)
    
    print(f"Final model - Training accuracy: {train_score:.4f}")
    print(f"Final model - Test accuracy: {test_score:.4f}")
    
    # Show pipeline summary
    print("\\n📋 Best Pipeline Summary:")
    print("-" * 40)
    
    summary = auto_plumber.get_pipeline_summary()
    for column, transformers in summary['column_pipelines'].items():
        print(f"\\n{column}:")
        for i, transformer in enumerate(transformers, 1):
            params_str = ", ".join([f"{k}={v}" for k, v in transformer['params'].items() 
                                  if not k.startswith('_') and k not in ['is_fitted']])
            if params_str:
                print(f"  {i}. {transformer['type']}({params_str})")
            else:
                print(f"  {i}. {transformer['type']}()")
    
    print(f"\\nTotal search iterations: {summary['total_iterations']}")
    print(f"Best CV score: {summary['best_score']:.4f}")
    
    # Example of using the pipeline on new data
    print("\\n🆕 Example: Processing new data...")
    new_data = create_sample_data().drop('target', axis=1).head(5)
    print("New data shape:", new_data.shape)
    
    new_data_transformed = auto_plumber.transform(new_data)
    print("Transformed new data shape:", new_data_transformed.shape)
    
    predictions = final_model.predict(new_data_transformed)
    print(f"Predictions: {predictions}")
    
    print("\\n🎉 Demo complete! AutoPlumber successfully found and applied the best preprocessing pipeline.")

if __name__ == "__main__":
    main()
