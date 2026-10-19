import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

def load_data():
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test.csv')
    return X_train, y_train, X_test

def create_pipeline(alpha_value):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', VarianceThreshold(threshold=0.01)),
        ('model', Ridge(alpha=alpha_value))
    ])

def main():
    ALPHA = 5000.0  
    
    # Load data
    X_train, y_train, X_test = load_data()
    
    features = [col for col in X_train.columns if col != 'id']
    X = X_train[features]
    y = y_train.iloc[:, 1]
    
    # Evaluate locally using Cross-Validation
    pipeline = create_pipeline(ALPHA)
    scores = cross_val_score(
        pipeline, X, y, 
        cv=5, 
        scoring='neg_root_mean_squared_error'
    )
    rmse_scores = -scores
    print(f"📊 Local Validation RMSE (alpha={ALPHA}): {rmse_scores.mean():.2f} years (+/- {rmse_scores.std():.2f})")
    
    # Train on all training data and predict on test data
    pipeline.fit(X, y)
    X_test_features = X_test[features]
    y_pred = pipeline.predict(X_test_features)
    
    # Generate submission file
    submission = pd.DataFrame({
        'id': X_test['id'],
        'y': y_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    print("✅ 'submission.csv' file successfully generated!")

if __name__ == "__main__":
    main()