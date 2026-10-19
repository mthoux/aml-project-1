import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge

def load_data():
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test.csv')
    return X_train, y_train, X_test

def preprocess_data(X_train, y_train, X_test):
    features = [col for col in X_train.columns if col != 'id']
    
    # Impute missing values using the median
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    X_train_imputed[features] = imputer.fit_transform(X_train[features])
    X_test_imputed[features] = imputer.transform(X_test[features])
    
    # Remove outliers
    iso = IsolationForest(contamination=0.01, random_state=42)
    outlier_preds = iso.fit_predict(X_train_imputed[features])
    clean_mask = (outlier_preds != -1)
    
    X_train_clean = X_train_imputed[clean_mask].copy()
    y_train_clean = y_train[clean_mask].copy()
    
    # Feature Selection using variance threshold
    selector = VarianceThreshold(threshold=0.01)
    X_train_selected = selector.fit_transform(X_train_clean[features])
    X_test_selected = selector.transform(X_test_imputed[features])
    
    return X_train_selected, y_train_clean, X_test_selected, X_test['id']

def train_and_predict(X_train, y_train, X_test):
    model = Ridge(alpha=1.0)
    
    # Extract ages column
    y_ages = y_train.iloc[:, 1]
    model.fit(X_train, y_ages)
    
    y_pred = model.predict(X_test)
    return y_pred

def main():
    X_train, y_train, X_test = load_data()
    
    X_train_clean, y_train_clean, X_test_clean, test_ids = preprocess_data(X_train, y_train, X_test)
    
    y_pred = train_and_predict(X_train_clean, y_train_clean, X_test_clean)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'y': y_pred
    })
    
    submission.to_csv('submission.csv', index=False)
    print("✅ 'submission.csv' file successfully generated!")

if __name__ == "__main__":
    main()