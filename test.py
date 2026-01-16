import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_and_compare_models(abalone_df):
    """
    Train and compare different ML models for abalone age prediction
    
    Args:
        abalone_df: Cleaned DataFrame containing abalone data
    
    Returns:
        metrics_data: Dictionary containing trained models and their metrics
        scaler: Fitted StandardScaler object
        feature_names: List of feature names after preprocessing
    """
    
    # Remove Ring_category column if it exists
    if 'Ring_category' in abalone_df.columns:
        abalone_df = abalone_df.drop(columns=['Ring_category'])
    
    # Prepare features and target
    y = abalone_df['Rings']
    X = abalone_df.drop(columns=['Rings'])
    
    # One-Hot Encode the categorical column
    X = pd.get_dummies(X, columns=['Type'], drop_first=True)
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # HYPERPARAMETER TUNING FOR EACH MODEL
    
    # 1. Linear Regression
    print("Training Linear Regression...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    # 2. KNN Regression with GridSearchCV
    print("Hyperparameter tuning for KNN Regression...")
    knn_params = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    knn_model = KNeighborsRegressor()
    knn_grid = GridSearchCV(knn_model, knn_params, cv=5, scoring='r2', n_jobs=-1)
    knn_grid.fit(X_train, y_train)
    best_knn = knn_grid.best_estimator_
    
    # 3. SVR with RandomizedSearchCV
    print("Hyperparameter tuning for SVR...")
    svr_params = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.5, 1.0],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }
    
    svr_model = SVR()
    svr_random = RandomizedSearchCV(svr_model, svr_params, n_iter=20, cv=5, 
                                    scoring='r2', random_state=42, n_jobs=-1)
    svr_random.fit(X_train, y_train)
    best_svr = svr_random.best_estimator_
    
    # Make predictions
    y_pred_linear = linear_model.predict(X_test)
    y_pred_knn = best_knn.predict(X_test)
    y_pred_svr = best_svr.predict(X_test)
    
    # Calculate metrics
    metrics_data = {
        'Linear Regression': {
            'model': linear_model,
            'mse': mean_squared_error(y_test, y_pred_linear),
            'r2': r2_score(y_test, y_pred_linear),
            'mae': mean_absolute_error(y_test, y_pred_linear),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_linear)),
            'best_params': 'No hyperparameters'
        },
        'KNN Regression': {
            'model': best_knn,
            'mse': mean_squared_error(y_test, y_pred_knn),
            'r2': r2_score(y_test, y_pred_knn),
            'mae': mean_absolute_error(y_test, y_pred_knn),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_knn)),
            'best_params': knn_grid.best_params_
        },
        'SVR': {
            'model': best_svr,
            'mse': mean_squared_error(y_test, y_pred_svr),
            'r2': r2_score(y_test, y_pred_svr),
            'mae': mean_absolute_error(y_test, y_pred_svr),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_svr)),
            'best_params': svr_random.best_params_
        }
    }
    
    print("Model training completed!")
    return metrics_data, scaler, feature_names

def train_single_model(abalone_df, model_name):
    """
    Train a single ML model for abalone age prediction
    
    Args:
        abalone_df: Cleaned DataFrame containing abalone data
        model_name: Name of the model to train ('Linear Regression', 'KNN Regression', or 'SVR')
    
    Returns:
        metrics_data: Dictionary containing trained model and its metrics
        scaler: Fitted StandardScaler object
        feature_names: List of feature names after preprocessing
        model: The trained model
    """
    
    # Prepare features and target
    y = abalone_df['Rings']
    X = abalone_df.drop(columns=['Rings'])
    
    # One-Hot Encode the categorical column
    X = pd.get_dummies(X, columns=['Type'], drop_first=True)
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    metrics_data = {}
    
    if model_name == 'Linear Regression':
        print(f"Training {model_name}...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics_data[model_name] = {
            'model': model,
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'best_params': 'No hyperparameters'
        }
    
    elif model_name == 'KNN Regression':
        print(f"Hyperparameter tuning for {model_name}...")
        knn_params = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        knn_model = KNeighborsRegressor()
        knn_grid = GridSearchCV(knn_model, knn_params, cv=5, scoring='r2', n_jobs=-1)
        knn_grid.fit(X_train, y_train)
        model = knn_grid.best_estimator_
        y_pred = model.predict(X_test)
        
        metrics_data[model_name] = {
            'model': model,
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'best_params': knn_grid.best_params_
        }
    
    elif model_name == 'SVR':
        print(f"Hyperparameter tuning for {model_name}...")
        svr_params = {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 0.5, 1.0],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1]
        }
        
        svr_model = SVR()
        svr_random = RandomizedSearchCV(svr_model, svr_params, n_iter=20, cv=5, 
                                        scoring='r2', random_state=42, n_jobs=-1)
        svr_random.fit(X_train, y_train)
        model = svr_random.best_estimator_
        y_pred = model.predict(X_test)
        
        metrics_data[model_name] = {
            'model': model,
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'best_params': svr_random.best_params_
        }
    else:
        print(f"Unknown model name: {model_name}")
        return None, None, None, None
    
    print(f"{model_name} training completed!")
    return metrics_data, scaler, feature_names, model