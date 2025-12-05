import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

def train():
    # 1. Load Data
    print('Loading Wine Dataset...')
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=params['base']['random_state']
    )

    # 3. Setup MLflow
    mlflow.set_experiment('WineQuality_Capstone')
    
    with mlflow.start_run():
        # Train Model
        n_est = params['model']['n_estimators']
        depth = params['model']['max_depth']
        
        print(f'Training RandomForest with n_estimators={n_est}, max_depth={depth}')
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f'Accuracy: {acc}')
        print(f'F1 Score: {f1}')

        # Log metrics and params to MLflow
        mlflow.log_param('n_estimators', n_est)
        mlflow.log_param('max_depth', depth)
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('f1_score', f1)

        # Log Model
        mlflow.sklearn.log_model(clf, 'model')
        
        # Save a local copy for DVC tracking/Docker
        os.makedirs('model', exist_ok=True)
        with open('model/metrics.txt', 'w') as f:
            f.write(f'Accuracy: {acc}\nF1: {f1}')

if __name__ == '__main__':
    train()
