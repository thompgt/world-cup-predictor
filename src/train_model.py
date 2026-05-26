import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def train_baseline():
    base_path = _repo_root()
    data = pd.read_csv(os.path.join(base_path, 'data', 'features', 'final_features.csv'))
    
    # Define features and target
    features = ['rank_diff', 'average_rank', 'point_diff', 'is_friendly', 'goals_rolling_diff']
    
    # Drop NaNs
    data = data.dropna(subset=features + ['target'])
    
    X = data[features]
    y = data['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Baseline Model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    print("Baseline Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
    with open(os.path.join(base_path, 'models', 'baseline_gb.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print("Baseline model saved.")

if __name__ == "__main__":
    train_baseline()
