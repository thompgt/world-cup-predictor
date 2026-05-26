import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import os


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def train_high_signal_model():
    base_path = _repo_root()
    data = pd.read_csv(os.path.join(base_path, 'data', 'features', 'high_signal_features.csv'))
    features = ['rank_diff_inv', 'rating_diff', 'form_diff', 'h_inv_rank', 'h_squad_rating', 'h_recent_form', 'squad_rank_interaction', 'rating_delta_form']
    data = data.dropna(subset=features + ['target'])
    
    X = data[features]
    y = data['target']
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    print(f"Best high-signal parameters: {grid_search.best_params_}")
    print(f"Best high-signal score: {grid_search.best_score_:.4f}")
    
    # Save optimized model
    os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
    with open(os.path.join(base_path, 'models', 'high_signal_gb.pkl'), 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    print("High-signal model saved.")

if __name__ == "__main__":
    train_high_signal_model()
