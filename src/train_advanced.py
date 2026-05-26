import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import os

def tune_advanced_model():
    data = pd.read_csv('world-cup-predictor/data/features/advanced_features.csv')
    features = ['rank_diff', 'average_rank', 'point_diff', 'is_friendly', 'goals_rolling_diff', 'streak_diff', 'opp_rank_diff']
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
    
    print(f"Best advanced parameters: {grid_search.best_params_}")
    print(f"Best advanced score: {grid_search.best_score_:.4f}")
    
    # Save optimized model
    os.makedirs('world-cup-predictor/models', exist_ok=True)
    with open('world-cup-predictor/models/advanced_gb.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    print("Advanced model saved.")

if __name__ == "__main__":
    tune_advanced_model()
