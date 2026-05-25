import pandas as pd
import numpy as np
import os

def calculate_rolling_features():
    results = pd.read_csv('world-cup-predictor/data/processed/results_2026_cycle.csv')
    results['date'] = pd.to_datetime(results['date'])
    results = results.sort_values('date')
    
    # We need to handle each team's history
    teams = set(results['home_team'].unique()) | set(results['away_team'].unique())
    
    # Store rolling data in a dictionary
    team_history = {team: [] for team in teams}
    
    rolling_home_goals = []
    rolling_away_goals = []
    
    for idx, row in results.iterrows():
        # Get history for home team
        h_hist = team_history[row['home_team']]
        if len(h_hist) == 0:
            rolling_home_goals.append(0)
        else:
            # Last 5 games avg goals
            rolling_home_goals.append(np.mean([x['goals'] for x in h_hist[-5:]]))
            
        # Get history for away team
        a_hist = team_history[row['away_team']]
        if len(a_hist) == 0:
            rolling_away_goals.append(0)
        else:
            rolling_away_goals.append(np.mean([x['goals'] for x in a_hist[-5:]]))
            
        # Update history
        team_history[row['home_team']].append({'goals': row['home_score'], 'date': row['date']})
        team_history[row['away_team']].append({'goals': row['away_score'], 'date': row['date']})

    results['home_rolling_goals'] = rolling_home_goals
    results['away_rolling_goals'] = rolling_away_goals
    
    os.makedirs('world-cup-predictor/data/features', exist_ok=True)
    results.to_csv('world-cup-predictor/data/features/rolling_features.csv', index=False)
    print("Rolling features saved.")

if __name__ == "__main__":
    calculate_rolling_features()
