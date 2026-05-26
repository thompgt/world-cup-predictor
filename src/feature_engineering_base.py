import pandas as pd
import numpy as np

def engineer_features():
    results = pd.read_csv('world-cup-predictor/data/processed/results_2026_cycle.csv')
    rankings = pd.read_csv('world-cup-predictor/data/processed/rankings_2026_cycle.csv')
    
    results['date'] = pd.to_datetime(results['date'])
    rankings['date'] = pd.to_datetime(rankings['date'])
    
    # Sort rankings to find the latest rank before a match
    rankings = rankings.sort_values(['team', 'date'])
    
    def get_rank_at_date(team, date):
        team_ranks = rankings[rankings['team'] == team]
        if team_ranks.empty:
            return 100, 0 # Default for unknown teams
        
        past_ranks = team_ranks[team_ranks['date'] <= date]
        if past_ranks.empty:
            return team_ranks.iloc[0]['rank'], team_ranks.iloc[0]['total_points']
        
        latest = past_ranks.iloc[-1]
        return latest['rank'], latest['total_points']

    # This might be slow for many matches, let's optimize with merge_asof if possible
    # But for a few thousand matches, it's okay.
    
    print("Calculating ranks for each match...")
    home_ranks = []
    away_ranks = []
    home_pts = []
    away_pts = []
    
    for idx, row in results.iterrows():
        h_r, h_p = get_rank_at_date(row['home_team'], row['date'])
        a_r, a_p = get_rank_at_date(row['away_team'], row['date'])
        home_ranks.append(h_r)
        away_ranks.append(a_r)
        home_pts.append(h_p)
        away_pts.append(a_p)
    
    results['home_rank'] = home_ranks
    results['away_rank'] = away_ranks
    results['home_points'] = home_pts
    results['away_points'] = away_pts
    
    # Feature: Rank difference
    results['rank_diff'] = results['home_rank'] - results['away_rank']
    results['point_diff'] = results['home_points'] - results['away_points']
    
    # Feature: Tournament type (Friendly vs Competitive)
    results['is_friendly'] = results['tournament'] == 'Friendly'
    
    # Target: Home Win (1), Draw (0), Away Win (-1) -> We'll simplify for binary classification later
    # 2022 notebook used: 1 if home_score > away_score else 0
    results['target'] = (results['home_score'] > results['away_score']).astype(int)
    
    # Save base features
    os.makedirs('world-cup-predictor/data/features', exist_ok=True)
    results.to_csv('world-cup-predictor/data/features/base_features.csv', index=False)
    print("Base features saved.")

if __name__ == "__main__":
    import os
    engineer_features()
