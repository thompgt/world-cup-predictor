import pandas as pd
import numpy as np
import os

def engineer_matchup_diffs():
    # Load rolling features
    results = pd.read_csv('world-cup-predictor/data/features/rolling_features.csv')
    rankings = pd.read_csv('world-cup-predictor/data/processed/rankings_2026_cycle.csv')
    
    results['date'] = pd.to_datetime(results['date'])
    rankings['date'] = pd.to_datetime(rankings['date'])
    
    # Sort rankings
    rankings = rankings.sort_values(['team', 'date'])
    
    def get_rank_at_date(team, date):
        team_ranks = rankings[rankings['team'] == team]
        if team_ranks.empty:
            return 100, 0
        past_ranks = team_ranks[team_ranks['date'] <= date]
        if past_ranks.empty:
            return team_ranks.iloc[0]['rank'], team_ranks.iloc[0]['total_points']
        latest = past_ranks.iloc[-1]
        return latest['rank'], latest['total_points']

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
    
    # Feature: Goal rolling difference
    results['goals_rolling_diff'] = results['home_rolling_goals'] - results['away_rolling_goals']
    
    # Target: Binary classification (Home Win or Not)
    # The 2022 notebook used a more complex approach but let's stick to Home Win vs Other
    results['target'] = (results['home_score'] > results['away_score']).astype(int)
    
    # Drop matches with NA scores (if any)
    results = results.dropna(subset=['home_score', 'away_score'])
    
    results.to_csv('world-cup-predictor/data/features/final_features.csv', index=False)
    print("Final features saved.")

if __name__ == "__main__":
    engineer_matchup_diffs()
