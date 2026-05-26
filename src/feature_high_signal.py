import pandas as pd
import numpy as np
import os

def engineer_high_signal_features():
    # Load data
    results = pd.read_csv('world-cup-predictor/data/processed/results_2026_cycle.csv')
    rankings = pd.read_csv('world-cup-predictor/data/processed/rankings_2026_cycle.csv')
    squad_ratings = pd.read_csv('world-cup-predictor/data/processed/squad_ratings.csv')
    
    results['date'] = pd.to_datetime(results['date'])
    rankings['date'] = pd.to_datetime(rankings['date'])
    
    # Sort
    results = results.sort_values('date')
    rankings = rankings.sort_values(['team', 'date'])
    
    # Convert squad ratings to dict
    squad_dict = dict(zip(squad_ratings['team'], squad_ratings['squad_rating']))

    def get_rank_at_date(team, date):
        team_ranks = rankings[rankings['team'] == team]
        if team_ranks.empty: return 100
        past_ranks = team_ranks[team_ranks['date'] <= date]
        if past_ranks.empty: return team_ranks.iloc[0]['rank']
        return past_ranks.iloc[-1]['rank']

    print("Calculating high-signal features...")
    teams = set(results['home_team'].unique()) | set(results['away_team'].unique())
    team_history = {team: [] for team in teams}
    
    signal_rows = []
    for idx, row in results.iterrows():
        h_team, a_team = row['home_team'], row['away_team']
        date = row['date']
        
        # 1. FIFA Ranks (Inverted: 212 - rank)
        h_r = get_rank_at_date(h_team, date)
        a_r = get_rank_at_date(a_team, date)
        h_inv_rank = 212 - h_r
        a_inv_rank = 212 - a_r
        
        # 2. Squad Ratings (EA FC 26)
        h_squad = squad_dict.get(h_team, 70)
        a_squad = squad_dict.get(a_team, 70)
        
        # 3. Recent Form (Rolling 5 matches points)
        def get_form(hist):
            if not hist: return 1.0 # Average
            return np.mean(hist[-5:])
        
        h_form = get_form(team_history[h_team])
        a_form = get_form(team_history[a_team])
        
        # Construct the 6-feature set (Plus target)
        feat_row = {
            'date': date,
            'home_team': h_team,
            'away_team': a_team,
            'rank_diff_inv': h_inv_rank - a_inv_rank,
            'rating_diff': h_squad - a_squad,
            'form_diff': h_form - a_form,
            'h_inv_rank': h_inv_rank,
            'h_squad_rating': h_squad,
            'h_recent_form': h_form,
            'target': int(row['home_score'] > row['away_score'])
        }
        signal_rows.append(feat_row)
        
        # Update history
        def get_pts(s, o):
            if s > o: return 3
            if s == o: return 1
            return 0
        team_history[h_team].append(get_pts(row['home_score'], row['away_score']))
        team_history[a_team].append(get_pts(row['away_score'], row['home_score']))

    final_df = pd.DataFrame(signal_rows)
    os.makedirs('world-cup-predictor/data/features', exist_ok=True)
    final_df.to_csv('world-cup-predictor/data/features/high_signal_features.csv', index=False)
    print("High-signal features saved.")

if __name__ == "__main__":
    engineer_high_signal_features()
