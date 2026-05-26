import pandas as pd
import numpy as np
import os

def engineer_weighted_features():
    # Load base data
    results = pd.read_csv('world-cup-predictor/data/processed/results_2026_cycle.csv')
    rankings = pd.read_csv('world-cup-predictor/data/processed/rankings_2026_cycle.csv')
    
    results['date'] = pd.to_datetime(results['date'])
    rankings['date'] = pd.to_datetime(rankings['date'])
    
    # Sort data
    results = results.sort_values('date')
    rankings = rankings.sort_values(['team', 'date'])

    # 1. Match Weighting Logic
    def get_tournament_weight(tournament):
        if 'FIFA World Cup' in tournament: return 1.5
        if any(x in tournament for x in ['Euro', 'Copa América', 'Asian Cup', 'African Cup of Nations']): return 1.2
        if 'Nations League' in tournament or 'Qualifying' in tournament: return 1.0
        return 0.8 # Friendly and others

    # 2. Temporal Weighting Logic (Decay)
    # Reference date: WC 2026 Start
    ref_date = pd.to_datetime('2026-06-11')
    def get_temporal_weight(date):
        days_diff = (ref_date - date).days
        # Exponential decay: e^(-lambda * days)
        # We want lambda such that 2 years ago (730 days) is ~0.5 indicative
        decay_lambda = -np.log(0.5) / 730
        return np.exp(-decay_lambda * days_diff)

    def get_rank_at_date(team, date):
        team_ranks = rankings[rankings['team'] == team]
        if team_ranks.empty: return 100, 0
        past_ranks = team_ranks[team_ranks['date'] <= date]
        if past_ranks.empty: return team_ranks.iloc[0]['rank'], team_ranks.iloc[0]['total_points']
        latest = past_ranks.iloc[-1]
        return latest['rank'], latest['total_points']

    print("Calculating weighted features...")
    teams = set(results['home_team'].unique()) | set(results['away_team'].unique())
    team_history = {team: [] for team in teams}
    
    advanced_rows = []
    for idx, row in results.iterrows():
        h_team, a_team = row['home_team'], row['away_team']
        date = row['date']
        
        # Basic Ranks
        h_rank, h_pts = get_rank_at_date(h_team, date)
        a_rank, a_pts = get_rank_at_date(a_team, date)
        
        # Weighted Rolling Stats
        def get_weighted_rolling(hist):
            if not hist: return 0, 0, 0
            last_5 = hist[-5:]
            # Weights are product of tournament and temporal
            weights = [x['weight'] for x in last_5]
            total_w = sum(weights)
            avg_goals = sum([x['goals'] * x['weight'] for x in last_5]) / total_w
            streak = sum([x['result_pts'] * x['weight'] for x in last_5]) / total_w
            avg_rank_faced = sum([x['opp_rank'] * x['weight'] for x in last_5]) / total_w
            return avg_goals, streak, avg_rank_faced

        h_goals_roll, h_streak, h_opp_rank = get_weighted_rolling(team_history[h_team])
        a_goals_roll, a_streak, a_opp_rank = get_weighted_rolling(team_history[a_team])
        
        match_w = get_tournament_weight(row['tournament']) * get_temporal_weight(date)
        
        feat_row = {
            'date': date,
            'home_team': h_team,
            'away_team': a_team,
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            'rank_diff': h_rank - a_rank,
            'average_rank': (h_rank + a_rank) / 2,
            'point_diff': h_pts - a_pts,
            'is_friendly': int(row['tournament'] == 'Friendly'),
            'goals_rolling_diff': h_goals_roll - a_goals_roll,
            'streak_diff': h_streak - a_streak,
            'opp_rank_diff': h_opp_rank - a_opp_rank,
            'match_weight': match_w,
            'target': int(row['home_score'] > row['away_score'])
        }
        advanced_rows.append(feat_row)
        
        def get_pts(score, opp_score):
            if score > opp_score: return 3
            if score == opp_score: return 1
            return 0
        
        team_history[h_team].append({'goals': row['home_score'], 'result_pts': get_pts(row['home_score'], row['away_score']), 'opp_rank': a_rank, 'weight': match_w})
        team_history[a_team].append({'goals': row['away_score'], 'result_pts': get_pts(row['away_score'], row['home_score']), 'opp_rank': h_rank, 'weight': match_w})

    final_df = pd.DataFrame(advanced_rows)
    os.makedirs('world-cup-predictor/data/features', exist_ok=True)
    final_df.to_csv('world-cup-predictor/data/features/weighted_features.csv', index=False)
    print("Weighted features saved.")

if __name__ == "__main__":
    engineer_weighted_features()
