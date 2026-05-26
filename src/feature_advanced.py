import pandas as pd
import numpy as np
import os

def engineer_advanced_features():
    # Load base data
    results = pd.read_csv('world-cup-predictor/data/processed/results_2026_cycle.csv')
    rankings = pd.read_csv('world-cup-predictor/data/processed/rankings_2026_cycle.csv')
    
    results['date'] = pd.to_datetime(results['date'])
    rankings['date'] = pd.to_datetime(rankings['date'])
    
    # Sort data
    results = results.sort_values('date')
    rankings = rankings.sort_values(['team', 'date'])

    # Helper for rank lookup
    def get_rank_at_date(team, date):
        team_ranks = rankings[rankings['team'] == team]
        if team_ranks.empty: return 100, 0
        past_ranks = team_ranks[team_ranks['date'] <= date]
        if past_ranks.empty: return team_ranks.iloc[0]['rank'], team_ranks.iloc[0]['total_points']
        latest = past_ranks.iloc[-1]
        return latest['rank'], latest['total_points']

    print("Calculating advanced features...")
    
    teams = set(results['home_team'].unique()) | set(results['away_team'].unique())
    team_history = {team: [] for team in teams}
    
    advanced_rows = []
    
    for idx, row in results.iterrows():
        h_team, a_team = row['home_team'], row['away_team']
        date = row['date']
        
        # 1. Basic Ranks
        h_rank, h_pts = get_rank_at_date(h_team, date)
        a_rank, a_pts = get_rank_at_date(a_team, date)
        
        # 2. Rolling features (Last 5)
        def get_rolling_stats(hist):
            if not hist: return 0, 0, 0 # goals, streak, rank_faced
            last_5 = hist[-5:]
            avg_goals = np.mean([x['goals'] for x in last_5])
            # Streak: 3 for win, 1 for draw, 0 for loss
            streak = np.mean([x['result_pts'] for x in last_5])
            avg_rank_faced = np.mean([x['opp_rank'] for x in last_5])
            return avg_goals, streak, avg_rank_faced

        h_goals_roll, h_streak, h_opp_rank = get_rolling_stats(team_history[h_team])
        a_goals_roll, a_streak, a_opp_rank = get_rolling_stats(team_history[a_team])
        
        # 3. Create the feature row
        feat_row = {
            'date': date,
            'home_team': h_team,
            'away_team': a_team,
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            'tournament': row['tournament'],
            'rank_diff': h_rank - a_rank,
            'average_rank': (h_rank + a_rank) / 2,
            'point_diff': h_pts - a_pts,
            'is_friendly': int(row['tournament'] == 'Friendly'),
            'goals_rolling_diff': h_goals_roll - a_goals_roll,
            'streak_diff': h_streak - a_streak,
            'opp_rank_diff': h_opp_rank - a_opp_rank,
            'target': int(row['home_score'] > row['away_score'])
        }
        advanced_rows.append(feat_row)
        
        # 4. Update Histories
        def get_pts(score, opp_score):
            if score > opp_score: return 3
            if score == opp_score: return 1
            return 0
        
        team_history[h_team].append({'goals': row['home_score'], 'result_pts': get_pts(row['home_score'], row['away_score']), 'opp_rank': a_rank})
        team_history[a_team].append({'goals': row['away_score'], 'result_pts': get_pts(row['away_score'], row['home_score']), 'opp_rank': h_rank})

    final_df = pd.DataFrame(advanced_rows)
    os.makedirs('world-cup-predictor/data/features', exist_ok=True)
    final_df.to_csv('world-cup-predictor/data/features/advanced_features.csv', index=False)
    print("Advanced features saved.")

if __name__ == "__main__":
    engineer_advanced_features()
