import pandas as pd
import numpy as np
import pickle
import os

# Standardization mapping (same as used before)
mapping = {
    'Cape Verde': 'Cabo Verde',
    'DR Congo': 'Congo DR',
    'Czech Republic': 'Czechia',
    'Ivory Coast': "Côte d'Ivoire",
    'Iran': 'IR Iran',
    'Gambia': 'The Gambia',
    'Brunei': 'Brunei Darussalam',
    'South Korea': 'Korea Republic',
    'North Korea': 'Korea DPR',
    'Taiwan': 'Chinese Taipei',
    'Kyrgyzstan': 'Kyrgyz Republic',
}

def get_current_features(team, date, rankings, results):
    # Get rank
    team_ranks = rankings[rankings['team'] == team]
    past_ranks = team_ranks[team_ranks['date'] < date]
    if past_ranks.empty:
        rank, pts = 100, 0
    else:
        latest = past_ranks.iloc[-1]
        rank, pts = latest['rank'], latest['total_points']
    
    # Get rolling features
    team_results = results[((results['home_team'] == team) | (results['away_team'] == team)) & (results['date'] < date)]
    if team_results.empty:
        rolling_goals = 0
        streak = 0
        opp_rank = 100
    else:
        last_5 = team_results.tail(5)
        goals = []
        pts_list = []
        opp_ranks = []
        for _, r in last_5.iterrows():
            if r['home_team'] == team:
                goals.append(r['home_score'])
                opp_ranks.append(r['away_rank'] if 'away_rank' in r else 100)
                if r['home_score'] > r['away_score']: pts_list.append(3)
                elif r['home_score'] == r['away_score']: pts_list.append(1)
                else: pts_list.append(0)
            else:
                goals.append(r['away_score'])
                opp_ranks.append(r['home_rank'] if 'home_rank' in r else 100)
                if r['away_score'] > r['home_score']: pts_list.append(3)
                elif r['away_score'] == r['home_score']: pts_list.append(1)
                else: pts_list.append(0)
        rolling_goals = np.mean(goals)
        streak = np.mean(pts_list)
        opp_rank = np.mean(opp_ranks)
        
    return rank, pts, rolling_goals, streak, opp_rank

def predict_match(home, away, date, model, rankings, results):
    h_rank, h_pts, h_roll, h_strk, h_opp_r = get_current_features(home, date, rankings, results)
    a_rank, a_pts, a_roll, a_strk, a_opp_r = get_current_features(away, date, rankings, results)
    
    def get_features(h_r, h_p, h_rl, h_s, h_or, a_r, a_p, a_rl, a_s, a_or):
        df = pd.DataFrame([{
            'rank_diff': h_r - a_r,
            'average_rank': (h_r + a_r) / 2,
            'point_diff': h_p - a_p,
            'is_friendly': 0,
            'goals_rolling_diff': h_rl - a_rl,
            'streak_diff': h_s - a_s,
            'opp_rank_diff': h_or - a_or
        }])
        return df.fillna(0)

    f1 = get_features(h_rank, h_pts, h_roll, h_strk, h_opp_r, a_rank, a_pts, a_roll, a_strk, a_opp_r)
    p1 = model.predict_proba(f1)[0][1]
    
    f2 = get_features(a_rank, a_pts, a_roll, a_strk, a_opp_r, h_rank, h_pts, h_roll, h_strk, h_opp_r)
    p2 = model.predict_proba(f2)[0][1]
    
    avg_p = (p1 + (1 - p2)) / 2
    return avg_p

def simulate_group(group_name, teams, date, model, rankings, results):
    standings = {team: {'points': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'prob_sum': 0} for team in teams}
    
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            t1, t2 = teams[i], teams[j]
            p = predict_match(t1, t2, date, model, rankings, results)
            
            if p > 0.55:
                standings[t1]['points'] += 3
                standings[t1]['wins'] += 1
                standings[t2]['losses'] += 1
            elif p < 0.45:
                standings[t2]['points'] += 3
                standings[t2]['wins'] += 1
                standings[t1]['losses'] += 1
            else:
                standings[t1]['points'] += 1
                standings[t1]['draws'] += 1
                standings[t2]['points'] += 1
                standings[t2]['draws'] += 1
            
            standings[t1]['prob_sum'] += p
            standings[t2]['prob_sum'] += (1 - p)
            
    sorted_teams = sorted(teams, key=lambda x: (standings[x]['points'], standings[x]['prob_sum']), reverse=True)
    return sorted_teams, standings

def run_simulation():
    current_dir = os.getcwd()
    if 'notebooks' in current_dir or 'src' in current_dir:
        base_path = '..'
    else:
        base_path = 'world-cup-predictor'

    # Use advanced model if available
    model_path = f'{base_path}/models/advanced_gb.pkl'
    if not os.path.exists(model_path):
        model_path = f'{base_path}/models/optimized_gb.pkl'
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    rankings = pd.read_csv(f'{base_path}/data/processed/rankings_2026_cycle.csv')
    # Add rank if missing for internal calculation
    if 'rank' not in rankings.columns:
        rankings = rankings.sort_values(['date', 'total_points'], ascending=[True, False])
        rankings['rank'] = rankings.groupby('date')['total_points'].rank(ascending=False, method='min')

    results = pd.read_csv(f'{base_path}/data/processed/results_2026_cycle.csv')
    rankings['date'] = pd.to_datetime(rankings['date'])
    results['date'] = pd.to_datetime(results['date'])
    
    groups = [
        ['Algeria', 'Argentina', 'Austria', 'Jordan'],
        ['Australia', 'Paraguay', 'Turkey', 'United States'],
        ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
        ['Bosnia and Herzegovina', 'Canada', 'Qatar', 'Switzerland'],
        ['Brazil', 'Haiti', 'Morocco', 'Scotland'],
        ['Cape Verde', 'Saudi Arabia', 'Spain', 'Uruguay'],
        ['Colombia', 'DR Congo', 'Portugal', 'Uzbekistan'],
        ['Croatia', 'England', 'Ghana', 'Panama'],
        ['Curaçao', 'Ecuador', 'Germany', 'Ivory Coast'],
        ['Czech Republic', 'Mexico', 'South Africa', 'South Korea'],
        ['France', 'Iraq', 'Norway', 'Senegal'],
        ['Japan', 'Netherlands', 'Sweden', 'Tunisia']
    ]
    
    standardized_groups = []
    for g in groups:
        standardized_groups.append([mapping.get(t, t) for t in g])
    
    wc_start_date = pd.to_datetime('2026-06-11')
    
    group_results = {}
    third_places = []
    
    for i, g in enumerate(standardized_groups):
        g_name = chr(65 + i)
        winners, standings = simulate_group(g_name, g, wc_start_date, model, rankings, results)
        group_results[g_name] = {'winners': winners, 'standings': standings}
        
        third_team = winners[2]
        third_places.append({
            'team': third_team,
            'group': g_name,
            'points': standings[third_team]['points'],
            'prob_sum': standings[third_team]['prob_sum']
        })
        
    best_thirds = sorted(third_places, key=lambda x: (x['points'], x['prob_sum']), reverse=True)[:8]
    best_third_teams = [x['team'] for x in best_thirds]
    
    group_winners = [group_results[g]['winners'][0] for g in 'ABCDEFGHIJKL']
    group_runners_up = [group_results[g]['winners'][1] for g in 'ABCDEFGHIJKL']
    
    bracket = []
    for i in range(8):
        bracket.append((group_winners[i], best_third_teams[i]))
    for i in range(8, 12):
        bracket.append((group_winners[i], group_runners_up[i-8]))
    for i in range(4, 12, 2):
        bracket.append((group_runners_up[i], group_runners_up[i+1]))
        
    def simulate_knockout(matchup, date, model, rankings, results):
        p = predict_match(matchup[0], matchup[1], date, model, rankings, results)
        winner = matchup[0] if p > 0.5 else matchup[1]
        return winner, p if winner == matchup[0] else 1-p

    knockout_data = []
    print("\n--- Round of 32 ---")
    next_round = []
    for m in bracket:
        winner, prob = simulate_knockout(m, wc_start_date + pd.Timedelta(days=20), model, rankings, results)
        next_round.append(winner)
        knockout_data.append({'Round': 'R32', 'Matchup': m, 'Winner': winner, 'Prob': prob})
        print(f"{m[0]} vs {m[1]} -> Winner: {winner} ({prob:.2f})")

    rounds = ["Round of 16", "Quarter-finals", "Semi-finals", "Final"]
    current_teams = next_round
    
    for round_name in rounds:
        print(f"\n--- {round_name} ---")
        winners = []
        for i in range(0, len(current_teams), 2):
            m = (current_teams[i], current_teams[i+1])
            winner, prob = simulate_knockout(m, wc_start_date + pd.Timedelta(days=30), model, rankings, results)
            winners.append(winner)
            knockout_data.append({'Round': round_name, 'Matchup': m, 'Winner': winner, 'Prob': prob})
            print(f"{m[0]} vs {m[1]} -> Winner: {winner} ({prob:.2f})")
        current_teams = winners
        if len(current_teams) == 1:
            print(f"\nWORLD CUP WINNER: {current_teams[0]}")
            break

    return group_results, best_third_teams, knockout_data

if __name__ == "__main__":
    run_simulation()
