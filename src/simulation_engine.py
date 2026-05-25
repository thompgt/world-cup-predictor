import pandas as pd
import numpy as np
import pickle

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
    
    # Get rolling goals
    team_results = results[((results['home_team'] == team) | (results['away_team'] == team)) & (results['date'] < date)]
    if team_results.empty:
        rolling_goals = 0
    else:
        last_5 = team_results.tail(5)
        goals = []
        for _, r in last_5.iterrows():
            if r['home_team'] == team:
                goals.append(r['home_score'])
            else:
                goals.append(r['away_score'])
        rolling_goals = np.mean(goals)
        
    return rank, pts, rolling_goals

def predict_match(home, away, date, model, rankings, results):
    h_rank, h_pts, h_roll = get_current_features(home, date, rankings, results)
    a_rank, a_pts, a_roll = get_current_features(away, date, rankings, results)
    
    # Feature engineering for the match
    # features = ['rank_diff', 'average_rank', 'point_diff', 'is_friendly', 'goals_rolling_diff']
    
    def get_features(h_r, h_p, h_rl, a_r, a_p, a_rl):
        return pd.DataFrame([{
            'rank_diff': h_r - a_r,
            'average_rank': (h_r + a_r) / 2,
            'point_diff': h_p - a_p,
            'is_friendly': 0, # WC matches are not friendly
            'goals_rolling_diff': h_rl - a_rl
        }])

    # The 2022 methodology predicts Home vs Away AND Away vs Home, then takes the average
    f1 = get_features(h_rank, h_pts, h_roll, a_rank, a_pts, a_roll)
    p1 = model.predict_proba(f1)[0][1] # Probability of Home Win
    
    f2 = get_features(a_rank, a_pts, a_roll, h_rank, h_pts, h_roll)
    p2 = model.predict_proba(f2)[0][1] # Probability of Away Win (if we swap)
    
    # Mean probability of Home winning
    avg_p = (p1 + (1 - p2)) / 2
    return avg_p

def simulate_group(group_name, teams, date, model, rankings, results):
    standings = {team: {'points': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'prob_sum': 0} for team in teams}
    
    matches = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            t1, t2 = teams[i], teams[j]
            p = predict_match(t1, t2, date, model, rankings, results)
            
            # Simple outcome based on probability
            if p > 0.55: # Win
                standings[t1]['points'] += 3
                standings[t1]['wins'] += 1
                standings[t2]['losses'] += 1
            elif p < 0.45: # Loss
                standings[t2]['points'] += 3
                standings[t2]['wins'] += 1
                standings[t1]['losses'] += 1
            else: # Draw
                standings[t1]['points'] += 1
                standings[t1]['draws'] += 1
                standings[t2]['points'] += 1
                standings[t2]['draws'] += 1
            
            standings[t1]['prob_sum'] += p
            standings[t2]['prob_sum'] += (1 - p)
            
    # Sort teams: Points -> prob_sum (as tie-breaker)
    sorted_teams = sorted(teams, key=lambda x: (standings[x]['points'], standings[x]['prob_sum']), reverse=True)
    return sorted_teams, standings

def run_simulation():
    with open('world-cup-predictor/models/optimized_gb.pkl', 'rb') as f:
        model = pickle.load(f)
    
    rankings = pd.read_csv('world-cup-predictor/data/processed/rankings_2026_cycle.csv')
    results = pd.read_csv('world-cup-predictor/data/processed/results_2026_cycle.csv')
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
    
    # Standardize team names in groups
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
        print(f"Group {g_name} Finished: 1st: {winners[0]}, 2nd: {winners[1]}, 3rd: {winners[2]}")
        
        # Track 3rd place for best-8 selection
        third_team = winners[2]
        third_places.append({
            'team': third_team,
            'group': g_name,
            'points': standings[third_team]['points'],
            'prob_sum': standings[third_team]['prob_sum']
        })
        
    # Select 8 best 3rd places
    best_thirds = sorted(third_places, key=lambda x: (x['points'], x['prob_sum']), reverse=True)[:8]
    best_third_teams = [x['team'] for x in best_thirds]
    
    print(f"\nBest 8 3rd-place teams: {', '.join(best_third_teams)}")
    
    return group_results, best_third_teams

if __name__ == "__main__":
    run_simulation()
