import pandas as pd
import numpy as np
import pickle
import os

# Standardization mapping
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

def get_current_features(team, date, rankings, results, squad_dict):
    # 1. Inverted Rank (212 - rank)
    team_ranks = rankings[rankings['team'] == team]
    past_ranks = team_ranks[team_ranks['date'] < date]
    rank = 100
    if not past_ranks.empty:
        rank = past_ranks.iloc[-1]['rank']
    inv_rank = 212 - rank
    
    # 2. Squad Rating (EA FC 26)
    squad_rating = squad_dict.get(team, 70)
    
    # 3. Recent Form (Rolling 5 matches avg points)
    team_results = results[((results['home_team'] == team) | (results['away_team'] == team)) & (results['date'] < date)]
    if team_results.empty:
        form = 1.0
    else:
        last_5 = team_results.tail(5)
        pts_list = []
        for _, r in last_5.iterrows():
            if r['home_team'] == team:
                if r['home_score'] > r['away_score']: pts_list.append(3)
                elif r['home_score'] == r['away_score']: pts_list.append(1)
                else: pts_list.append(0)
            else:
                if r['away_score'] > r['home_score']: pts_list.append(3)
                elif r['away_score'] == r['home_score']: pts_list.append(1)
                else: pts_list.append(0)
        form = np.mean(pts_list)
        
    return inv_rank, squad_rating, form

def predict_match(home, away, date, model, rankings, results, squad_dict):
    h_ir, h_sr, h_f = get_current_features(home, date, rankings, results, squad_dict)
    a_ir, a_sr, a_f = get_current_features(away, date, rankings, results, squad_dict)
    
    def get_features(h_ir, h_sr, h_f, a_ir, a_sr, a_f):
        df = pd.DataFrame([{
            'rank_diff_inv': h_ir - a_ir,
            'rating_diff': h_sr - a_sr,
            'form_diff': h_f - a_f,
            'h_inv_rank': h_ir,
            'h_squad_rating': h_sr,
            'h_recent_form': h_f,
            'squad_rank_interaction': h_sr * h_ir,
            'rating_delta_form': (h_sr - a_sr) * h_f
        }])
        return df.fillna(0)

    # Bi-directional prediction
    f1 = get_features(h_ir, h_sr, h_f, a_ir, a_sr, a_f)
    p1 = model.predict_proba(f1)[0][1]
    
    f2 = get_features(a_ir, a_sr, a_f, h_ir, h_sr, h_f)
    p2 = model.predict_proba(f2)[0][1]
    
    avg_p = (p1 + (1 - p2)) / 2
    
    # Apply Squad Superiority Bias (User Request)
    # Every 1 point of rating delta adds 3% to the win probability (more weighted)
    squad_bias = (h_sr - a_sr) * 0.03
    avg_p = np.clip(avg_p + squad_bias, 0.01, 0.99)
    
    return avg_p

def simulate_group(teams, date, model, rankings, results, squad_dict):
    standings = {team: {'points': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'prob_sum': 0} for team in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            t1, t2 = teams[i], teams[j]
            p = predict_match(t1, t2, date, model, rankings, results, squad_dict)
            if p > 0.55:
                standings[t1]['points'] += 3; standings[t1]['wins'] += 1; standings[t2]['losses'] += 1
            elif p < 0.45:
                standings[t2]['points'] += 3; standings[t2]['wins'] += 1; standings[t1]['losses'] += 1
            else:
                standings[t1]['points'] += 1; standings[t1]['draws'] += 1
                standings[t2]['points'] += 1; standings[t2]['draws'] += 1
            standings[t1]['prob_sum'] += p
            standings[t2]['prob_sum'] += (1 - p)
    sorted_teams = sorted(teams, key=lambda x: (standings[x]['points'], standings[x]['prob_sum']), reverse=True)
    return sorted_teams, standings

def run_simulation():
    current_dir = os.getcwd()
    base_path = '..' if 'notebooks' in current_dir or 'src' in current_dir else 'world-cup-predictor'

    model_path = f'{base_path}/models/high_signal_gb.pkl'
    if not os.path.exists(model_path): model_path = f'{base_path}/models/weighted_gb.pkl'
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load LIVE rankings
    rankings = pd.read_csv(f'{base_path}/data/fifa_ranking_live.csv')
    results = pd.read_csv(f'{base_path}/data/processed/results_2026_cycle.csv')
    squad_ratings = pd.read_csv(f'{base_path}/data/processed/squad_ratings.csv')
    squad_dict = dict(zip(squad_ratings['team'], squad_ratings['squad_rating']))

    results['date'] = pd.to_datetime(results['date'])
    
    # Standardize column name in live rankings to 'total_points' for compatibility if needed, 
    # but the engine currently uses 'rank' and 'total_points'. 
    # In live csv it is 'rank' and 'points'.
    if 'total_points' not in rankings.columns:
        rankings = rankings.rename(columns={'points': 'total_points'})
    
    # Ensure rank is present
    if 'rank' not in rankings.columns:
        rankings = rankings.sort_values(['total_points'], ascending=False)
        rankings['rank'] = rankings['total_points'].rank(ascending=False, method='min')

    # Since it is a live snapshot, we treat all entries as 'latest'
    rankings['date'] = pd.to_datetime('2026-05-25') 

    
    groups = [
        ['Algeria', 'Argentina', 'Austria', 'Jordan'],
        ['Australia', 'Paraguay', 'Turkey', 'United States'],
        ['Belgium', 'Egypt', 'IR Iran', 'New Zealand'],
        ['Bosnia and Herzegovina', 'Canada', 'Qatar', 'Switzerland'],
        ['Brazil', 'Haiti', 'Morocco', 'Scotland'],
        ['Cabo Verde', 'Saudi Arabia', 'Spain', 'Uruguay'],
        ['Colombia', 'Congo DR', 'Portugal', 'Uzbekistan'],
        ['Croatia', 'England', 'Ghana', 'Panama'],
        ['Curaçao', 'Ecuador', 'Germany', "Côte d'Ivoire"],
        ['Czechia', 'Mexico', 'South Africa', 'Korea Republic'],
        ['France', 'Iraq', 'Norway', 'Senegal'],
        ['Japan', 'Netherlands', 'Sweden', 'Tunisia']
    ]
    
    wc_start_date = pd.to_datetime('2026-06-11')
    group_results = {}
    third_places = []
    
    for i, g in enumerate(groups):
        g_name = chr(65 + i)
        winners, standings = simulate_group(g, wc_start_date, model, rankings, results, squad_dict)
        group_results[g_name] = {'winners': winners, 'standings': standings}
        third_team = winners[2]
        third_places.append({'team': third_team, 'group': g_name, 'points': standings[third_team]['points'], 'prob_sum': standings[third_team]['prob_sum']})
        
    best_thirds = sorted(third_places, key=lambda x: (x['points'], x['prob_sum']), reverse=True)[:8]
    best_third_teams = [x['team'] for x in best_thirds]
    
    group_winners = [group_results[g]['winners'][0] for g in 'ABCDEFGHIJKL']
    group_runners_up = [group_results[g]['winners'][1] for g in 'ABCDEFGHIJKL']
    
    bracket = []
    for i in range(8): bracket.append((group_winners[i], best_third_teams[i]))
    for i in range(8, 12): bracket.append((group_winners[i], group_runners_up[i-8]))
    for i in range(4, 12, 2): bracket.append((group_runners_up[i], group_runners_up[i+1]))
        
    def simulate_ko(matchup, date, model, rankings, results, squad_dict):
        p = predict_match(matchup[0], matchup[1], date, model, rankings, results, squad_dict)
        winner = matchup[0] if p > 0.5 else matchup[1]
        return winner, p if winner == matchup[0] else 1-p

    knockout_data = []
    next_round = []
    for m in bracket:
        winner, prob = simulate_ko(m, wc_start_date + pd.Timedelta(days=20), model, rankings, results, squad_dict)
        next_round.append(winner)
        knockout_data.append({'Round': 'R32', 'Matchup': m, 'Winner': winner, 'Prob': prob})

    rounds = ["Round of 16", "Quarter-finals", "Semi-finals", "Final"]
    curr = next_round
    for round_name in rounds:
        winners = []
        for i in range(0, len(curr), 2):
            m = (curr[i], curr[i+1])
            winner, prob = simulate_ko(m, wc_start_date + pd.Timedelta(days=30), model, rankings, results, squad_dict)
            winners.append(winner)
            knockout_data.append({'Round': round_name, 'Matchup': m, 'Winner': winner, 'Prob': prob})
        curr = winners
        if len(curr) == 1: break

    return group_results, best_third_teams, knockout_data

if __name__ == "__main__":
    run_simulation()
