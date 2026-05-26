import pandas as pd
import numpy as np
import pickle
import os
import collections
import random

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


def load_rosters(base_path):
    rosters_path = f"{base_path}/data/fifa_players/rosters.csv"
    roster_map = {}
    if os.path.exists(rosters_path):
        df = pd.read_csv(rosters_path)
        # expected columns: team,player,position,weight (optional)
        for team, g in df.groupby('team'):
            players = []
            for _, r in g.iterrows():
                weight = r.get('weight', 1.0) if 'weight' in r.index else 1.0
                players.append({'player': r['player'], 'position': r.get('position', ''), 'weight': float(weight)})
            roster_map[team] = players
    return roster_map


def build_historical_scorers_map(base_path):
    path = f"{base_path}/data/goalscorers.csv"
    if not os.path.exists(path):
        return {}
    gdf = pd.read_csv(path)
    # use recent history more by weighting later dates slightly higher
    if 'date' in gdf.columns:
        try:
            gdf['date'] = pd.to_datetime(gdf['date'])
            gdf = gdf.sort_values('date')
            gdf['time_weight'] = np.linspace(0.5, 1.5, len(gdf))
        except Exception:
            gdf['time_weight'] = 1.0
    else:
        gdf['time_weight'] = 1.0

    scorer_map = {}
    for team, g in gdf.groupby('team'):
        counts = {}
        for _, r in g.iterrows():
            name = r['scorer']
            w = r.get('time_weight', 1.0)
            counts[name] = counts.get(name, 0) + w
        scorer_map[team] = counts
    return scorer_map


def select_scorers(team, n_goals, roster_map, hist_scorer_map):
    choices = []
    if team in roster_map and roster_map[team]:
        players = roster_map[team]
        weights = [p.get('weight', 1.0) for p in players]
        names = [p['player'] for p in players]
        # sample with replacement
        choices = random.choices(names, weights=weights, k=n_goals)
    elif team in hist_scorer_map and hist_scorer_map[team]:
        names = list(hist_scorer_map[team].keys())
        weights = list(hist_scorer_map[team].values())
        choices = random.choices(names, weights=weights, k=n_goals)
    else:
        # fallback synthetic attackers
        choices = [f"Attacker_{i+1}_{team}" for i in range(n_goals)]
    return choices

def simulate_group(teams, date, model, rankings, results, squad_dict, roster_map=None, hist_scorer_map=None, scorer_counts=None, match_events=None):
    if roster_map is None: roster_map = {}
    if hist_scorer_map is None: hist_scorer_map = {}
    if scorer_counts is None: scorer_counts = collections.defaultdict(int)
    if match_events is None: match_events = []

    standings = {team: {'points': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'prob_sum': 0} for team in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            t1, t2 = teams[i], teams[j]
            p = predict_match(t1, t2, date, model, rankings, results, squad_dict)

            # use squad ratings to influence expected goals
            _, h_sr, _ = get_current_features(t1, date, rankings, results, squad_dict)
            _, a_sr, _ = get_current_features(t2, date, rankings, results, squad_dict)
            base_lambda = 1.1
            h_lambda = max(0.05, base_lambda + (h_sr - a_sr) / 10.0 + (p - 0.5))
            a_lambda = max(0.05, base_lambda + (a_sr - h_sr) / 10.0 + ((1 - p) - 0.5))

            h_goals = int(np.random.poisson(h_lambda))
            a_goals = int(np.random.poisson(a_lambda))

            # assign scorers
            h_scorers = select_scorers(t1, h_goals, roster_map, hist_scorer_map) if h_goals > 0 else []
            a_scorers = select_scorers(t2, a_goals, roster_map, hist_scorer_map) if a_goals > 0 else []
            for s in h_scorers: scorer_counts[s] += 1
            for s in a_scorers: scorer_counts[s] += 1

            # determine points from goals
            if h_goals > a_goals:
                standings[t1]['points'] += 3; standings[t1]['wins'] += 1; standings[t2]['losses'] += 1
            elif a_goals > h_goals:
                standings[t2]['points'] += 3; standings[t2]['wins'] += 1; standings[t1]['losses'] += 1
            else:
                standings[t1]['points'] += 1; standings[t1]['draws'] += 1
                standings[t2]['points'] += 1; standings[t2]['draws'] += 1

            standings[t1]['prob_sum'] += p
            standings[t2]['prob_sum'] += (1 - p)

            match_events.append({'date': date, 'home': t1, 'away': t2, 'home_goals': h_goals, 'away_goals': a_goals,
                                 'home_scorers': h_scorers, 'away_scorers': a_scorers, 'prob_home_win': p})

    sorted_teams = sorted(teams, key=lambda x: (standings[x]['points'], standings[x]['prob_sum']), reverse=True)
    return sorted_teams, standings, scorer_counts, match_events

def run_simulation():
    # Resolve repository root reliably based on this file's location
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base_path = repo_root

    model_path = os.path.join(base_path, 'models', 'high_signal_gb.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_path, 'models', 'weighted_gb.pkl')

    model = None
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception:
            model = None

    # If model couldn't be loaded, attempt to train a high-signal model if training code exists
    if model is None:
        try:
            # Import training routine dynamically to avoid heavy imports at module load
            try:
                from train_high_signal import train_high_signal_model
            except Exception:
                from src.train_high_signal import train_high_signal_model
            print('Model not found or failed to load; attempting to train high-signal model...')
            train_high_signal_model()
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
        except Exception as e:
            # Provide clear guidance instead of a random/dummy model
            raise RuntimeError(f"Model missing and automatic training failed: {e}.\nPlease run 'python src/train_high_signal.py' to train and save the model to models/high_signal_gb.pkl")
    
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
    
    # load rosters and historical scorers
    roster_map = load_rosters(base_path)
    hist_scorer_map = build_historical_scorers_map(base_path)
    scorer_counts = collections.defaultdict(int)
    match_events = []

    for i, g in enumerate(groups):
        g_name = chr(65 + i)
        winners, standings, scorer_counts, match_events = simulate_group(g, wc_start_date, model, rankings, results, squad_dict,
                                                                          roster_map=roster_map, hist_scorer_map=hist_scorer_map,
                                                                          scorer_counts=scorer_counts, match_events=match_events)
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

    def simulate_ko(matchup, date, model, rankings, results, squad_dict, roster_map=None, hist_scorer_map=None, scorer_counts=None, match_events=None):
        if roster_map is None: roster_map = {}
        if hist_scorer_map is None: hist_scorer_map = {}
        if scorer_counts is None: scorer_counts = collections.defaultdict(int)
        if match_events is None: match_events = []

        p = predict_match(matchup[0], matchup[1], date, model, rankings, results, squad_dict)

        # expected goals similar to group
        _, h_sr, _ = get_current_features(matchup[0], date, rankings, results, squad_dict)
        _, a_sr, _ = get_current_features(matchup[1], date, rankings, results, squad_dict)
        base_lambda = 1.1
        h_lambda = max(0.05, base_lambda + (h_sr - a_sr) / 10.0 + (p - 0.5))
        a_lambda = max(0.05, base_lambda + (a_sr - h_sr) / 10.0 + ((1 - p) - 0.5))

        h_goals = int(np.random.poisson(h_lambda))
        a_goals = int(np.random.poisson(a_lambda))

        h_scorers = select_scorers(matchup[0], h_goals, roster_map, hist_scorer_map) if h_goals > 0 else []
        a_scorers = select_scorers(matchup[1], a_goals, roster_map, hist_scorer_map) if a_goals > 0 else []
        for s in h_scorers: scorer_counts[s] += 1
        for s in a_scorers: scorer_counts[s] += 1

        # decide winner (if tie, use probability to break)
        if h_goals > a_goals:
            winner = matchup[0]; win_prob = p
        elif a_goals > h_goals:
            winner = matchup[1]; win_prob = 1-p
        else:
            # tie -> decide by p (or coin flip if exactly 0.5)
            if p > 0.5:
                winner = matchup[0]; win_prob = p
            elif p < 0.5:
                winner = matchup[1]; win_prob = 1-p
            else:
                winner = random.choice(list(matchup)); win_prob = 0.5

        match_events.append({'date': date, 'home': matchup[0], 'away': matchup[1], 'home_goals': h_goals, 'away_goals': a_goals,
                             'home_scorers': h_scorers, 'away_scorers': a_scorers, 'prob_home_win': p})

        return winner, win_prob, scorer_counts, match_events

    knockout_data = []
    next_round = []
    for m in bracket:
        winner, prob, scorer_counts, match_events = simulate_ko(m, wc_start_date + pd.Timedelta(days=20), model, rankings, results, squad_dict,
                                                                 roster_map=roster_map, hist_scorer_map=hist_scorer_map,
                                                                 scorer_counts=scorer_counts, match_events=match_events)
        next_round.append(winner)
        # include scorers in KO records
        last_event = match_events[-1] if match_events else {}
        knockout_data.append({'Round': 'R32', 'Matchup': m, 'Winner': winner, 'Prob': prob,
                              'Home_Scorers': last_event.get('home_scorers', []), 'Away_Scorers': last_event.get('away_scorers', [])})

    rounds = ["Round of 16", "Quarter-finals", "Semi-finals", "Final"]
    curr = next_round
    for round_name in rounds:
        winners = []
        for i in range(0, len(curr), 2):
            m = (curr[i], curr[i+1])
            winner, prob, scorer_counts, match_events = simulate_ko(m, wc_start_date + pd.Timedelta(days=30), model, rankings, results, squad_dict,
                                                                     roster_map=roster_map, hist_scorer_map=hist_scorer_map,
                                                                     scorer_counts=scorer_counts, match_events=match_events)
            winners.append(winner)
            last_event = match_events[-1] if match_events else {}
            knockout_data.append({'Round': round_name, 'Matchup': m, 'Winner': winner, 'Prob': prob,
                                  'Home_Scorers': last_event.get('home_scorers', []), 'Away_Scorers': last_event.get('away_scorers', [])})
        curr = winners
        if len(curr) == 1: break

    # write projected scorers summary
    try:
        assets_dir = f"{base_path}/assets"
        os.makedirs(assets_dir, exist_ok=True)
        scorers_df = pd.DataFrame(sorted(scorer_counts.items(), key=lambda x: x[1], reverse=True), columns=['scorer', 'projected_goals'])
        scorers_df.to_csv(f"{assets_dir}/projected_scorers.csv", index=False)
        # detailed match events
        me_df = pd.DataFrame(match_events)
        me_df.to_json(f"{assets_dir}/match_scorers.json", orient='records', date_format='iso')
    except Exception:
        pass

    return group_results, best_third_teams, knockout_data

if __name__ == "__main__":
    run_simulation()
