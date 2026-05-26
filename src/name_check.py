import pandas as pd

def check_names():
    results = pd.read_csv('world-cup-predictor/data/processed/results_2026_cycle.csv')
    rankings = pd.read_csv('world-cup-predictor/data/processed/rankings_2026_cycle.csv')
    
    res_teams = set(results['home_team'].unique()) | set(results['away_team'].unique())
    rank_teams = set(rankings['team'].unique())
    
    mismatches = res_teams - rank_teams
    print(f"Teams in results but not in rankings ({len(mismatches)}):")
    print(sorted(list(mismatches))[:20]) # Show first 20
    
    mismatches_inv = rank_teams - res_teams
    print(f"\nTeams in rankings but not in results ({len(mismatches_inv)}):")
    print(sorted(list(mismatches_inv))[:20])

if __name__ == "__main__":
    check_names()
