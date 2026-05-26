import pandas as pd

def extract_groups():
    results = pd.read_csv('world-cup-predictor/data/results.csv')
    wc_2026 = results[(results['date'] >= '2026-06-11') & (results['tournament'] == 'FIFA World Cup')].copy()
    
    # We'll assume the first 72 matches are the group stage
    # 12 groups * 6 matches/group = 72 matches
    group_stage_matches = wc_2026.head(72).copy()
    
    teams = set(group_stage_matches['home_team'].unique()) | set(group_stage_matches['away_team'].unique())
    print(f"Total teams in group stage: {len(teams)}")
    
    # To find groups, we can look at which teams play each other
    # A group of 4 teams plays each other.
    groups = []
    processed_teams = set()
    
    for team in sorted(list(teams)):
        if team in processed_teams:
            continue
        
        # Find all opponents of this team in the group stage
        opponents = set(group_stage_matches[group_stage_matches['home_team'] == team]['away_team']) | \
                    set(group_stage_matches[group_stage_matches['away_team'] == team]['home_team'])
        
        group = {team} | opponents
        if len(group) == 4:
            groups.append(sorted(list(group)))
            processed_teams.update(group)
        else:
            print(f"Warning: Team {team} has {len(opponents)} opponents: {opponents}")

    print(f"Found {len(groups)} groups.")
    for i, g in enumerate(groups):
        print(f"Group {chr(65+i)}: {', '.join(g)}")
        
    return groups

if __name__ == "__main__":
    extract_groups()
