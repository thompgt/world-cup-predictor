import pandas as pd
import os

def standardize_names():
    results_path = 'world-cup-predictor/data/processed/results_2026_cycle.csv'
    rankings_path = 'world-cup-predictor/data/processed/rankings_2026_cycle.csv'
    
    results = pd.read_csv(results_path)
    rankings = pd.read_csv(rankings_path)
    
    # Mapping for Results teams -> Rankings names (or vice versa, let's pick a standard)
    # The 2022 methodology usually mapped Rankings to Results names or vice versa.
    # Let's map everything to match the Rankings dataset as it's the source of features.
    
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
        'Chinese Taipei': 'Taiwan', # Wait, check which one is which
        'Taiwan': 'Chinese Taipei',
        'Kyrgyzstan': 'Kyrgyz Republic',
        'Saint Kitts and Nevis': 'St Kitts and Nevis',
        'Saint Lucia': 'St Lucia',
        'Saint Vincent and the Grenadines': 'St Vincent and the Grenadines',
    }
    
    results['home_team'] = results['home_team'].replace(mapping)
    results['away_team'] = results['away_team'].replace(mapping)
    
    # Remove "(unranked)" suffix in rankings if it exists
    rankings['team'] = rankings['team'].str.replace(r' \(unranked\)', '', regex=True)
    
    # Save back
    results.to_csv(results_path, index=False)
    rankings.to_csv(rankings_path, index=False)
    
    print("Standardization complete.")

if __name__ == "__main__":
    standardize_names()
