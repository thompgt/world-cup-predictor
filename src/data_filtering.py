import pandas as pd
import os

def filter_data():
    results_path = 'world-cup-predictor/data/results.csv'
    rankings_path = 'world-cup-predictor/data/fifa_ranking.csv'
    
    # Load data
    results = pd.read_csv(results_path)
    rankings = pd.read_csv(rankings_path)
    
    # Convert dates
    results['date'] = pd.to_datetime(results['date'])
    rankings['date'] = pd.to_datetime(rankings['date'])
    
    # Calculate Rank if missing
    if 'rank' not in rankings.columns:
        print("Calculating rank from total_points...")
        rankings = rankings.sort_values(['date', 'total_points'], ascending=[True, False])
        rankings['rank'] = rankings.groupby('date')['total_points'].rank(ascending=False, method='min')
    
    # Define 2026 Cycle start (Day after 2022 World Cup Final)
    cycle_start = pd.to_datetime('2022-12-19')
    
    # Filter results for the 2026 cycle
    # Note: We keep some history for rolling averages, but define the 'cycle' matches here
    results_2026_cycle = results[results['date'] >= cycle_start].copy()
    
    # Filter rankings (last few years is enough, but let's keep it consistent)
    rankings_2026_cycle = rankings[rankings['date'] >= cycle_start].copy()
    
    # Save filtered data
    os.makedirs('world-cup-predictor/data/processed', exist_ok=True)
    results_2026_cycle.to_csv('world-cup-predictor/data/processed/results_2026_cycle.csv', index=False)
    rankings_2026_cycle.to_csv('world-cup-predictor/data/processed/rankings_2026_cycle.csv', index=False)
    
    print(f"Filtered results: {len(results_2026_cycle)} matches since {cycle_start.date()}")
    print(f"Filtered rankings: {len(rankings_2026_cycle)} updates since {cycle_start.date()}")

if __name__ == "__main__":
    filter_data()
