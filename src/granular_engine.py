import pandas as pd
import numpy as np
import random
import pickle
import os

# Updated Rosters for 2026 (Focus on Active Players & Expected Starters)
rosters = {
    "Argentina": [("L. Messi", 0.3), ("L. Martínez", 0.25), ("J. Álvarez", 0.2), ("A. Mac Allister", 0.1), ("E. Fernández", 0.1), ("N. González", 0.05)],
    "France": [("K. Mbappé", 0.4), ("M. Thuram", 0.2), ("O. Dembélé", 0.15), ("R. Kolo Muani", 0.1), ("C. Nkunku", 0.1), ("B. Barcola", 0.05)],
    "England": [("H. Kane", 0.35), ("J. Bellingham", 0.2), ("B. Saka", 0.15), ("P. Foden", 0.15), ("O. Watkins", 0.1), ("C. Palmer", 0.05)],
    "Brazil": [("Vinícius Jr.", 0.3), ("Rodrygo", 0.25), ("Raphinha", 0.15), ("Endrick", 0.15), ("Gabriel Martinelli", 0.1), ("Bruno Guimarães", 0.05)],
    "Spain": [("L. Yamal", 0.25), ("Alvaro Morata", 0.25), ("Nico Williams", 0.2), ("Dani Olmo", 0.15), ("Pedri", 0.1), ("Ferran Torres", 0.05)],
    "Portugal": [("C. Ronaldo", 0.3), ("Rafael Leão", 0.2), ("Bruno Fernandes", 0.15), ("Bernardo Silva", 0.15), ("Diogo Jota", 0.1), ("Gonçalo Ramos", 0.1)],
    "Netherlands": [("C. Gakpo", 0.25), ("Xavi Simons", 0.2), ("Memphis Depay", 0.2), ("D. Malen", 0.15), ("T. Koopmeiners", 0.1), ("V. van Dijk", 0.1)],
    "Germany": [("J. Musiala", 0.25), ("F. Wirtz", 0.25), ("K. Havertz", 0.2), ("N. Füllkrug", 0.15), ("L. Sané", 0.1), ("S. Gnabry", 0.05)],
    "Belgium": [("R. Lukaku", 0.35), ("K. De Bruyne", 0.2), ("J. Doku", 0.15), ("L. Trossard", 0.15), ("Loïs Openda", 0.1), ("C. De Ketelaere", 0.05)],
    "Colombia": [("Luis Díaz", 0.35), ("J. Rodríguez", 0.25), ("J. Córdoba", 0.15), ("R. Borré", 0.1), ("J. Arias", 0.1), ("D. Muñoz", 0.05)],
    "USA": [("C. Pulisic", 0.3), ("F. Balogun", 0.25), ("T. Weah", 0.15), ("R. Pepi", 0.15), ("W. McKennie", 0.1), ("G. Reyna", 0.05)],
    "Mexico": [("S. Giménez", 0.35), ("H. Martín", 0.25), ("U. Antuna", 0.15), ("J. Quiñones", 0.15), ("E. Álvarez", 0.1)],
    "Canada": [("J. David", 0.4), ("C. Larin", 0.3), ("A. Davies", 0.15), ("I. Kone", 0.1), ("T. Buchanan", 0.05)],
    "Morocco": [("A. El Kaabi", 0.3), ("Y. En-Nesyri", 0.3), ("H. Ziyech", 0.15), ("B. Diaz", 0.15), ("A. Hakimi", 0.1)],
    "Uruguay": [("D. Núñez", 0.4), ("F. Pellistri", 0.2), ("F. Valverde", 0.2), ("N. de la Cruz", 0.1), ("M. Araújo", 0.1)],
    "Japan": [("A. Ueda", 0.3), ("K. Mitoma", 0.25), ("T. Kubo", 0.2), ("R. Doan", 0.15), ("W. Endo", 0.1)],
    "South Korea": [("Son Heung-min", 0.4), ("Hwang Hee-chan", 0.3), ("Lee Kang-in", 0.2), ("Cho Gue-sung", 0.1)],
}

def get_scorers_list(team, num_goals):
    if num_goals == 0: return []
    players_raw = rosters.get(team, [("Squad Member", 0.4), ("Attacker", 0.3), ("Midfielder", 0.2), ("Defender", 0.1)])
    names = [p[0] for p in players_raw]
    weights = [p[1] for p in players_raw]
    return random.choices(names, weights=weights, k=num_goals)

def simulate_match_detail(home, away, model, rankings, results, squad_dict, date, is_ko=False):
    import pandas as pd
    from simulation_engine import predict_match
    
    p = predict_match(home, away, date, model, rankings, results, squad_dict)
    
    # Lambda based on historical WC averages (~2.6 per game)
    l_home = 1.35 * (p / 0.5)
    l_away = 1.35 * ((1-p) / 0.5)
    
    h_goals = np.random.poisson(l_home)
    a_goals = np.random.poisson(l_away)
    
    # Tie-breaking for knockout
    if is_ko and h_goals == a_goals:
        if p > 0.5: h_goals += 1
        else: a_goals += 1
        
    h_scorers = get_scorers_list(home, h_goals)
    a_scorers = get_scorers_list(away, a_goals)
    
    return {
        "Match": f"{home} vs {away}",
        "Home": home,
        "Away": away,
        "Win Prob": round(p, 2),
        "Home Goals": h_goals,
        "Away Goals": a_goals,
        "Score": f"{h_goals} - {a_goals}",
        "Home Scorers": h_scorers,
        "Away Scorers": a_scorers,
        "Winner": home if h_goals > a_goals else away,
        "Clean Sheet Home": 1 if a_goals == 0 else 0,
        "Clean Sheet Away": 1 if h_goals == 0 else 0
    }
