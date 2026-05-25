# 🏆 2026 FIFA World Cup Predictor

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced machine learning pipeline to predict the **2026 FIFA World Cup** outcomes, adapting the proven **Gradient Boosting** methodology from the 2022 Kaggle tournament for the newly expanded **48-team format**.

---

## 📖 Overview

The 2026 World Cup introduces a massive expansion from 32 to 48 teams, featuring 104 matches and a new Round of 32 knockout stage. This project replicates the "Direct Translation" approach:
- **Core Model:** Gradient Boosting Classifier.
- **Data Source:** FIFA World Rankings and International Match Results (2022-2026 cycle).
- **Simulation Engine:** Custom-built to handle 12 groups of 4 and the complex selection of the **8 best 3rd-place teams**.

## 🚀 Key Features

- **2026 Cycle Data:** Trained exclusively on matches played after the 2022 World Cup Final.
- **Granular Feature Engineering:**
    - `rank_diff`: FIFA rank difference between opponents.
    - `goals_rolling_diff`: Difference in average goals scored over the last 5 matches.
    - `average_rank`: Combined quality of the matchup.
    - `point_diff`: FIFA points difference.
- **Robust Simulation:** Predicts every stage from Group A-L through to the Final at MetLife Stadium.
- **Visual Analysis:** Comprehensive Jupyter Notebook with EDA, group standings, and knockout bracket visualizations.

## 📂 Project Structure

```text
world-cup-predictor/
├── data/               # Raw and processed datasets
│   ├── processed/      # Cleaned 2026 cycle data
│   └── features/       # Engineered feature sets
├── models/             # Saved Scikit-Learn model pickles
├── notebooks/          # End-to-end analysis & visualizations
├── src/                # Modular Python scripts
│   ├── data_filtering.py
│   ├── feature_rolling.py
│   ├── train_model.py
│   └── simulation_engine.py  # The core 2026 simulation logic
└── README.md
```

## 🛠️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/thompgt/world-cup-predictor.git
cd world-cup-predictor
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the Simulation
```bash
python src/simulation_engine.py
```

## 📊 Final Prediction: 2026 Champion

Based on our final optimized model (Accuracy: **69.08%**), the predicted winner of the 2026 FIFA World Cup is:

### 🇦🇷 **ARGENTINA**

**Path to Victory:**
- **Group Stage:** Topped Group A (Argentina, Austria, Algeria, Jordan).
- **Round of 32:** Defeated Côte d'Ivoire.
- **Round of 16:** Defeated Australia.
- **Quarter-Final:** Defeated Belgium.
- **Semi-Final:** Defeated Spain.
- **Final:** Defeated France in a historic rematch of the 2022 final.

## 📈 Methodology & Adaptation

This project adapts the work of **sslp23** (Kaggle 2022). The primary challenge was the **Simulation Engine**, which had to be rewritten to support:
1. **12 Groups:** Handling 72 group-stage matches.
2. **Best 3rds Selection:** Implementing the tie-breaker logic (Points > Prob_Sum) to select the 8 best 3rd-place teams.
3. **Round of 32:** Expanding the knockout logic from 4 rounds to 5 rounds.

---

*Disclaimer: This is a statistical model based on historical data and does not account for real-world variables like injuries, tactics, or home-crowd influence beyond what is captured in FIFA rankings.*
