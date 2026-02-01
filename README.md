# Player Value Analysis & Optimal Lineup Selection

## Overview
This project analyzes player performance data to determine optimal team lineups using machine learning and SHAP values. The analysis trains team-specific models, extracts player values, and solves a constrained optimization problem to build the best lineups.

### Python Version
- Python 3.8 or higher recommended

## Data Files
- `stint_data.csv` - Game stint data with lineups and goal differentials
- `player_data.csv` - Player ratings and metadata

## Approach

### 1. Train Team-Specific Models
- Tests 4 model types per team: Ridge, Lasso, Random Forest, Gradient Boosting
- Automatically selects the best performing model based on R² score
- Models predict goal differential based on player lineups and features

### 2. Extract Player Values via SHAP
- **Tree-based models**: Uses SHAP (SHapley Additive exPlanations) to measure each player's marginal contribution
- **Linear models**: Uses model coefficients as player values
- SHAP values represent how much each player impacts team performance

### 3. Solve Optimal Lineup Problem (Knapsack)
- **Objective**: Maximize total player value (sum of SHAP values)
- **Constraint**: Total player rating ≤ 8.0 (the budget), >= 7.0 (minimum rating present in data)
- **Selection**: Choose exactly 4 players
- Evaluates all possible 4-player combinations to find optimal lineup

## How to Run

1. **Install Requirementx**: Activate a virtual environment (recommended) and run `pip install -r requirements.txt`
2. **Dashboard**: Run the Streamlit dashboard with:
   ```bash
   streamlit run lineup_dashboard.py
   ```
   Note it may take a few moments to load

The **Analysis.ipynb** Notebook can be ran by clicking run all, like any other Jupyter Notebook
## Key Results
- Model performance metrics (R²) for all teams
- Player value rankings based on SHAP analysis
- Optimal 4-player lineups for each team within rating constraints

