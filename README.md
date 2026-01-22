# Player Value Analysis & Optimal Lineup Selection

## Overview
This project analyzes player performance data to determine optimal team lineups using machine learning and SHAP values. The analysis trains team-specific models, extracts player values, and solves a constrained optimization problem to build the best lineups.

## Environment Setup

### Required Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap streamlit
```

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
- **Constraint**: Total player rating ≤ 8.0 (the budget)
- **Selection**: Choose exactly 4 players
- Evaluates all possible 4-player combinations to find optimal lineup

## How to Run

1. **Analysis Notebook**: Open `analysis.ipynb` and run all cells sequentially
2. **Dashboard**: Run the Streamlit dashboard with:
   ```bash
   streamlit run lineup_dashboard.py
   ```

## Key Results
- Model performance metrics (R², RMSE) for all teams
- Player value rankings based on SHAP analysis
- Optimal 4-player lineups for each team within rating constraints

## Files
- `analysis.ipynb` - Main analysis notebook
- `lineup_dashboard.py` - Interactive Streamlit dashboard
- `model.ipynb` - Additional modeling experiments
- `data_dict.md` - Data dictionary
- `README_DASHBOARD.md` - Dashboard documentation
