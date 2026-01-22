import pandas as pd
import numpy as np
import streamlit as st
from itertools import combinations
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap

# Page configuration
st.set_page_config(page_title="Optimal Lineup Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.title("Optimal Lineup Dashboard")

# Load data
@st.cache_data
def load_data():
    stint_data = pd.read_csv('stint_data.csv')
    player_data = pd.read_csv('player_data.csv')
    return stint_data, player_data

# Build team player models
@st.cache_resource
def build_models(stint_data, player_data):
    # Calculate team stats
    team_plus_minus = {}
    for _, row in stint_data.iterrows():
        h_team = row['h_team']
        a_team = row['a_team']
        goal_diff = row['h_goals'] - row['a_goals']
        
        if h_team not in team_plus_minus:
            team_plus_minus[h_team] = 0
        team_plus_minus[h_team] += goal_diff
        
        if a_team not in team_plus_minus:
            team_plus_minus[a_team] = 0
        team_plus_minus[a_team] -= goal_diff

    team_plus_minus_df = pd.DataFrame(list(team_plus_minus.items()), 
                                       columns=['team', 'plus_minus'])
    
    team_minutes = {}
    for _, row in stint_data.iterrows():
        h_team = row['h_team']
        a_team = row['a_team']
        minutes = row['minutes']
        
        if h_team not in team_minutes:
            team_minutes[h_team] = 0
        team_minutes[h_team] += minutes
        
        if a_team not in team_minutes:
            team_minutes[a_team] = 0
        team_minutes[a_team] += minutes

    team_minutes_df = pd.DataFrame(list(team_minutes.items()), 
                                    columns=['team', 'total_minutes'])

    team_stats = team_plus_minus_df.merge(team_minutes_df, on='team')
    team_stats['plus_minus_per_10min'] = (team_stats['plus_minus'] / team_stats['total_minutes']) * 10
    
    rating_lookup = player_data.set_index('player')['rating'].to_dict()
    team_strength = team_stats.set_index('team')['plus_minus_per_10min'].to_dict()
    all_teams = sorted(stint_data['h_team'].unique())

    team_player_models = {}

    for team in all_teams:
        team_player_list = player_data[player_data['player'].str.startswith(f"{team}_")]['player'].tolist()
        
        if len(team_player_list) == 0:
            continue
        
        player_to_idx = {player: idx for idx, player in enumerate(team_player_list)}
        
        X_features = []
        y_values = []
        
        for _, row in stint_data.iterrows():
            is_home = row['h_team'] == team
            is_away = row['a_team'] == team
            
            if not (is_home or is_away):
                continue
                
            if is_home:
                goal_diff = row['h_goals'] - row['a_goals']
                our_players = [row['home1'], row['home2'], row['home3'], row['home4']]
                opp_players = [row['away1'], row['away2'], row['away3'], row['away4']]
                opp_team = row['a_team']
            else:
                goal_diff = row['a_goals'] - row['h_goals']
                our_players = [row['away1'], row['away2'], row['away3'], row['away4']]
                opp_players = [row['home1'], row['home2'], row['home3'], row['home4']]
                opp_team = row['h_team']
            
            player_indicators = np.zeros(len(team_player_list))
            player_ratings = np.zeros(len(team_player_list))
            
            for p in our_players:
                if p in player_to_idx:
                    idx = player_to_idx[p]
                    player_indicators[idx] = 1
                    player_ratings[idx] = rating_lookup.get(p, 0)
            
            our_total_rating = sum([rating_lookup.get(p, 0) for p in our_players])
            our_avg_rating = our_total_rating / 4.0
            
            opp_total_rating = sum([rating_lookup.get(p, 0) for p in opp_players])
            opp_avg_rating = opp_total_rating / 4.0
            opp_strength = team_strength.get(opp_team, 0)
            
            home_advantage = 1.0 if is_home else 0.0
            
            rating_diff = our_avg_rating - opp_avg_rating
            rating_product = our_avg_rating * opp_avg_rating
            
            features = np.concatenate([
                player_indicators,
                player_ratings,
                [our_total_rating],
                [our_avg_rating],
                [opp_total_rating],
                [opp_avg_rating],
                [opp_strength],
                [rating_diff],
                [rating_product],
                [home_advantage]
            ])
            
            X_features.append(features)
            # Normalize goal differential by minutes
            y_values.append(goal_diff / row['minutes'])
        
        if len(X_features) == 0:
            continue
        
        feature_names = []
        
        for player in team_player_list:
            feature_names.append(f"indicator_{player}")
        
        for player in team_player_list:
            feature_names.append(f"rating_{player}")
        
        feature_names.extend([
            'our_total_rating',
            'our_avg_rating',
            'opp_total_rating',
            'opp_avg_rating',
            'opp_strength',
            'rating_diff',
            'rating_product',
            'home_advantage'
        ])
        
        X = pd.DataFrame(X_features, columns=feature_names)
        y = np.array(y_values)
        
        models_to_try = {
            'Ridge (α=50)': Ridge(alpha=50.0),
            'Ridge (α=10)': Ridge(alpha=10.0),
            'Lasso (α=1)': Lasso(alpha=1.0, max_iter=10000),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
            'Gradient Boost': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
        }
        
        best_r2 = -float('inf')
        best_model_name = None
        best_model = None
        
        for name, model in models_to_try.items():
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model = model
        
        if hasattr(best_model, 'coef_'):
            player_values = best_model.coef_[:len(team_player_list)]
            rating_effects = best_model.coef_[len(team_player_list):2*len(team_player_list)]
        else:
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X)
            
            player_values = np.mean(shap_values[:, :len(team_player_list)], axis=0)
            rating_effects = np.mean(shap_values[:, len(team_player_list):2*len(team_player_list)], axis=0)
        
        results_df = pd.DataFrame({
            'player': team_player_list,
            'player_value': player_values,
            'rating_effect': rating_effects
        })
        
        results_df['rating'] = results_df['player'].map(rating_lookup)
        
        # Normalize player_value and rating_effect to [-1, 1]
        for col in ['player_value', 'rating_effect']:
            min_val = results_df[col].min()
            max_val = results_df[col].max()
            if max_val != min_val:  # Avoid division by zero
                results_df[col] = 2 * (results_df[col] - min_val) / (max_val - min_val) - 1
            else:
                results_df[col] = 0  # If all values are the same, set to 0
        
        # Combined impact is the sum of normalized values
        results_df['combined_impact'] = results_df['player_value'] + results_df['rating_effect']
        
        results_df = results_df.sort_values('player_value', ascending=False).reset_index(drop=True)
        
        team_player_models[team] = {
            'model': best_model,
            'model_name': best_model_name,
            'results': results_df,
            'r2': best_r2
        }
    
    return team_player_models, rating_lookup

# Find optimal lineup
def find_optimal_lineup(team_results, selected_players, rating_lookup):
    if len(selected_players) < 4:
        return None, "Please select at least 4 players"
    
    players_list = []
    for _, row in team_results.iterrows():
        if row['player'] in selected_players:
            players_list.append({
                'player': row['player'],
                'value': row['player_value'],
                'rating': row['rating'],
                'combined_impact': row['combined_impact']
            })
    
    if len(players_list) < 4:
        return None, "Not enough valid players selected"
    
    all_combinations = list(combinations(players_list, 4))
    valid_total_ratings = [7.0, 7.5, 8.0]
    
    best_lineup = None
    best_total_value = -float('inf')
    lineup_evaluations = []
    
    for combo in all_combinations:
        total_value = sum([p['value'] for p in combo])
        total_rating = sum([p['rating'] for p in combo])
        avg_rating = total_rating / 4.0
        total_combined = sum([p['combined_impact'] for p in combo])
        
        if total_rating not in valid_total_ratings:
            continue
        
        lineup_evaluations.append({
            'players': [p['player'] for p in combo],
            'total_value': total_value,
            'total_rating': total_rating,
            'avg_rating': avg_rating,
            'total_combined_impact': total_combined
        })
        
        if total_value > best_total_value:
            best_total_value = total_value
            best_lineup = combo
    
    if best_lineup is None:
        return None, "No valid lineups found with total rating of 7.0, 7.5, or 8.0"
    
    return best_lineup, lineup_evaluations

# Main app
try:
    stint_data, player_data = load_data()
    
    with st.spinner("Building player value models..."):
        team_player_models, rating_lookup = build_models(stint_data, player_data)
    
    # Compact header with team selection and controls
    col_head1, col_head2, col_head3 = st.columns([2, 2, 1])
    
    with col_head1:
        available_teams = sorted(team_player_models.keys())
        selected_team = st.selectbox("🏀 Select Team", available_teams, label_visibility="collapsed", 
                                     placeholder="Choose a team...")
    
    with col_head2:
        if selected_team:
            team_results = team_player_models[selected_team]['results']
            selection_mode = st.radio("Selection", 
                                      ["All", "Top 10", "Custom"], 
                                      horizontal=True, label_visibility="collapsed")
    
    with col_head3:
        if selected_team:
            generate_btn = st.button("🎯 Find Lineup", type="primary", use_container_width=True)
    
    if selected_team:
        team_results = team_player_models[selected_team]['results']
        model_name = team_player_models[selected_team]['model_name']
        r2_score = team_player_models[selected_team]['r2']
        
        # Compact info bar
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Team", selected_team, label_visibility="visible")
        with info_col2:
            st.metric("Players", len(team_results))
        with info_col3:
            st.metric("Model", model_name)
        with info_col4:
            st.metric("R²", f"{r2_score:.3f}")
        
        st.divider()
        
        # Player selection
        if selection_mode == "All":
            selected_players = team_results['player'].tolist()
        elif selection_mode == "Top 10":
            selected_players = team_results.head(10)['player'].tolist()
        else:  # Custom
            with st.expander("🎯 Custom Player Selection", expanded=True):
                selected_players = st.multiselect(
                    f"Select players (at least 4) - {len(team_results['player'])} available",
                    team_results['player'].tolist(),
                    default=team_results['player'].tolist(),
                    label_visibility="collapsed",
                    key=f"player_select_{selected_team}"
                )
        
        # Two-column layout: Player stats on left, Results on right
        left_col, right_col = st.columns([3, 2])
        
        with left_col:
            st.markdown("**📊 Player Statistics**")
            display_df = team_results[team_results['player'].isin(selected_players)].copy()
            display_df['player_short'] = display_df['player'].str.split('_').str[1]
            display_df = display_df[['player_short', 'rating', 'player_value', 'rating_effect', 'combined_impact']]
            display_df.columns = ['Player', 'Rating', 'Player Value', 'Rating Value', 'Combined Value']
            
            st.dataframe(
                display_df.style.format({
                    'Rating': '{:.1f}',
                    'Player Value': '{:.4f}',
                    'Rating Value': '{:.4f}',
                    'Combined Value': '{:.4f}'
                }).background_gradient(subset=['Player Value'], cmap='RdYlGn'),
                height=350,
                use_container_width=True
            )
        
        with right_col:
            st.markdown("**🎯 Optimal Lineup**")
            
            if generate_btn:
                best_lineup, lineup_evaluations = find_optimal_lineup(
                    team_results, selected_players, rating_lookup
                )
                
                if best_lineup is None:
                    st.error(lineup_evaluations)
                else:
                    lineup_df = pd.DataFrame([{
                        'Player': p['player'].split('_')[1],
                        'Rating': p['rating'],
                        'Player Value': p['value'],
                        'Combined Value': p['combined_impact']
                    } for p in best_lineup])
                    
                    total_rating = sum([p['rating'] for p in best_lineup])
                    total_value = sum([p['value'] for p in best_lineup])
                    total_combined = sum([p['combined_impact'] for p in best_lineup])
                    
                    # Compact metrics in 2 columns
                    m1, m2 = st.columns(2)
                    m1.metric("Rating", f"{total_rating:.1f}")
                    m2.metric("Combined Value (objective value)", f"{total_combined:.3f}")
                    
                    # Compact lineup table
                    st.dataframe(
                        lineup_df.style.format({
                            'Rating': '{:.1f}',
                            'Player Value': '{:.4f}',
                            'Combined Value': '{:.4f}'
                        }),
                        hide_index=True,
                        height=180,
                        use_container_width=True
                    )
                    
                    # Alternatives in expander
                    if isinstance(lineup_evaluations, list) and len(lineup_evaluations) > 1:
                        with st.expander("📋 Top 10 Alternative Lineups"):
                            lineup_eval_df = pd.DataFrame([{
                                'Rank': i + 1,
                                'Players': ', '.join([p.split('_')[1] for p in lineup['players']]),
                                'Rating': lineup['total_rating'],
                                'Player Value': lineup['total_value'],
                                'Combined Value': lineup['total_combined_impact']
                            } for i, lineup in enumerate(sorted(lineup_evaluations, 
                                                                key=lambda x: x['total_value'], 
                                                                reverse=True)[:10])])
                            
                            st.dataframe(
                                lineup_eval_df.style.format({
                                    'Rating': '{:.1f}',
                                    'Player Value': '{:.4f}',
                                    'Combined Value': '{:.4f}'
                                }),
                                height=250,
                                use_container_width=True
                            )
            else:
                st.info("👆 Click 'Find Lineup' to generate optimal lineup")

except FileNotFoundError as e:
    st.error(f"❌ Error loading data files: {e}")
    st.info("Please make sure 'stint_data.csv' and 'player_data.csv' are in the same directory.")
except Exception as e:
    st.error(f"❌ An error occurred: {e}")
    st.exception(e)
