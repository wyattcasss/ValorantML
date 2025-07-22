# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
matches = pd.read_csv("win_loss_methods_count (1).csv", index_col=3)
futureMatches = pd.read_csv("win_loss_methods_count.csv", index_col=3)

# Encode categories
matches["map_code"] = matches["Map"].astype("category").cat.codes
matches["team_code"] = matches["Team"].astype("category").cat.codes
matches["target"] = ((matches["Elimination"] + matches["Time Expiry (No Plant)"] +
                      matches["Detonated"] + matches["Defused"]) == 13).astype(int)

futureMatches["team_code"] = futureMatches["Team"].astype("category").cat.codes
futureMatches["map_code"] = futureMatches["Map"].astype("category").cat.codes
futureMatches["target"] = ((futureMatches["Elimination"] + futureMatches["Time Expiry (No Plant)"] +
                            futureMatches["Detonated"] + futureMatches["Defused"]) == 13).astype(int)

# Mappings
map_to_code = dict(enumerate(matches["Map"].astype("category").cat.categories))
code_to_map = {v: k for k, v in map_to_code.items()}
team_to_code = dict(enumerate(matches["Team"].astype("category").cat.categories))
code_to_team = {v: k for k, v in team_to_code.items()}

# Model training
predictors = ["team_code", "map_code", "Eliminated", "Defused Failed"]
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(matches[predictors], matches["target"])

# Streamlit UI
st.title("Win/Loss Predictor")
st.markdown("Enter match data to predict if the team will **Win (1)** or **Lose (0)**.")

team_name = st.selectbox("Team", list(code_to_team.keys()))
map_name = st.selectbox("Map", list(code_to_map.keys()))
eliminated = st.number_input("Number of Eliminations", min_value=0, step=1)
defused_failed = st.number_input("Number of Defused Failed", min_value=0, step=1)

if st.button("Predict"):
    # Convert inputs to codes
    team_code = code_to_team[team_name]
    map_code = code_to_map[map_name]

    user_input = pd.DataFrame([[team_code, map_code, eliminated, defused_failed]],
                              columns=predictors)

    prediction = rf.predict(user_input)[0]
    probability = rf.predict_proba(user_input)[0][prediction]

    if prediction == 1:
        st.success(f"Prediction: **Win (1)** with {probability:.2%} confidence")
    else:
        st.error(f"Prediction: **Loss (0)** with {probability:.2%} confidence")
