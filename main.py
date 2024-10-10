#Author: Wyatt Cassiotis
import pandas as pd
matches = pd.read_csv("win_loss_methods_count (1).csv", index_col=3)
futureMatches =pd.read_csv("win_loss_methods_count.csv", index_col=3)
matches["map_code"]=matches["Map"].astype("category").cat.codes
matches["team_code"]=matches["Team"].astype("category").cat.codes
matches["target"] =((matches["Elimination"]+matches["Time Expiry (No Plant)"]+matches["Detonated"]+matches["Defused"])==13).astype(int)
futureMatches["team_code"]=futureMatches["Team"].astype("category").cat.codes
futureMatches["map_code"]=futureMatches["Map"].astype("category").cat.codes
futureMatches["target"] =((futureMatches["Elimination"]+futureMatches["Time Expiry (No Plant)"]+futureMatches["Detonated"]+futureMatches["Defused"])==13).astype(int)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches 
test = futureMatches 
map_to_code = dict(enumerate(matches["Map"].astype("category").cat.categories))
code_to_map = {v: k for k, v in map_to_code.items()}
team_to_code = dict(enumerate(matches["Team"].astype("category").cat.categories))
code_to_team = {v: k for k, v in team_to_code.items()}
predictors = ["team_code","map_code","Eliminated","Defused Failed"]
rf.fit(train[predictors],train["target"]) 
preds=rf.predict(test[predictors])
from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target"],preds)
combined = pd.DataFrame(dict(actual=test["target"],predicted=preds))
pd.crosstab(index=combined["actual"],columns=combined["predicted"])
def predict_win_or_loss():
    team_name = input("Enter the team name: ")
    if team_name not in code_to_team:
        print(f"Team '{team_name}' not found. Please try again.")
        print(code_to_team)
        return
    team_code = code_to_team[team_name]

    map_name = input("Enter the map name: ")
    if map_name not in code_to_map:
        print(f"Map '{map_name}' not found. Please try again.")
        return
    map_code = code_to_map[map_name]
    eliminated = int(input("Enter number of eliminations: "))
    defused_failed = int(input("Enter Defused Failed: "))
    user_input = pd.DataFrame([[team_code,map_code, eliminated, defused_failed]],
                              columns=predictors)
    prediction = rf.predict(user_input)

    if prediction == 1:
        print("Prediction: Win (1)")
    else:
        print("Prediction: Loss (0)")
    
while(1):
    predict_win_or_loss()
    quits= input("Type 0 to quit or anything else to continue: ")
    if(quits=='0'):
        break
    

    