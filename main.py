import pandas as pd
matches = pd.read_csv("win_loss_methods_count (1).csv", index_col=5)
futureMatches =pd.read_csv("win_loss_methods_count (1).csv", index_col=5)
matches["map_code"]=matches["Map"].astype("category").cat.codes
matches["target"] =((matches["Elimination"]+matches["Time Expiry (No Plant)"]+matches["Detonated"]+matches["Defused"])==13).astype(int)
futureMatches["map_code"]=futureMatches["Map"].astype("category").cat.codes
futureMatches["target"] =((futureMatches["Elimination"]+matches["Time Expiry (No Plant)"]+matches["Detonated"]+matches["Defused"])==13).astype(int)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches 
test = futureMatches 
map_to_code = dict(enumerate(matches["Map"].astype("category").cat.categories))
code_to_map = {v: k for k, v in map_to_code.items()}
predictors = ["map_code","Eliminated","Defused Failed"]
rf.fit(train[predictors],train["target"]) 
preds=rf.predict(test[predictors])
from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target"],preds)
combined = pd.DataFrame(dict(actual=test["target"],predicted=preds))
pd.crosstab(index=combined["actual"],columns=combined["predicted"])
def predict_win_or_loss():
    map_name = input("Enter the map name: ")
    if map_name not in code_to_map:
        print(f"Map '{map_name}' not found. Please try again.")
        return
    map_code = code_to_map[map_name]
    eliminated = int(input("Enter number of eliminations: "))
    defused_failed = int(input("Enter Defused Failed: "))
    user_input = pd.DataFrame([[map_code, eliminated, defused_failed]],
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
    

    