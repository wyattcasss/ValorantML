import pandas as pd
matches = pd.read_csv("win_loss_methods_count (1).csv", index_col=0)
futureMatches =pd.read_csv("win_loss_methods_count (1).csv", index_col=0)
matches["map_code"]=matches["Map"].astype("category").cat.codes
matches["target"] =(matches["Elimination"])
futureMatches["map_code"]=futureMatches["Map"].astype("category").cat.codes
futureMatches["target"] =(futureMatches["Elimination"])
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = matches 
test = futureMatches 
predictors = ["map_code",
              "Time Expiry (No Plant)",
              "Eliminated","Defused Failed",
              "Detonation Denied","Time Expiry (Failed to Plant)"]
rf.fit(train[predictors],train["target"]) 
preds=rf.predict(test[predictors])
from sklearn.metrics import accuracy_score
acc = accuracy_score(test["target"],preds)
print(acc)