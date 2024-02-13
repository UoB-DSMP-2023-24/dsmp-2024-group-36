import pandas as pd
from sklearn.ensemble import IsolationForest

data = pd.read_csv("shop_transfer_data.csv")
X = data[["monopoly_money_amount", "Shop_type", "Expected consumption level"]]

X_encoded = pd.get_dummies(X)

isolation_forest = IsolationForest(n_estimators=1000, contamination=0.1, random_state=42)

isolation_forest.fit(X_encoded)
scores = isolation_forest.decision_function(X_encoded)
data["anomaly_score"] = scores

threshold = data["anomaly_score"].quantile(0.01) # need survey to determin the quantile
anomalies = data[data["anomaly_score"] < threshold]

print("Detected anomalies:")
print(anomalies)
anomalies.to_csv("shop_anomalies.csv", index=False)
