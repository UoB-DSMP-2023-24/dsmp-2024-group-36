import pandas as pd
from sklearn.ensemble import IsolationForest

data = pd.read_csv("shop_transfer_data.csv")


averages = pd.read_csv("shop_statistics.csv")

data_merged = pd.merge(data, averages, on="to_randomly_generated_account", how="left")

X = data_merged[["monopoly_money_amount", "Average_Amount"]]

isolation_forest = IsolationForest(n_estimators=1000, contamination='auto', random_state=42)
isolation_forest.fit(X)
scores = isolation_forest.decision_function(X)
data_merged["anomaly_score"] = scores

threshold = data_merged["anomaly_score"].quantile(0.06)
anomalies = data_merged[data_merged["anomaly_score"] < threshold]

print("Detected probable fraud:")
print(anomalies)

anomalies.to_csv("probable_fraud.csv", index=False)
