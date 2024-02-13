import pandas as pd


data = pd.read_csv("shop_transfer_data.csv")

interested_shops = ["CINEMA", "HIPSTER_COFFEE_SHOP", "TOTALLY_A_REAL_COFFEE_SHOP", "COFFEE_SHOP"]
filtered_data = data[data["to_randomly_generated_account"].isin(interested_shops)]

shop_counts = filtered_data["to_randomly_generated_account"].value_counts()
print(shop_counts)


# data = pd.read_csv("shop_transfer_data.csv")
#
# shop_statistics = data.groupby("to_randomly_generated_account").agg(
#     Transaction_Count=("monopoly_money_amount", "count"),
#     Total_Amount=("monopoly_money_amount", "sum"),
#     Average_Amount=("monopoly_money_amount", "mean")
# ).reset_index()
#
# shop_statistics.to_csv("shop_statistics.csv", index=False)


