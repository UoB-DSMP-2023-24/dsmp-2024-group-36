import pandas as pd

filepath = '/Users/weiwei/PycharmProjects/DSMP/shop_transfer_data.csv'  # 将此路径替换为您的CSV文件的实际路径
df = pd.read_csv(filepath)

# 删除 'to_randomly_generated_account' 列
df = df.drop(columns=['to_randomly_generated_account'])

# 保存为新的CSV文件
new_filepath = 'processed_shop_transfer_data.csv'  # 指定新CSV文件的保存路径
df.to_csv(new_filepath, index=False)
