import pandas as pd

# 加载数据集
data = pd.read_csv('Dataset.csv')

# 使用 describe() 方法查看描述性统计分布情况
description = data.describe()
print("描述性统计分布情况：")
print(description)

# 使用 info() 方法查看缺失情况
print("\n缺失情况：")
info = data.info()
print(info)

# 删除重复值
data_without_duplicates = data.drop_duplicates()

# 删除缺失值
data_without_missing_values = data.dropna()

# 打印删除重复值后的数据集
print("\n删除重复值后的数据集：")
print(data_without_duplicates.head())

# 打印删除缺失值后的数据集
print("\n删除缺失值后的数据集：")
print(data_without_missing_values.head())
