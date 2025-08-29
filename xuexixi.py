import numpy as np
import pandas as pd
import datetime

# 读取数据
data = pd.read_csv("D:/wenjian/UserBehavior.csv",names=["user_id", "item_id", "category_id", "behavior_type", "timestamp"])

# 查看数据
data.describe() # 数值型字段分布
data.info() # 查看缺失值和数据类型
data.shape # 查看行数与列数
data["behavior_type"].value_counts() # 行为类型分布

# 转换日期格式
data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s", errors="coerce")

# 提取日期、小时、星期
data["date"] = data["timestamp"].dt.date
data["hour"] = data["timestamp"].dt.hour
data['weekday'] = data["timestamp"].dt.weekday

# 查看是否有缺失值
data.isnull().sum()
# 查看数据时间范围
"时间从：",data.min(),"到",data.max()

# 分块大小
chunksize = 100000
# 目标抽样量
sample_size = 1_000_000
# 随机种子
random_seed = 42

# 第一次遍历：统计各行为类型的总量
behavior_counts = {'pv': 0, 'buy': 0, 'cart': 0, 'fav': 0}
reader = pd.read_csv('D:/wenjian/UserBehavior.csv', names=['user_id', 'item_id', 'category_type', 'behavior_type', 'timestamp'], chunksize=chunksize)
for chunk in reader:
    counts = chunk['behavior_type'].value_counts()
    for k in behavior_counts:
        behavior_counts[k] += counts.get(k, 0)

# 计算原始数据中各行为类型的比例
total_records = sum(behavior_counts.values())
original_proportions = {k: v / total_records for k, v in behavior_counts.items()}
print("原始数据中各行为类型的比例：")
print(pd.Series(original_proportions))

# 计算各行为类型的抽样比例
sample_ratios = {k: (sample_size * v / total_records) / v
                 for k, v in behavior_counts.items()}

# 第二次遍历：分块进行分层抽样
sample_chunks = []
reader = pd.read_csv("D:/wenjian/UserBehavior.csv",names=['user_id', 'item_id', 'category_type', 'behavior_type', 'timestamp'], chunksize=chunksize)
for chunk in reader:
    sampled = chunk.groupby('behavior_type', group_keys=False).apply(
        lambda g: g.sample(
            n=int(len(g) * sample_ratios[g.name]),
            random_state=random_seed
        )
    ).reset_index(drop=True)
    sample_chunks.append(sampled)

# 合并所有抽样块
sample_df = pd.concat(sample_chunks)

# 计算抽样后数据中各行为类型的比例
sampled_proportions = sample_df['behavior_type'].value_counts(normalize=True)
print("\n抽样后数据中各行为类型的比例：")
print(sampled_proportions)

# 将抽样后的数据存储到本地D:/wenjian
sample_df.to_csv("D:/wenjian/sample_data.csv", index=False)

# 读取抽样后的数据文件
file_path = "D:/wenjian/sample_data.csv"
sampled_df = pd.read_csv(file_path)

# 数据预览
print(sampled_df.info())

# 查看数据行数和列数
rows, columns = sampled_df.shape

# 数据行数大于0才进行后续操作
if rows > 0:
    print("\n数据前几行信息：")
    print(sampled_df.head().to_csv(sep='\t', na_rep='nan'))

    # 查看各列缺失值情况
    missing_values = sampled_df.isnull().sum()
    print("\n各列缺失值数量：")
    print(missing_values)

    # 处理缺失值
    if missing_values.sum() > 0:
        sampled_df = sampled_df.dropna()
        print("\n已删除含有缺失值的行。")

    # 查看重复值数量
    duplicate_count = sampled_df.duplicated().sum()
    print(f"\n重复值数量：{duplicate_count}")

    # 处理重复值
    if duplicate_count > 0:
        sampled_df = sampled_df.drop_duplicated()
        print("已删除重复值")

    # 处理异常值（以timestamp列为例，假设时间戳应在合理范围内）
    if 'timestamp' in sampled_df.columns:
        sampled_df['timestamp'] = pd.to_datetime(sampled_df['timestamp'], unit='s')
        # 假设数据时间范围在2017年11月25日至2017年12月3日
        start_date = pd.Timestamp('2017-11-25')
        end_date = pd.Timestamp('2017-12-03')
        outlier_mask = (sampled_df['timestamp'] < start_date) | (sampled_df['timestamp'] > end_date)
        outlier_count = outlier_mask.sum()
        print(f"\ntimestamp列异常值数量：{outlier_count}")

        # 处理异常值
        cleaned_file_path = "D:/wenjian/cleaned_sampled_data.csv"
        sampled_df.to_csv(cleaned_file_path, index=False)
        print(f"\n清洗后的数据已保存至{cleaned_file_path}")
    else:
        print("抽样数据文件为空，无法进行后续操作。")