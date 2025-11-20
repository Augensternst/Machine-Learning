'''
---------------------------------------
             数据预处理2
             检查数据切片

---------------------------------------
'''




import pandas as pd
import random

# 读取 CSV 文件
df_a = pd.read_csv('A.csv', usecols=[0, 1, 3], header=None)  # 辅助列表 3
df_b = pd.read_csv('Y.csv', header=None)  # Rul 1
df_c = pd.read_csv('W.csv', header=None)  # 工况 4
df_d = pd.read_csv('X_s.csv', header=None)
df_e = pd.read_csv('X_v.csv', header=None)
df_f = pd.read_csv('T.csv', usecols=[6, 8, 9], header=None)

window_length = 30
# 使用 pd.concat() 进行左右列拼接
combined_df = pd.concat([df_a, df_b, df_c, df_d, df_e, df_f], axis=1)
print(combined_df.shape[1])
# 创建一个空的DataFrame来存储结果
result_df = pd.DataFrame(columns=combined_df.columns)

# 获取唯一的发动机编号（第一列数据）
engine_numbers = combined_df.iloc[:, 0].unique()

# 遍历每个发动机编号
for engine_number in engine_numbers:
    # 获取该发动机的所有RUL值（第四列数据）
    ruls = combined_df[combined_df.iloc[:, 0] == engine_number].iloc[:, 3].unique()

    # 遍历每个RUL值
    for rul in ruls:
        # 筛选该发动机编号和RUL值对应的行
        engine_rul_df = combined_df[(combined_df.iloc[:, 0] == engine_number) & (combined_df.iloc[:, 3] == rul)]

        # 检查是否有足够的行来选择30行数据
        if len(engine_rul_df) >= window_length:
            # 选择随机的起始行
            start_index = random.randint(0, len(engine_rul_df) - 30)
            # 提取连续的30行数据
            selected_data = engine_rul_df.iloc[start_index:start_index + 30]
            # 将选定的数据添加到结果DataFrame中
            result_df = pd.concat([result_df, selected_data], ignore_index=True)

# 保存结果到新的CSV文件
output_file = 'selected_data.csv'
result_df.to_csv(output_file, index=False, header=None)
