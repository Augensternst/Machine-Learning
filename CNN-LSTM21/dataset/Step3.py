'''
---------------------------------------
             数据预处理3
             数据归一化

---------------------------------------
'''


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 读取CSV文件
input_file = 'selected_data.csv'
data = pd.read_csv(input_file, header=None)

# 选择第8列以后的数据
columns_to_normalize = data.columns[8:]

# """使用MinMaxScaler进行归一化"""
# scaler = MinMaxScaler()

"""使用StandardScaler进行Z-Score归一化"""
scaler = StandardScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# 保存结果到新的CSV文件
output_file = './N-CMAPSS/dataset.csv'
data.to_csv(output_file, index=False, header=False)
