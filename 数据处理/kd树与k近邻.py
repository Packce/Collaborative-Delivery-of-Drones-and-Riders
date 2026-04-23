import pandas as pd
import numpy as np
from scipy.spatial import KDTree

# 1. 读取人口密度数据
density_df = pd.read_csv("D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据\人口密度数据.csv")
# 提取坐标和密度值（过滤掉无效值 -3.402823e+38）
valid_density = density_df[density_df['Value'] > -3e38].copy()
points = valid_density[['X', 'Y']].values
values = valid_density['Value'].values

# 2. 构建KD树用于快速最近邻搜索
tree = KDTree(points)

# 3. 读取POI数据
poi_df = pd.read_csv("D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据\顾客数据.csv", encoding='utf-8')
poi_coords = poi_df[['Center_X', 'Center_Y']].values

# 4. 查找每个POI最近的密度点，获取对应密度值
distances, indices = tree.query(poi_coords)
poi_df['人口密度'] = values[indices]

# 5. 保存结果
poi_df.to_csv("D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据\顾客数据_带人口密度.csv", index=False, encoding='utf-8-sig')

print("处理完成！共处理 {} 条POI记录".format(len(poi_df)))
print("\n人口密度统计：")
print(poi_df['人口密度'].describe())