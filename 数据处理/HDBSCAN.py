import pandas as pd
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# ================== 配置参数 ==================
DATA_PATH = "D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据\顾客数据_带人口密度.csv"
MIN_CLUSTER_SIZE = 18               # 最小簇大小
MIN_SAMPLES = 9                     # 密度估计参数
USE_POP_DENSITY = True              # 是否将人口密度作为聚类特征
POP_DENSITY_WEIGHT = 2.0            # 人口密度权重，值越大人口密度影响越大
CENTER_WEIGHT = 0.6      # 密度中心权重（距离倒数的权重）
DENSITY_WEIGHT = 0.4     # 人口密度权重
OUTPUT_CSV = "D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据处理\聚类结果_适中.csv"
OUTPUT_PLOT = "D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据处理\聚类结果_适中.png"

# ================== 读取数据 ==================
df = pd.read_csv(DATA_PATH)
print(f"原始数据行数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

required_cols = ["Center_X", "Center_Y", "人口密度"]
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"数据中缺少必要的列，请确保包含: {required_cols}")

df = df.dropna(subset=required_cols).reset_index(drop=True)
print(f"剔除缺失值后行数: {len(df)}")

# ================== 特征提取（添加权重） ==================
X_coord = df[["Center_X", "Center_Y"]].values

if USE_POP_DENSITY:
    # 人口密度乘以权重后再参与聚类
    pop_density_weighted = df[["人口密度"]].values * POP_DENSITY_WEIGHT
    features = np.hstack([X_coord, pop_density_weighted])
    # 标准化所有特征，使各维度量纲一致
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"已标准化坐标 + 加权后的人口密度特征（权重={POP_DENSITY_WEIGHT}）")
else:
    features_scaled = X_coord
    print("仅使用坐标特征（未标准化）")

# ================== HDBSCAN 聚类 ==================
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric='euclidean',
    prediction_data=True
)

labels = clusterer.fit_predict(features_scaled)
df['cluster_label'] = labels
probabilities = clusterer.probabilities_
df['cluster_probability'] = probabilities

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = sum(labels == -1)
print(f"聚类完成: 共发现 {n_clusters} 个簇，噪声点 {n_noise} 个")
print("各簇大小:")
print(df['cluster_label'].value_counts().sort_index())

# ================== 提取候选起降点（密度中心+人口密度加权） ==================
candidate_points = []
for label in set(labels):
    if label == -1:
        continue
    cluster_data = df[df['cluster_label'] == label]
    coords = cluster_data[['Center_X', 'Center_Y']].values
    pop_density = cluster_data['人口密度'].values
    
    n_points = len(coords)
    if n_points == 1:
        # 只有一个点时，直接选该点
        best_idx = cluster_data.index[0]
    else:
        # 1. 计算每点到其他点的平均距离
        pairwise_dist = squareform(pdist(coords))   # 对称距离矩阵
        # 对角线为0，计算每行的平均值（即该点到其他点的平均距离）
        avg_dist = pairwise_dist.sum(axis=1) / (n_points - 1)
        # 2. 密度中心得分 = 平均距离的倒数（加小常数防除零）
        center_score = 1.0 / (avg_dist + 1e-8)
        
        # 3. 对密度中心得分和人口密度进行 Min-Max 标准化（簇内）
        center_score_norm = (center_score - center_score.min()) / (center_score.max() - center_score.min() + 1e-8)
        pop_norm = (pop_density - pop_density.min()) / (pop_density.max() - pop_density.min() + 1e-8)
        
        # 4. 加权综合得分
        combined_score = CENTER_WEIGHT * center_score_norm + DENSITY_WEIGHT * pop_norm
        
        # 5. 选择综合得分最高的点
        best_idx = cluster_data.index[np.argmax(combined_score)]
    
    candidate = cluster_data.loc[best_idx]
    candidate_points.append({
        'cluster_id': label,
        'X': candidate['Center_X'],
        'Y': candidate['Center_Y'],
        '人口密度': candidate['人口密度'],
        '点数量': len(cluster_data),
        '平均人口密度': cluster_data['人口密度'].mean(),
        '推荐点名称': candidate.get('名称', 'N/A'),
        '核心点类型': '加权中心'
    })

candidate_df = pd.DataFrame(candidate_points)
print("\n候选起降点（密度中心+人口密度加权）:")
print(candidate_df)

# ================== 可视化（点大小按人口密度映射） ==================
plt.figure(figsize=(12, 10))

# 计算点大小：将人口密度线性映射到 [min_size, max_size] 区间
pop = df['人口密度'].values
# 为避免极端值影响，可以取99分位数作为最大值
p99 = np.percentile(pop, 99)
pop_clipped = np.clip(pop, 0, p99)  # 截断极端大值
# 线性映射：最小点大小5，最大点大小100
min_size, max_size = 5, 100
sizes = min_size + (pop_clipped - pop_clipped.min()) / (pop_clipped.max() - pop_clipped.min()) * (max_size - min_size)
# 如果所有人口密度相同，则设为默认大小
if np.isnan(sizes).any():
    sizes = np.full_like(pop, 15)

# 绘制聚类结果
unique_labels = set(labels)
palette = sns.color_palette("tab10", len(unique_labels) - (1 if -1 in labels else 0))

for label in unique_labels:
    if label == -1:
        color = 'lightgray'
        label_name = 'Noise'
        alpha = 0.5
    else:
        color = palette[label % len(palette)]
        label_name = f'Cluster {label}'
        alpha = 0.7

    mask = df['cluster_label'] == label
    plt.scatter(df.loc[mask, 'Center_X'], df.loc[mask, 'Center_Y'],
                c=color, label=label_name, alpha=alpha,
                s=sizes[mask], edgecolors='none')

# 标记候选起降点（星标大小固定）
if not candidate_df.empty:
    plt.scatter(candidate_df['X'], candidate_df['Y'],
                marker='*', s=200, c='red', edgecolors='black',
                label='Candidate Sites')

plt.title(f'HDBSCAN Clustering (min_cluster_size={MIN_CLUSTER_SIZE}, '
          f'min_samples={MIN_SAMPLES}, pop_weight={POP_DENSITY_WEIGHT})')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
plt.show()
print(f"聚类图已保存至: {OUTPUT_PLOT}")

# ================== 保存结果 ==================
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"聚类结果已保存至: {OUTPUT_CSV}")

candidate_df.to_csv("D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据处理\候选点_适中.csv", index=False, encoding='utf-8-sig')
print("候选点列表已保存至: D:\文档\大学文件\竞赛\统计建模\真正的统模\代码和数据\数据处理\候选点_适中.csv")
