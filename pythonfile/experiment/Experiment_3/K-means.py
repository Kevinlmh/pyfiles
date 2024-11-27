import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据
data = np.loadtxt('arab_digits_training.txt', delimiter='\t')
features = data[:, 1:]  # 第2-785列为特征数据

# 标准化数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 应用PCA降维
pca = PCA(n_components=2)  # 降到2维
features_pca = pca.fit_transform(features_scaled)

# 应用K-均值聚类
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(features_pca)

# 绘制PCA降维后的空间分布
plt.figure(figsize=(10, 8))
colors = plt.colormaps["tab10"]  # 使用现代方式获取 colormap 对象

# 绘制各聚类点
for cluster_id in range(10):
    cluster_points = features_pca[clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                color=colors(cluster_id / 10), label=f"Cluster {cluster_id}", alpha=0.6)

# 标注聚类中心
for center in kmeans.cluster_centers_:
    plt.scatter(center[0], center[1], color='black', marker='o', s=100)  # 黑点标记聚类中心

# 图形设置
plt.title("K-Means Clustering on PCA-Reduced Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()
