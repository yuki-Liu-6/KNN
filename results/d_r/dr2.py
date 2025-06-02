"""
乳腺癌数据集可视化分析（修正版）
生成论文所需的三种图表：
1. 类别分布图
2. 特征对比图（箱线图+小提琴图）
3. 标准化效果对比图
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

# === 修正点1：更新样式设置 ===
# 设置风格（使用Seaborn的whitegrid风格）
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'  # 学术论文常用字体
plt.rcParams['font.size'] = 12

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
target_names = data.target_names

# 创建DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['diagnosis'] = np.where(y == 0, 'malignant', 'benign')

# 创建结果目录
os.makedirs("results", exist_ok=True)

# %% 1. 类别分布图
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='diagnosis', data=df, order=['malignant', 'benign'])
plt.title("Class Distribution (Malignant vs Benign)", pad=20)
plt.xlabel("Diagnosis")
plt.ylabel("Count")

# 添加百分比标签
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    ax.annotate(percentage, 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points')

plt.tight_layout()
plt.savefig("results/class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# %% 2. 特征对比图（选择最具区分度的两个特征）
plt.figure(figsize=(12, 6))

# 特征1：最差半径（worst radius）
plt.subplot(1, 2, 1)
sns.boxplot(x='diagnosis', y='worst radius', data=df, 
            order=['malignant', 'benign'], width=0.6)
plt.title("Worst Radius by Diagnosis", pad=15)
plt.xlabel("")
plt.ylabel("Worst Radius (standardized)")

# 特征2：平均纹理（mean texture）
plt.subplot(1, 2, 2)
sns.violinplot(x='diagnosis', y='mean texture', data=df,
               order=['malignant', 'benign'], cut=0, inner="quartile")
plt.title("Mean Texture by Diagnosis", pad=15)
plt.xlabel("")
plt.ylabel("Mean Texture (standardized)")

plt.suptitle("Feature Distribution Comparison", y=1.02)
plt.tight_layout()
plt.savefig("results/feature_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# %% 3. 标准化效果对比图（以mean radius为例）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(12, 5))

# 标准化前
plt.subplot(1, 2, 1)
sns.histplot(df['mean radius'], kde=True, bins=30)
plt.title("Before Standardization", pad=15)
plt.xlabel("Mean Radius (original scale)")
plt.ylabel("Density")

# 标准化后
plt.subplot(1, 2, 2)
sns.histplot(X_scaled[:, 0], kde=True, bins=30)  # 第0列是mean radius
plt.title("After Standardization", pad=15)
plt.xlabel("Mean Radius (standardized)")
plt.ylabel("")

plt.suptitle("Standardization Effect on Mean Radius", y=1.02)
plt.tight_layout()
plt.savefig("results/standardization_effect.png", dpi=300, bbox_inches='tight')
plt.close()

print("可视化图表已保存至results/目录：")
print("- class_distribution.png")
print("- feature_comparison.png")
print("- standardization_effect.png")