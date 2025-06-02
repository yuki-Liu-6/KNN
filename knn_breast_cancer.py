"""
《人工智能》课程作业 - KNN算法在乳腺癌数据集上的实现
作者：[你的姓名]
学号：[你的学号]
日期：[提交日期]

完整实现包含：
1. 数据加载与探索
2. 数据预处理
3. 模型训练与评估
4. 结果可视化
5. 实验分析
"""

# %% 1. 导入库（需在论文中说明每个库的作用）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from pathlib import Path

# 设置随机种子保证可复现性（需在论文实验设置部分说明）
np.random.seed(42)

# %% 2. 数据加载与探索（对应论文4.1数据描述）
def load_and_explore_data():
    """加载并探索数据集"""
    print("\n" + "="*50)
    print("1. 数据加载与探索")
    print("="*50)
    
    # 官方数据集加载
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    # 创建DataFrame便于分析（论文中可展示前5行）
    df = pd.DataFrame(X, columns=feature_names)
    df['diagnosis'] = y
    df['diagnosis'] = df['diagnosis'].map({0: 'malignant', 1: 'benign'})
    
    # 数据集关键信息（需写入论文）
    print(f"""
    数据集名称：{data.DESCR.splitlines()[0]}
    样本数量：{X.shape[0]}
    特征数量：{X.shape[1]}
    目标类别：{target_names}
    特征示例：{feature_names[:5]}
    类别分布：
    {df['diagnosis'].value_counts()}
    """)
    
    # 数据可视化（论文加分项）
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='diagnosis', y=feature_names[0])
    plt.title(f"Distribution of {feature_names[0]} by Diagnosis")
    plt.savefig("results/feature_dist.png", dpi=300, bbox_inches='tight')
    
    return X, y, feature_names, target_names, df

# %% 3. 数据预处理（对应论文4.1数据预处理）
def preprocess_data(X, y, test_size=0.3):
    """数据预处理与划分"""
    print("\n" + "="*50)
    print("2. 数据预处理")
    print("="*50)
    
    # 标准化处理（需在论文中解释必要性）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集（需说明stratify的作用）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=test_size, 
        stratify=y,
        random_state=42
    )
    
    print(f"训练集大小：{X_train.shape[0]} ({1-test_size:.0%})")
    print(f"测试集大小：{X_test.shape[0]} ({test_size:.0%})")
    print("训练集类别比例：", np.bincount(y_train)/len(y_train))
    print("测试集类别比例：", np.bincount(y_test)/len(y_test))
    
    return X_train, X_test, y_train, y_test, scaler

# %% 4. 模型训练与评估（对应论文5.结果报道与分析）
def train_and_evaluate(X_train, X_test, y_train, y_test, target_names):
    """模型训练与评估"""
    print("\n" + "="*50)
    print("3. 模型训练与评估")
    print("="*50)
    
    # 确保结果目录存在
    Path("results").mkdir(exist_ok=True)
    
    # 4.1 基础模型训练（参数设置需在论文中说明）
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        metric='euclidean'
    )
    knn.fit(X_train, y_train)
    
    # 4.2 交叉验证（加分项）
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"\n交叉验证准确率：{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 4.3 测试集评估
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)[:, 1]
    
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 4.4 混淆矩阵可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ConfusionMatrixDisplay.from_estimator(
        knn, X_test, y_test,
        display_labels=target_names,
        cmap='Blues',
        ax=ax1
    )
    ax1.set_title("Confusion Matrix")
    
    # 4.5 ROC曲线（加分项）
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc='lower right')
    
    plt.savefig("results/model_performance.png", dpi=300, bbox_inches='tight')
    
    # 4.6 K值调优实验（加分项）
    k_values = range(1, 21)
    train_acc = []
    test_acc = []
    
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, knn_temp.predict(X_train)))
        test_acc.append(accuracy_score(y_test, knn_temp.predict(X_test)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_acc, label='Train Accuracy')
    plt.plot(k_values, test_acc, label='Test Accuracy')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. k Value')
    plt.xticks(k_values)
    plt.legend()
    plt.grid()
    plt.savefig("results/k_tuning.png", dpi=300, bbox_inches='tight')
    
    return knn

# %% 5. 主函数
def main():
    """主执行流程"""
    # 1. 数据加载与探索
    X, y, feature_names, target_names, df = load_and_explore_data()
    
    # 2. 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # 3. 模型训练与评估
    model = train_and_evaluate(X_train, X_test, y_train, y_test, target_names)
    
    # 4. 保存关键结果（加分项）
    with open("results/new/metrics.txt", "w") as f:
        f.write(str(classification_report(y_test, model.predict(X_test))))
    
    print("\n" + "="*50)
    print("程序执行完成！结果已保存至results/目录")
    print("="*50)

if __name__ == "__main__":
    main()

