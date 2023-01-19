import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA, KernelPCA

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 查看数据与预测属性的相关度
'''fig = plt.figure(figsize=(14, 8))
abs(train_data.corr()['SalePrice']).sort_values(ascending=False).plot.bar()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()'''

# 查看某些属性的离群情况(不用全去离群，可保留噪声)
'''figure = plt.figure()
sns.pairplot(x_vars=['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF'],
             y_vars=['SalePrice'], data=train_data, dropna=True)
plt.show()'''

train_data = train_data.drop(test_data[(train_data['OverallQual'] < 5) &
                                       (train_data['SalePrice'] > 200000)].index)

train_data = train_data.drop(train_data[(train_data['GrLivArea'] > 4000) &
                                        (train_data['SalePrice'] < 300000)].index)

train_data = train_data.drop(train_data[(train_data['YearBuilt'] < 1900) &
                                        (train_data['SalePrice'] > 400000)].index)

train_data = train_data.drop(train_data[(train_data['TotalBsmtSF'] > 6000) &
                                        (train_data['SalePrice'] < 200000)].index)

'''figure = plt.figure()
sns.pairplot(x_vars=['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF'],
             y_vars=['SalePrice'], data=train_data, dropna=True)
plt.show()'''

# 利用4填补缺失值
# 合并
all_features = pd.concat((train_data.iloc[:, 1: -1], test_data.iloc[:, 1:]))

# 查看缺失情况
nan_index = ((all_features.isnull().sum() / len(all_features))).sort_values(ascending=False)
# print(nan_index)

# 填补none的
columns1 = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for column in columns1:
    all_features[column] = all_features[column].fillna('None')

# 填补0的
columns2 = ['MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars',
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

for column in columns2:
    all_features[column] = all_features[column].fillna(0)

# 填补众数的
columns3 = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
for column in columns3:
    all_features[column] = all_features[column].fillna(all_features[column].mode()[0])

# 填补邻居中位数
all_features["LotFrontage"] = all_features.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# 填补默认值
all_features["Functional"] = all_features["Functional"].fillna("Typ")

# 删掉无用属性  因为基本都是一样的类别
all_features = all_features.drop(['Utilities'], axis=1)

# print(all_features.isnull().sum().max())
# print(all_features)


# 标准化及ont-hot
num_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[num_features] = all_features[num_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

all_features = pd.get_dummies(all_features, dummy_na=True)
# print(all_features.isnull().sum().max())
# print(all_features)


# lasso查看属性重要性（为了特征组合） + PCA（组合后去相关性）
n_train = train_data.shape[0]

'''train_features = all_features[:n_train]
test_features = all_features[n_train:]
train_label = train_data.SalePrice

scaler = RobustScaler()
train_features_scaled = scaler.fit(train_features).transform(train_features)
train_label_log = np.log(train_label)

lasso = Lasso(alpha=0.001)
lasso.fit(train_features_scaled, train_label_log)
FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=all_features.columns)
FI_lasso.sort_values("Feature Importance", ascending=False)
FI_lasso[FI_lasso["Feature Importance"] != 0].sort_values("Feature Importance").plot(kind="barh", figsize=(15, 25))
plt.xticks(rotation=90)
plt.show()'''
# PCA前可先进行特征组合 具体可参照2

pca = PCA(n_components='mle')
all_features_pca = pca.fit_transform(all_features)
print(all_features.shape)
print(all_features_pca.shape)

# 模型做法1 根据特征选择线性或树形算法 选择其中表现最优的几个算法取平均 后应用stacking 具体见2



# 模型做法2 神经网络
