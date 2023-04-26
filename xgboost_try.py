import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 数据读取
data_train = pd.read_excel(os.path.join("".join(os.path.abspath('..').split('/')[:-1]), "train.xlsx")) 
data_test = pd.read_excel(os.path.join("".join(os.path.abspath('..').split('/')[:-1]), "test.xlsx")) 
print("Train data shape:", data_train.shape) 
print("Test data shape:", data_test.shape)

# 检查特征数量是否一致
if not data_train.columns.equals(data_test.columns):
    # 特征不一致，去除测试集中多余的特征或者往训练集中添加缺失的特征
    for col in data_test.columns:
        if col not in data_train.columns:
            data_test = data_test.drop(columns=col)
    for col in data_train.columns:
        if col not in data_test.columns:
            data_test[col] = np.nan
    data_test = data_test[data_train.columns]

# 数据预处理
data_train = data_train.fillna(method='pad')
data_test = data_test.fillna(method='pad')

# 处理训练集缺失值
start = 0
end = len(data_train) - 1
if data_train.iloc[:, -1].isna().any():
    gwd = data_train.iloc[:, -1].values
    while np.isnan(gwd[start]):
        start += 1
    while np.isnan(gwd[end]):
        end -= 1
    end += 1
data_train = data_train.iloc[start:end]
gwd = data_train.iloc[:, -1].values
index = np.array(list(range(len(gwd))))
y = gwd[np.logical_not(np.isnan(gwd))]
x = index[np.logical_not(np.isnan(gwd))]
lag = interp1d(x, y, kind='cubic')
data_train.iloc[:, -1] = lag(index)

# 归一化
data_train_temp = data_train.values.astype(float)
data_test_temp = data_test.values.astype(float)
window_length = 10
train_max_length = window_length * (len(data_train_temp) // window_length)
test_max_length = window_length * (len(data_test_temp) // window_length)
X_train_temp = (data_train_temp - np.min(data_train_temp, axis=0)) / (np.max(data_train_temp, axis=0) - np.min(data_train_temp, axis=0))
X_test_temp = (data_test_temp - np.min(data_train_temp, axis=0)) / (np.max(data_train_temp, axis=0) - np.min(data_train_temp, axis=0))
X_train_temp = X_train_temp[:train_max_length]
X_test_temp = X_test_temp[:test_max_length]

# 移动窗口法构建数据集
X_train = []
y_train = []
for i in range(window_length, len(X_train_temp)):
    X_train.append(X_train_temp[i - window_length:i])
    y_train.append(X_train_temp[i, -1])
X_train = np.array(X_train).reshape(-1, window_length, data_train_temp.shape[1])
y_train = np.array(y_train).reshape(-1)
X_train_2d = X_train.reshape(-1, window_length * data_train_temp.shape[1])
y_train_2d = y_train.reshape(-1)

# 训练模型
xgb = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.5, gamma=0.5, reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, random_state=2023, n_jobs=-1)
xgb.fit(X_train_2d, y_train_2d)

# 在训练集上评价模型
train_pred = xgb.predict(X_train_2d)
train_pred_inv = train_pred * (np.max(data_train_temp[:, -1]) - np.min(data_train_temp[:, -1])) + np.min(data_train_temp[:, -1])
y_train_inv = y_train[window_length - 1:] * (np.max(data_train_temp[:, -1]) - np.min(data_train_temp[:, -1])) + np.min(data_train_temp[:, -1])
print("训练集R2", r2_score(y_train_inv, train_pred_inv))
print("训练集MAE", mean_absolute_error(y_train_inv, train_pred_inv))
print("训练集RMSE", np.sqrt(mean_squared_error(y_train_inv, train_pred_inv)))
pd.DataFrame(np.concatenate([y_train_inv.reshape(-1, 1), train_pred_inv.reshape(-1, 1)], axis=-1), columns=["actual", "predicted"]).to_excel("XGBoost训练集预测对比.xlsx", index=False)
plt.plot(range(start + window_length, end + 1), y_train_inv, label="actual")
plt.plot(range(start + window_length, end), train_pred_inv, label="predicted")
plt.ylabel("y")
plt.xlabel("Month")
plt.legend()
plt.savefig("训练集预测结果.jpg")
plt.show()

# 在测试集上评价模型
X_test = []
for i in range(window_length, len(X_test_temp)):
    X_test.append(X_test_temp[i - window_length:i])
X_test = np.array(X_test).reshape(-1, window_length, data_train_temp.shape[1])
test_pred = xgb.predict(X_test.reshape(-1, window_length * data_train_temp.shape[1]))
test_label = data_test_temp[window_length:window_length + len(test_pred), -1]
test_pred_inv = test_pred * (np.max(data_train_temp[:, -1]) - np.min(data_train_temp[:, -1])) + np.min(data_train_temp[:, -1])
print("测试集R2", r2_score(test_label, test_pred_inv))
print("测试集MAE", mean_absolute_error(test_label, test_pred_inv))
print("测试集RMSE", np.sqrt(mean_squared_error(test_label, test_pred_inv)))
pd.DataFrame(np.concatenate([test_label.reshape(-1, 1), test_pred_inv.reshape(-1, 1)], axis=-1), columns=["actual", "predicted"]).to_excel("XGBoost测试集预测对比.xlsx", index=False)
plt.plot(range(window_length, window_length + len(test_label)), test_label, label="actual")
plt.plot(range(window_length, window_length + len(test_pred)), test_pred_inv, label="predicted")
plt.ylabel("y")
plt.xlabel("Month")
plt.legend()
plt.savefig("测试集预测结果.jpg")
plt.show()
