import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# # path_ori = '疫情防控数据.csv'
# # data_ori = pd.read_csv(path_ori, encoding='GBK')
# # data2save = data_ori.head(100)['温度']
# # data2save.to_csv('temperature.csv')

path = 'weather_temperature_yilan.csv'
data = pd.read_csv(path, encoding='GBK')

# 数据清洗
for i in range(len(data)):
    if isinstance(data.iloc[i, 1], str):
        try:
            data.iloc[i, 1] = float(data.iloc[i, 1])
        except ValueError:
            data.iloc[i, 1] = float(data.iloc[i, 1][:4])
    elif pd.isna(data.iloc[i, 1]) and i != 0:
        data.iloc[i, 1] = float(data.iloc[i-1, 1])

data = data.values[:, 1].reshape(-1, 1)
data_rand = data * np.random.randn(1, 4)
data_rand = data_rand @ np.random.randn(4, 1)
data_all = np.concatenate((data, data_rand), axis=1)
print(data, data_rand)
pca = PCA(n_components=1)
data = pca.fit_transform(data_all)


# 归一化
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
print(data)


# 创建数据集函数
def create_dataset(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(data[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# 划分数据集和定义时间步长
train_days = 1500
testing_days = 500
time_step = 100
train_data = data[0:train_days]
test_data = data[train_days:train_days+testing_days]
X, y = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 维度及格式转换
X = X.reshape(X.shape[0], X.shape[1], 1)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
# lstm模型
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=1, batch_first=True)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        y_pred = self.fc(output[:, -1, :])
        return y_pred

# 初始化模型、损失函数和优化器
model = lstm()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100

# 训练模型
log_loss = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')
        log_loss.append([epoch, loss.detach()])

# 测试模型
model.eval()
with torch.no_grad():
    predicted = model(X_test)
    predicted = predicted.numpy().squeeze()

# 用于验证的反归一化
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 绘制实际值和预测值
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predicted, label='Predicted')
plt.title('Temperature Prediction')
plt.xlabel('Time Step')
plt.ylabel('Temperature')
plt.legend()
plt.show()
plt.savefig('weather pred.png')

# loss
log_loss = np.array(log_loss)
plt.plot(log_loss[:, 0], log_loss[:, 1])
plt.title('Loss curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig('loss.png')