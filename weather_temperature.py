import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

path = 'weather_temperature_yilan.csv'
dataset = pd.read_csv(path)
print(dataset)

time_step = 10
train_days = 1500
testing_days = 500
n_epochs = 25
filter_on = 1
model_type = 1

# 滤波处理和数据切片以及归一化
if filter_on == 1:
    dataset['Temperature'] = medfilt(dataset['Temperature'], 3)
    dataset['Temperature'] = gaussian_filter1d(dataset['Temperature'], 1.2)

train_set = dataset[0:train_days].reset_index(drop=True)
test_set = dataset[train_days: train_days+testing_days].reset_index(drop=True)
training_set = train_set.iloc[:, 1:2].values
testing_set = test_set.iloc[:, 1:2].values

scaler = MinMaxScaler(feature_range = (0, 1))
training_set = scaler.fit_transform(training_set)
testing_set = scaler.fit_transform(testing_set)

#

def data_split(sequence, n_timestamp):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp
        if end_ix > len(sequence) - 1:
            break
        # i to end_ix as input
        # end_ix as target output
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


X_train, y_train = data_split(training_set, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test, y_test = data_split(testing_set, time_step)

# 模型构建
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=3, num_layers=1)
        self.fc = nn.Linear(3, 1)

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
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# 测试模型
model.eval()
with torch.no_grad():
    predicted = model(X_test)
    predicted = predicted.numpy().squeeze()

# 反归一化
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
y = scaler.inverse_transform(y_test.reshape(-1, 1))



# 展示
plt.figure(figsize=(8, 7))

plt.subplot(3, 1, 1)
plt.plot(dataset['Temperature'], color='black', linewidth=1, label='True value')
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("All data")

plt.subplot(3, 2, 3)
plt.plot(y_test, color='black', linewidth=1, label='True value')
plt.plot(y, color='red', linewidth=1, label='Predicted')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("Predicted data (n days)")

plt.subplot(3, 2, 4)
plt.plot(y_test[0:75], color='black', linewidth=1, label='True value')
plt.plot(y[0:75], color='red', label='Predicted')
plt.legend(frameon=False)
plt.ylabel("Temperature")
plt.xlabel("Day")
plt.title("Predicted data (first 75 days)")

plt.subplot(3, 3, 7)
plt.plot(epochs, loss, color='black')
plt.ylabel("Loss (MSE)")
plt.xlabel("Epoch")
plt.title("Training curve")

plt.subplot(3, 3, 8)
plt.plot(y_test_descaled - y_predicted_descaled, color='black')
plt.ylabel("Residual")
plt.xlabel("Day")
plt.title("Residual plot")

plt.subplot(3, 3, 9)
plt.scatter(y_predicted_descaled, y_test_descaled, s=2, color='black')
plt.ylabel("Y true")
plt.xlabel("Y predicted")
plt.title("Scatter plot")

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()

mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
r2 = r2_score(y_test_descaled, y_predicted_descaled)
print("mse=" + str(round(mse, 2)))
print("r2=" + str(round(r2, 2)))