import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import os
base_dir = os.path.dirname(__file__)

csv_path = os.path.join(
    base_dir,
    "student_performance_ml.csv"
) #tạo path cho file csv

df = pd.read_csv(csv_path) #import file vào code

print(df["passed"].value_counts()) #check số lượng mỗi label

X = df[[
    "study_hours",
    "attendance",
    "sleep_hours",
    "phone_time",
    "stress_level"
]].values

y = df["passed"].values 
'''
bước này đang làm việc khai báo, dán nhãn passed đại diện cho y là output
còn x là stack vector dựa theo như cái bảng
'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) #normalization

X_train = torch.tensor(X_train, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)

y_train = torch.tensor(y_train, dtype = torch.float32).view(-1,1)
y_test = torch.tensor(y_test, dtype = torch.float32).view(-1,1) 
#convert qua torch tensor

#chia tiếp train thành validation và train 2
X_val = X_train[:200]
y_val = y_train[:200]

X_train2 = X_train[200:]
y_train2 = y_train[200:]

train_losses = []
val_losses = []

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 4),
            nn.ReLU(),
            
            nn.Linear(4, 1)
        )
    def forward(self, x):
        return self.net(x)
model = DeepNet()

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)

for epoch in range(1000):
    pred = model(X_train2)
    loss = criterion(pred, y_train2)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred_val = model(X_val)

        loss_val = criterion(pred_val, y_val)
    train_losses.append(loss.item())
    val_losses.append(loss_val.item())
    if epoch%100 == 0:
        print(epoch, loss.item(), loss_val.item())
with torch.no_grad():
    pred = model(X_test)
    pred = torch.sigmoid(pred)
    pred_class = (pred > 0.7).float()

accuracy = (pred_class.squeeze()
            == y_test.squeeze()).float().mean()
print(accuracy)
y_true = y_test.squeeze().numpy()

y_pred = pred_class.squeeze().numpy()
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
cm = confusion_matrix(
    y_true, 
    y_pred
)
print('confusion matrix:', cm)

cl_rp = classification_report(
    y_true,
    y_pred
)
print('classification_report:', cl_rp)

precision = precision_score(
    y_true,
    y_pred
)
print('precision: ', precision)

recall = recall_score(
    y_true,
    y_pred
)

print('recall: ', recall)
f1 = f1_score(
    y_true,
    y_pred
)
print('f1', f1)
import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.plot(val_losses)

plt.legend([
    "train",
    "validation"
])
plt.show()