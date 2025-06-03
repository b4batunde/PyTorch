import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(42)

TV = np.random.uniform(5, 300, 100)
Radio = np.random.uniform(0, 50, 100)
Newspaper = np.random.uniform(0, 80, 100)
noise = np.random.normal(0, 2, 100)
Sales = 0.05 * TV + 0.1 * Radio + 0.02 * Newspaper + 5 + noise

df = pd.DataFrame({
    "TV": TV,
    "Radio": Radio,
    "Newspaper": Newspaper,
    "Sales": Sales
})

train_size = int(0.8 * len(df))
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

XTrain = train_data[["TV", "Radio", "Newspaper"]].values
yTrain = train_data["Sales"].values.reshape(-1, 1)
XTest = test_data[["TV", "Radio", "Newspaper"]].values
yTest = test_data["Sales"].values.reshape(-1, 1)

# -------------------------- Feature Scaling (Manual Standardization) --------------------------

# Standardize manually
X_mean = XTrain.mean(axis=0)
X_std = XTrain.std(axis=0)

XTrain = (XTrain - X_mean) / X_std
XTest = (XTest - X_mean) / X_std

# Convert to torch tensors
XTrain = torch.tensor(XTrain, dtype=torch.float32)
yTrain = torch.tensor(yTrain, dtype=torch.float32)
XTest = torch.tensor(XTest, dtype=torch.float32)
yTest = torch.tensor(yTest, dtype=torch.float32)

# -------------------------- Linear Regression Model --------------------------

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # 3 input features --> 1 output
        
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# Loss and Optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training Loop
epochs = 100000
loss_history = []
for epoch in range(epochs):
    model.train()
    y_pred = model(XTrain)
    loss = loss_fn(y_pred, yTrain)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    if epoch % 10000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_test = model(XTest)

# -------------------------- Plot Actual vs Predicted --------------------------

def plotPredictions(testLabels, prediction):
    plt.figure(figsize=(8,6))
    plt.scatter(testLabels.numpy(), prediction.numpy(), color="red", label="Predicted vs Actual")
    plt.plot([testLabels.min(), testLabels.max()], [testLabels.min(), testLabels.max()], 'k--', lw=2)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.legend()
    plt.title("Actual vs Predicted Sales")
    plt.show()

plotPredictions(yTest, y_pred_test)

# -------------------------- Plot Loss over Epochs --------------------------

plt.figure(figsize=(8,6))
plt.plot(range(epochs), loss_history, color="purple")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.show()

# -------------------------- Print Final Parameters --------------------------

print("Learned Weights:", model.linear.weight.data.numpy())
print("Learned Bias:", model.linear.bias.data.numpy())
