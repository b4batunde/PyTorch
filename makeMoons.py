from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import numpy as np

X, y = make_moons(n_samples = 1000, noise = 0.2, random_state = 42)
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32).unsqueeze(dim = 1)

XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size = 0.9, random_state = 42)

def accuracy(tnsr1, tnsr2):
    correct = torch.eq(tnsr1, tnsr2).sum().item()
    correct = (correct/len(tnsr1)) * 100
    return correct

def plotLoss(epochVals, lossVals):
    plt.figure()
    plt.plot(epochVals, lossVals, color = "Orange", label = "Loss Value Overtime")
    plt.title("Loss Value Function")
    plt.legend()
    plt.show()

class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features = 2, out_features = 16),
            nn.Softplus(),
            nn.Linear(in_features = 16, out_features = 16),
            nn.Softplus(),
            nn.Linear(in_features = 16, out_features = 1)
        )
    def forward(self, tnsr : torch.Tensor) -> torch.Tensor:
        return self.model(tnsr)
    
model = BinaryClassificationModel()
lossFunction = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)

epochs = 500

epochValues = []
lossValues = []

for epoch in range(epochs):
    model.train()
    
    yTrainPred = model(XTrain)
    trainLoss = lossFunction(yTrainPred, yTrain)
    optimizer.zero_grad()
    trainLoss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        yTestPred = model(XTest)
        testLoss = lossFunction(yTestPred, yTest)
        epochValues.append(epoch)
        lossValues.append(testLoss)


model.eval()
with torch.inference_mode():
    predictions = model(XTest)
    predictions = torch.sigmoid(predictions)
    predictions = torch.round(predictions)





def plot_decision_boundary(model, X, y):
    model.eval()
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid).float()

    with torch.inference_mode():
        preds = model(grid_tensor)
        preds = preds.reshape(xx.shape)
        preds = preds.numpy()

    plt.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.6, cmap="coolwarm")

    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap="coolwarm", edgecolors='k')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


print(accuracy(yTest, predictions))
plot_decision_boundary(model, X.numpy(), y.numpy())

