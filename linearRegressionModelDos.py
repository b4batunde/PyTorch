import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

weight = 0.9
bias = 0.3

X = torch.rand(100)
y = weight * X + bias

XTrain, yTrain = X[:80], y[:80]
XTest, yTest = X[80:], y[80:]

def plotPrediction(trainData = XTrain, trainLabels = yTrain, testData = XTest, testLabels = yTest, prediction = None):
    plt.figure()
    plt.scatter(trainData, trainLabels, color = "Blue", s = 4, label = "Training Data")
    plt.scatter(testData, testLabels, color = "Green", s = 4, label = "Testing Data")
    if prediction is not None:
        plt.scatter(testData, prediction, color = "Red",  s = 4, label = "Prediction")
    plt.legend()
    plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
    
    def forward(self, tnsr : torch.Tensor) -> torch.Tensor:
        return self.weight * tnsr + self.bias
    
model = LinearRegressionModel()
    
lossFunction = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001) 
    
epochs = 10000

epochValues = []
lossValues = []

for epoch in range(epochs):
    yTrainPred = model(XTrain)
    loss = lossFunction(yTrainPred, yTrain)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        yTestPred = model(XTest)
        loss = lossFunction(yTestPred, yTest)
        epochValues.append(epoch)
        lossValues.append(loss.detach().numpy())
        if epoch % 1000 == 0:
            print(f"In epoch {epoch}, the loss is {loss}.")


plt.plot(epochValues, lossValues, color = "Orange", label = "Loss Value Overtime")
plt.legend()
plt.show()

model.eval()
with torch.inference_mode():
  yPred = model(XTest)
plotPrediction()
plotPrediction(prediction = yPred)

