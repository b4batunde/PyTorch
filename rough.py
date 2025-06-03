import sklearn
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

X, y = make_circles(1000, noise = 0.03, random_state = 42)
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)


XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size = 0.8, random_state = 42)

def accuracyFunction(true, pred):
    correct = torch.eq(true, pred).sum().item()
    accuracy = (correct/len(pred)) * 100
    return accuracy

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.model(x)
    
model = Model()

lossFunction = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.01)  

epochs = 1000
for epoch in range(epochs):
    model.train()

    yTrainPred = model(XTrain).squeeze()
    trainLoss = lossFunction(yTrainPred, yTrain)
    optimizer.zero_grad()
    trainLoss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        yTestPred = model(XTest)
        testLoss = lossFunction(yTestPred, yTest)
        


model.eval()
with torch.inference_mode():
    prediction = model(XTest).squeeze()
    predictionProb = torch.sigmoid(prediction)  # <--- Sigmoid here
    predictionClass = torch.round(predictionProb)

testAccuracy = accuracyFunction(yTest, predictionClass)
print(f"Test Accuracy: {testAccuracy:.2f}%")