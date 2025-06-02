import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

weight = 0.7
bias = 0.3

X = torch.arange(0, 2, 0.02)
y = weight * X + bias

XTrain, yTrain = X[:80], y[:80]
XTest, yTest = X[80:], y[80:]

def plot_predictions(train_data=XTrain,
                     train_labels=yTrain,
                     test_data=XTest,
                     test_labels=yTest,
                     predictions=None):
  plt.figure(figsize=(10, 7))
  plt.scatter(train_data, train_labels, color = "blue", s = 4, label = "Training data")
  plt.scatter(test_data, test_labels, color = "green", s = 4, label = "Testing data")

  if predictions is not None:
    plt.scatter(test_data, predictions, color = "red", s = 4, label = "Predictions")

  plt.legend(prop={"size": 14})
  plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

model_0 = LinearRegressionModel()

lossFunction = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001) 

epochs = 10000

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_0.train()

    yPred = model_0(XTrain)
    loss = lossFunction(yPred, yTrain)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
      test_pred = model_0(XTest)
      test_loss = lossFunction(test_pred, yTest.type(torch.float)) 
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

model_0.eval()
with torch.inference_mode():
  y_preds = model_0(XTest)

plot_predictions(predictions = y_preds)