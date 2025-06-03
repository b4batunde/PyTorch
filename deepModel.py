import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# Generate noisy data
X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32).unsqueeze(1)

# Train-test split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.8, random_state=42)

# Accuracy function
def accuracyFunction(true, pred):
    correct = torch.eq(true, pred).sum().item()
    accuracy = (correct / len(pred)) * 100
    return accuracy

# Deeper and Wider Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, tnsr: torch.Tensor) -> torch.Tensor:
        return self.net(tnsr)

model = Model()

# Loss and optimizer
lossFunction = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # Slightly smaller learning rate

# Training
epochs = 5000
for epoch in range(epochs):
    model.train()
    yPred = model(XTrain)
    loss = lossFunction(yPred, yTrain)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Optional progress display
    if epoch % 500 == 0:
        with torch.inference_mode():
            val_pred = torch.sigmoid(model(XTest))
            val_pred = torch.round(val_pred)
            acc = accuracyFunction(yTest, val_pred)
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={acc:.2f}%")

# Final Evaluation
with torch.inference_mode():
    predictions = model(XTest)
    predictions = torch.sigmoid(predictions)
    predictions = torch.round(predictions)

print(f"Final Accuracy: {accuracyFunction(yTest, predictions):.2f}%")

# Decision Boundary Plot
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.float32)

    with torch.inference_mode():
        preds = model(grid)
        preds = torch.sigmoid(preds)
        preds = preds.reshape(xx.shape)
    
    plt.contourf(xx, yy, preds, cmap="bwr", alpha=0.7)  # <- changed to "bwr"
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap="bwr", edgecolors='k', s=20)  # Larger points
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision Boundary (High Contrast)")
    plt.show()

# Plot the decision boundary
plot_decision_boundary(model, X.numpy(), y.numpy())
