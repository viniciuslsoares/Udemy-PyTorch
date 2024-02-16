import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_predictions(train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    predctions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predctions is not None:
        plt.scatter(test_data, predctions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                    out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

def plot_loss(epoch_count, loss_values, test_loss_values):

    plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title("Training and test loss curves") 
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

# Data
weight = 0.7
bias = 0.3

# Range Values
start = 0
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
X[:10], y[:10]
# print(X[:10], y[:10])

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
# print(len(X_train), len(X_test))

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1.state_dict())

#Training 

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_1.parameters(),
    lr=0.01)

torch.manual_seed(42)

epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_1.train()
    
    y_pred = model_1(X_train)

    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    #testing
    
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        
        test_loss = loss_fn(test_pred, y_test)
    
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        
        print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
        print(model_1.state_dict())
        

model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)

model_name = "01_pytorch_model_1.pth"
model_save_path = model_path / model_name

print(f"Saving model to: {model_save_path}")
torch.save(obj=model_1.state_dict(),
        f=model_save_path)

new_model_1 = LinearRegressionModelV2()
new_model_1.load_state_dict(torch.load(f=model_save_path))
        
new_model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)
    # plot_predictions(X_train, y_train, X_test, y_test, predctions=y_preds.detach().numpy())
    
    new_y_preds = new_model_1(X_test)
    # print(y_preds == new_y_preds)
    
plot_loss(epoch_count, loss_values, test_loss_values)
# plot_predictions(X_train, y_train, X_test, y_test, predctions=y_preds.detach().numpy())

    
