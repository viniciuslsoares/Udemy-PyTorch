import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wieghts = nn.Parameter(torch.randn(1,
                                                dtype=torch.float,
                                                requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1,
                                            dtype=torch.float,
                                            requires_grad=True))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wieghts * x + self.bias
    
    
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



weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# plot_predictions(X_train, y_train, X_test, y_test)

torch.manual_seed(42)

model_0 = LinearRegressionModel()

print(list(model_0.parameters()))
    
loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_0.train()
    
    #1.
    y_pred = model_0(X_train)
    
    #2.
    loss = loss_fn(y_pred, y_train)     #input, target
    
    #3.
    optimizer.zero_grad()
    
    #4.
    loss.backward()
    
    #5.
    optimizer.step()
    
    # Testing
    model_0.eval()
    with torch.inference_mode():
        #1.
        test_pred = model_0(X_test)
        
        #2.
        test_loss = loss_fn(test_pred, y_test)
        
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())
        
# with torch.inference_mode():
#     y_preds_new = model_0(X_test)

# plot_predictions(X_train, y_train, X_test, y_test, predctions=y_preds_new)
# plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves") 
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

from pathlib import Path

#1.
model_path = Path("models")
model_path.mkdir(parents=True, exist_ok=True)

#2.
model_name = "01_pytorch_model_0.pth"
model_save_path = model_path / model_name

#3.
print(model_save_path)
torch.save(obj=model_0.state_dict(),
        f=model_save_path)

loaded_model_0 = LinearRegressionModel()

loaded_model_0.load_state_dict(torch.load(f=model_save_path))
print(loaded_model_0.state_dict())