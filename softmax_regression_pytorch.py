import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1) - 1  # Shift range to [-1, 1]
y = (X > 0).astype(int)  # Binary classification (0 or 1)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long).squeeze()

# Split data into training and testing sets
def train_test_split(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2)

# Define Softmax Regression Model
class SoftmaxRegressionSGD(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxRegressionSGD, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# Model, loss, and optimizer
model = SoftmaxRegressionSGD(input_dim=1, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Make predictions
y_pred_probs = model(X_test).detach()
y_pred = torch.argmax(y_pred_probs, axis=1)

# Plot results
plt.scatter(X_test.numpy(), y_test.numpy(), color='blue', label='Actual')
plt.scatter(X_test.numpy(), y_pred.numpy(), color='red', marker='x', label='Predicted')
plt.axvline(x=0, color='black', linestyle='--', label='Decision Boundary')  # Vertical line at x=0
plt.xlabel('X')
plt.ylabel('Class')
plt.legend()
plt.show()

# Print model parameters
for name, param in model.named_parameters():
    print(f'{name}: {param.data.numpy()}')
