# Gradient
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target  

num_classes = len(np.unique(y)) 
num_features = X.shape[1]
weights = np.random.randn(num_features, num_classes)
bias = np.zeros((1, num_classes))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def log_loss(y_true, y_pred):
    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1
    return -np.mean(np.sum(y_true_one_hot * np.log(y_pred + 1e-15), axis=1))

learning_rate = 0.01
num_epochs = 1000
losses = []  # List to store the loss values

for epoch in range(num_epochs):
    z = X.dot(weights) + bias
    y_pred = softmax(z)

    loss = log_loss(y, y_pred)
    losses.append(loss)  # Store the loss value
    grad_weights = X.T.dot(y_pred - np.eye(num_classes)[y]) / X.shape[0]
    grad_bias = np.mean(y_pred - np.eye(num_classes)[y], axis=0, keepdims=True)

    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

final_z = X.dot(weights) + bias
final_pred = softmax(final_z)
final_loss = log_loss(y, final_pred)
print(f'Final Log Loss: {final_loss:.4f}')

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss', color='blue')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()
