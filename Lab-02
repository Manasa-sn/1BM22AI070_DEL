import pandas as pd
import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0

def predict(X, weights):
    z = np.dot(X, weights)
    return np.array([step_function(z_i) for z_i in z])

def binary_output(y_prob):
    return y_prob  # Already binary from step function

def train_perceptron(X, y, learning_rate=0.01):
    X = np.insert(X, 0, 1, axis=1)
    weights = np.zeros(X.shape[1])
    flag = False
    while not flag:
        flag = True
        for i in range(len(X)):
            z = np.dot(X[i], weights)
            y_pred = step_function(z)
            error = y[i] - y_pred
            if (y[i] == 1 and y_pred == 0) or (y[i] == 0 and y_pred == 1):
                flag = False
                weights += learning_rate * error * X[i]
    return weights

def evaluate(X, y, weights):
    X = np.insert(X, 0, 1, axis=1)
    y_pred = predict(X, weights)
    accuracy = np.mean(y_pred == y)
    return accuracy

def classify_new_input(new_input, weights):
    new_input = np.array(new_input)
    new_input_with_bias = np.insert(new_input, 0, 1)
    y_prob = step_function(np.dot(new_input_with_bias, weights))
    return y_prob

# Load your data
data = pd.read_csv("data.csv")

y = data.iloc[:, 0].values
X = data.iloc[:, 1:].values

learning_rate = 0.5
weights = train_perceptron(X, y, learning_rate)

accuracy = evaluate(X, y, weights)

bias = weights[0]
feature_weights = weights[1:]

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Bias (x0): {bias}")
print(f"Weights: {feature_weights}")

new_input = [3.4, 5.7]
predicted_class = classify_new_input(new_input, weights)

print(f"Predicted class for the new input {new_input}: {predicted_class}")
