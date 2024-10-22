import numpy as np
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[1, 0],
              [0, 1],
              [0, 1],
              [1, 0]])
input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 2
np.random.seed(42)
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size)
bias_hidden = np.zeros((1, hidden_layer_size))

weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size)
bias_output = np.zeros((1, output_layer_size))

def feedforward(X):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = relu(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = softmax(output_input)
    return output

output = feedforward(X)

print("Output probabilities for each input:")
print(output)
predictions = np.argmax(output, axis=1)
print("\nPredictions:", predictions)
