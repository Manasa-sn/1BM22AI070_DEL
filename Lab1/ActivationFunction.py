import math
import numpy as np
import matplotlib.pyplot as plt

def activate(inputs, weights):
    h = sum(x * w for x, w in zip(inputs, weights))
    return [
        sigmoid(h),
        relu(h),
        tanh(h),
        linear(h),
        unit_step(h),
        sign(h)
    ]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    return np.tanh(x)

def linear(x):
    return x

def unit_step(x):
    return 1 if x > 0 else 0

def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

if __name__ == "__main__":
    inputs = [.5, .3, .2]
    weights = [.4, .7, .2]
    output = activate(inputs, weights)
    print("sigmoid: ", output[0])
    print("ReLu: ", output[1])
    print("tanh: ", output[2])
    print("linear: ", output[3])
    print("unit step: ", output[4])
    print("sign: ", output[5])

x_values = np.linspace(-10, 10, 400)
def plot_activation(activation, name):
  y_values = [activation(x) for x in x_values]
  plt.figure(figsize=(8, 5))
  plt.plot(x_values, y_values, label=name, color='r')
  plt.title(f"{name} Activation Function")
  plt.xlabel('Input')
  plt.ylabel("Output")
  plt.grid()
  plt.legend()
  plt.show()
  print()

plot_activation(sigmoid, "Sigmoid")
plot_activation(relu, "Relu")
plot_activation(tanh, 'tanh')
plot_activation(linear, "linear")
plot_activation(unit_step, "Unit Step")
plot_activation(sign, "Sign")
