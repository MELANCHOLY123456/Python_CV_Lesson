import numpy as np
from numpy import ones, arange, array, hstack, dot
import matplotlib.pyplot as plt

m = 20

x0 = ones((m, 1))
x1 = arange(1, m+1).reshape(m, 1)
X = hstack((x0, x1))

Y = array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)

alpha = 0.01

def cost_function(theta, X, Y):
    diff = dot(X, theta) - Y
    return (1/(2*m)) * dot(diff.T, diff)

def gradient_function(theta, X, Y):
    diff = dot(X, theta) - Y
    return (1/m) * dot(X.T, diff)

def gradient_descent(X, Y, alpha):
    theta = array([1,1]).reshape(2,1)
    gradient = gradient_function(theta, X, Y)
    while not all(abs(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, Y)
    return theta

optimal = gradient_descent(X, Y, alpha)
print('optimal:', optimal)
print('cost function:', cost_function(optimal, X, Y)[0][0])

def plot(X, Y, theta):
    ax = plt.subplot(111)
    ax.scatter(X[:, 1], Y, s=30, c="red", marker="s")
    plt.xlabel("X")
    plt.ylabel("Y")
    x = arange(0, 21, 0.2)
    y = theta[0] + theta[1] * x
    ax.plot(x, y)
    plt.title(f"Linear Regression: y = {theta[0][0]:.2f} + {theta[1][0]:.2f}x")
    plt.grid(True)
    plt.show()

# 调用plot函数显示图像
plot(X, Y, optimal)