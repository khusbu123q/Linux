import numpy as np

def hypo(theta, x):
    return np.dot(x, theta)

def cost(x, y, theta):
    n = len(y)
    amount = hypo(theta, x)
    error = amount - y
    cos = (1 / (2 * n)) * np.sum(error ** 2)
    return cos


x = np.array([1, 2])
y = np.array([8])
theta = np.array([0.5, 1.5])

print(cost(x, y, theta))
