import math


def l2_norm(vector):
    return math.sqrt(sum(v**2 for v in vector))


def l1_norm(vector):
    return sum(abs(v) for v in vector)

# Example usage
vector = [3, -4, 2]
print(l1_norm(vector))
print(l2_norm(vector))