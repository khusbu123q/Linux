import numpy as np
import matplotlib.pyplot as plt
def function(x):
    y=x**2
    return y
def derivative(x):
    return 2*x
x=np.linspace(-100,100,100)
y=function(x)

plt.figure(figsize=(8,5))
plt.plot(x,y,label="y=x2",color="blue")


plt.xlabel("x")
plt.ylabel("y")
plt.title("graph of y=x2")
# x=np.linspace(-100,100,100)
dy=derivative(x)

# plt.figure(figsize=(8,5))
plt.plot(x,dy,label="y=2x",color="orange")


# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("graph of y=2x")

plt.show()






plt.show()



