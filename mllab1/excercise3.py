import numpy as np
import matplotlib.pyplot as plt
def function(x):
   return 2**x+3*x+4
x=np.linspace(-10,10,100)
y=function(x)

plt.figure(figsize=(8,5))
plt.plot(x,y,label="y=x2+3x+4",color="blue")


plt.xlabel("x")
plt.ylabel("y")
plt.title("graph of y=x2+3x+4")

plt.show()