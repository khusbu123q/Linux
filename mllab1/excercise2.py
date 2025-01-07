import numpy as np
import matplotlib.pyplot as plt
def function(x):
   return 2*x+3
x=np.linspace(-100,100,100)
y=function(x)

plt.figure(figsize=(8,5))
plt.plot(x,y,label="y=2x+3",color="blue")


plt.xlabel("x")
plt.ylabel("y")
plt.title("graph of y=2x+3")

plt.show()

