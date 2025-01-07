import numpy as np
import matplotlib.pyplot as plt
def gaussian_pdf(x,mean,sigma):
    y= 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*sigma**2))
    return y

mean=0
sigma=15

x=np.linspace(-100,100,100)
y=gaussian_pdf(x,mean,sigma)

plt.figure(figsize=(10,10))
plt.plot(x,y,label=f"gaussian pdf(mean={mean},sigma={sigma})")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Gaussian PDF")
plt.grid(True)
plt.legend()

plt.show()