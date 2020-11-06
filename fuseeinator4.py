from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import *


g = 9.8
m = 1.2

mp = 0.8

Co = 0.15
K = 0.5
rho = 1.293
pi = 3.14159

r = 0.04
h = 0.07

k = (Co * K * rho * pi * r * sqrt((r*r) + (h*h)))

n = int(input("n = "))
dt = 1/n

def high(yo):
    i = 0
    high = []
    while i < n:
        print("[", i ,"/", n, "]")
        h = 0
        yp = yo[i]
        yr = yo[i]
        while yp > 0:
            yp = yr + dt*(-(k/m)*(yr**3/abs(yr))-g)
            yr = yp
            h = h + yr*dt
        high.append(h)
        i = i + 1
    return(high)

x = np.linspace(0, 1000, n)
y = high(x)

plt.plot(x, y)
plt.show()
