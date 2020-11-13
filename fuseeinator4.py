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

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

def high(yo, tm):
    i = 0
    j = 0
    hight = []
    high= []
    while j < n:
        yoj = yo[j][j]
        i = 0
        print(int((j/n)*100), "%")
        print("\n"*22)
        high = []
        while i < n:
            t = 0
            h = 0
            yp = yoj
            yr = yoj
            tmi = tm[i][1]
            while t <= tmi:
                yp = yr + dt*(-(k/m)*(yr**3/abs(yr))-g)
                yr = yp
                h = h + yr*dt
                t = t + dt
                if h < 0:
                    h = 0
            high.append(h)
            i = i + 1
        hight.append(high)
        j = j + 1
    hightF = np.array(hight)
    return(hightF)

X = np.linspace(0.0001, 400, n)
Y = np.linspace(0.0001, 38, n)

x, y = np.meshgrid(X, Y)

Z = high(x,y)

ax.plot_surface(y, x, Z, cmap = cm.plasma, linewidth=0, antialiased=True)

ax.set_xlabel('vfp')
ax.set_ylabel('t')
ax.set_zlabel('h')

plt.show()
