import pygame
from math import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib as cm

plt.clf()



z = 0

t = 0
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

v = 0

tp = 1.8

x = np.linspace(0, 1000, 4000)

y = [(V*V)/(2*(g + (Co*K*rho*((V*V)/3)*pi*r*(sqrt((r*r)+(h*h)))/m))) for V in x]
y2 = [((V*g)/((g + (V*V*k)/(3*m))*(g + (V*V*k)/(3*m)))) for V in x]
y4 = [((g-((4*V*V*g *(k/(3*m))) / (g+V*V*(k/(3*m))))) / ((g+V*V*(k/(3*m)))**2)) for V in x ]
fp = [(V + ((g + ((Co*K*rho*(V*V)*pi*r*(sqrt((r*r)+(h*h))))/m))*tp))/((tp/m)*(log((m + mp)/m))) for V in x]

plt.plot(x, y,label = "hmax ( m )")
plt.plot(x, fp, label = "force de prop ( J )")
plt.legend()
plt.show()
plt.savefig('img.png')
plt.plot(x, y2, label = "d hmax / d vfp ( m )")
plt.legend()
plt.show()
plt.plot(x, y4, label = "d^2 hmax / d vfp^2 ( m )")
plt.legend()
plt.show()


fig = plt.figure()

ax =  fig.add_subplot(111, projection='3d')

g = 9.8
cx = 0.15
r = 0.04
h = 0.07
k = cx * 0.5 * 1.292 * pi * r * (sqrt(r*r + h*h))
tp = 2

def f(x,y):
    a = (x**2)/(2*(g + x**2*k)/y)
    return(a)

X = np.linspace(0, 300, 200)
Y = np.linspace(0.01, 10, 200)
x, y = np.meshgrid(X, Y)
Z = f(x,y)

ax.plot_surface(x, y, Z, cmap = 'plasma', label = "hmax")
ax.set_xlabel("vfp (m/s)")
ax.set_ylabel("m (kg)")
ax.set_zlabel("hmax (m)")
plt.show()

fig2 = plt.figure()

ay =  fig2.add_subplot(111, projection='3d')

def p(u, i):
    b = ((((u*tp)/i)*np.log((i + (i/6))/i)) - (((g + ((((u*tp)/i)*np.log((i + i/6)/i))-(g*tp)) *k)/i)*tp))
    return(b)

X = np.linspace(0, 700, 200)
Y = np.linspace(0.5, 10, 200)
u, i = np.meshgrid(X, Y)
Z = p(u,i)

ay.plot_surface(u, i, Z, cmap = 'plasma', label = "vfp")
ay.set_xlabel("fp(N)")
ay.set_ylabel("m")
ay.set_zlabel("vfp(m/s)")
plt.show()

fig = plt.figure()

ax2 =  fig.add_subplot(111, projection='3d')

def j(x,y):
    a = (x*g)/(g + x**2 * (k/(3*y)))**2
    return(a)

X = np.linspace(0, 300, 200)
Y = np.linspace(0.01, 10, 200)
x, y = np.meshgrid(X, Y)
Z = j(x,y)

ax2.plot_surface(x, y, Z, cmap = 'plasma', label = "hmax")
ax2.set_xlabel("vfp (m/s)")
ax2.set_ylabel("m (kg)")
plt.show()

fig2 = plt.figure()

ay2 =  fig2.add_subplot(111, projection='3d')


def s(x,y):
    b = (x**4*k)/(2*y**2*(g + x**2*k/y))
    return(b)

X = np.linspace(0, 300, 200)
Y = np.linspace(0.01, 10, 200)
x, y = np.meshgrid(X, Y)
Z = s(x,y)

ay2.plot_surface(x, y, Z, cmap = 'plasma', label = "hmax")
ay2.set_xlabel("vfp (m/s)")
ay2.set_ylabel("m (kg)")
plt.show()


u = 10
da = 0.1

while ((u*g)/((g + (u*u*k)/(3*m))*(g + (u*u*k)/(3*m)))) > da:
    u = u + 0.0001

print(u)

a = sqrt(g/(k/m))

print(a)

print("x = n with n  in [", a, ",", u, "]")

vfp = float(input("vfp = "))

R = Co * K * rho * v * v * pi * r * (sqrt((r * r) + (h * h)))
Rm = Co * K * rho * ((vfp * vfp) / 3) * pi * r * (sqrt((r * r) + (h * h)))



dt = float(input("dt = (0.039 conseillé) "))

xh = np.linspace(0, vfp / (g + (Rm / m)), int(1/dt))
y = [((vfp*T)-((g + R/m) * (T*T) * 0.5)) for T in xh]

pygame.init()


class Simulation:
    def __init__(self):
        self.fusee = Fusee()
        self.pressed = {}


class Fusee:
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('assets/physics/fusee.png')
        self.rect = self.image.get_rect()
        self.rect.x = 180
        self.rect.y = 1080

    def startsimu(self):
        super().__init__()
        self.rect.y -= (v*0.717*dt)
        print(v)


pygame.display.set_caption("fuseeintaor")
screen = pygame.display.set_mode((400, 1060))

background = pygame.image.load('assets/physics/background.png')
simulation = Simulation()

clock = pygame.time.Clock()
FPS = 1/dt

simu = True
running = True


plt.plot(xh, y, label="h(t)")
plt.legend()
plt.show()

while running:
    screen.blit(background, (0, 0))
    screen.blit(simulation.fusee.image, simulation.fusee.rect)
    pygame.display.flip()

    if t < (vfp*(vfp/95) / (g + (Rm / m))):
        clock.tick(FPS)
        R = Co * K * rho * v * v * pi * r * (sqrt((r * r) + (h * h)))
        v = vfp - (g + (R / m)) * t
        z = z + v*dt
        simulation.fusee.startsimu()
        pygame.display.flip()
        t = t + dt

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
