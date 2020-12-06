import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import *
import pygame
import time

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

def high(yo, tmi):
    t = 0
    h = 0
    yp = yo
    yr = yo
    while t <= tmi:
        yp = yr + dt*(-(k/m)*(yr**3/abs(yr))-g)
        yr = yp
        h = h + yr*dt
        t = t + dt
    return(h)


def plot3d():
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
    return("plot")

yo = int(input("vfp = "))

pygame.init()

L = int(input("lenght = "))
H = int(input("hight = "))

# vo = float(input("vO = "))
vo = 0



pygame.display.set_caption("Fuseeinator")
screen = pygame.display.set_mode((L,H))

background = pygame.image.load('assets/img/background.png')
backgroundS = pygame.image.load('assets/img/backgroundS.png')

S = pygame.image.load('assets/img/S.png')
S_rect = S.get_rect()
S_rect.x = L/2
S_rect.y = H/2.8

g_a = pygame.image.load('assets/img/g_a.png')
g_a_rect = g_a.get_rect()
g_a_rect.x = L/2
g_a_rect.y = H/2.8 + 80

class Simu:
    def __init__(self):
        self.rocket = Rocket()
        self.is_playing = False
        self.g_a = False
    
    def update(self, screen, t):
        if high(yo, t) >= 0:
            time.sleep(dt)
            y = H - (high(yo, t)/2 + 120/2)
            self.rocket.rect.y = y
            print(high(yo, t))
            screen.blit(backgroundS,(0, 0))
            screen.blit(backgroundS,(0, 0))
            screen.blit(simu.rocket.image, simu.rocket.rect)
        else:
            self.is_playing = False
            screen.blit(backgroundS,(0, 0))
            screen.blit(simu.rocket.image, simu.rocket.rect)
            

    def ga(self, screen):
        screen.blit(backgroundS,(0, 0))
        plot3d()
        simu.g_a = False



class Rocket(pygame.sprite.Sprite):
    def __init__(self):
        self.vel = vo
        x = L/2 - (40/2)
        y = H - 120/2
        self.image = pygame.image.load('assets/img/rocket.png')
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y



simu = Simu()
rocket = Rocket()

running = True

while running == True:

    pygame.display.flip()

    if simu.is_playing == True:
        simu.update(screen, t)
        t = t + dt
    if simu.g_a == True:
        simu.ga(screen)
    elif simu.is_playing == False:
        if simu.g_a == False:
            screen.blit(background,(0, 0))
            screen.blit(S, S_rect)
            screen.blit(g_a, g_a_rect)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            print("close py")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if S_rect.collidepoint(event.pos):
                simu.is_playing = True
                screen.blit(backgroundS,(0, 0))
                t = 0

            if g_a_rect.collidepoint(event.pos):
                simu.g_a = True
                screen.blit(backgroundS,(0, 0))
