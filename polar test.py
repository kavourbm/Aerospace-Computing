from pygame.locals import *
import pygame as pg
import numpy as np

pg.init()

# Screen Variables
screen_width = 1400
screen_height = 1200

scalef = .01

# Colors
white = (255, 255, 255)
grey = (15, 15, 15)
black = (0, 0, 0)
teal = (0, 255, 255)

consolas = pg.font.SysFont('Consolas',15,True)

# Create Window
screen = pg.display.set_mode((screen_width, screen_height))
pg.display.set_caption('Prime Polars')

clock = pg.time.Clock()

numbers = np.array([0,1,2])
pnums = np.array([1,2])

locationp = np.array([[0*np.cos(0),0*np.sin(0)], [1*np.cos(1),1*np.sin(1)], [2*np.cos(2),2*np.sin(2)]])
locationo = np.array([[0,0]])

crashed = False
i = 2

while not crashed:
    for event in pg.event.get():
        if event.type == QUIT:
            crashed = True
        if event.type == KEYDOWN:
            if pg.key.get_pressed()[K_ESCAPE]:
                crashed = True
    screen.fill(black)
    i += 1
    numbers = np.append(numbers,i)
    tmp = 0
    for a in pnums:
        if a == 1:
            continue
        elif i%a == 0:
            tmp = 0
            locationo = np.vstack((locationo,[i*np.cos(i),i*np.sin(i)]))
            break
        else:
            tmp = 1
    if tmp == 1:
        locationp = np.vstack((locationp,[i*np.cos(i),i*np.sin(i)]))
        pnums = np.append(pnums,i)
    scalef = (numbers[-1])/(screen_height/2)*1.05
    for l in locationo:
        locx = (l[0]*1/scalef)+(screen_width/2)
        locy = -1*(l[1]*1/scalef)+(screen_height/2)
        pg.draw.circle(screen,grey,(locx,locy),1)
    for p in locationp:
        locx = (p[0]*1/scalef)+(screen_width/2)
        locy = -1*(p[1]*1/scalef)+(screen_height/2)
        pg.draw.circle(screen,teal,(locx,locy),1)
    screen.blit(consolas.render(str(i),True,white),(0,0))
    pg.display.update()
    clock.tick(60)
