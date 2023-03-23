from pygame.locals import *
import pygame
import datetime
import math

pygame.init()

crashed = False
screen = pygame.display.set_mode((0,0))

clock = pygame.time.Clock()

#colors
black = (0,0,0)
white = (255,255,255)

consolas = pygame.font.SysFont('Consolas',15,True)

#get screen dimensions
screenX = pygame.display.get_surface().get_width()
screenY = pygame.display.get_surface().get_height()

posx = 250
posy = 100

rects = []

dirs = 6
size = 2
i = 1
dire = 6
primenumbers = [2]

def rainbow(num):
    if num > 255:
        num = num % 256
    if num < 85:
        if num < 42:
            return (255,255-(num*6),0)
        else:
            return (255,0,((num-42)*6))
    elif num > 170:
        if num < 213:
            return (0,255,255-((num-170)*6))
        else:
            return (((num-213)*6),255,0)
    else:
        if num < 128:
            return (255-((num-85)*6),0,255)
        else:
            return (0,((num-128)*6),255)

while not crashed:
    now = datetime.datetime.now()
    drift = now.hour/3
    for event in pygame.event.get():
        if event.type == QUIT:
            crashed = True
        if event.type == KEYDOWN:
            if pygame.key.get_pressed()[K_ESCAPE]:
                crashed = True
    screen.fill(black)
    newrect = [posx,posy,size,rainbow(i)]
    rects.append(newrect)
    for rect in rects:
        pygame.draw.rect(screen,rect[3],((rect[0]+drift,rect[1]),(rect[2],rect[2])))
    i += 1
    for a in primenumbers:
        if i == a:
            primecheck = True
            break
        elif i % a == 0:
            primecheck = False
            break
        primecheck = True
    if primecheck:
        if i not in primenumbers:
            primenumbers.append(i)
    if primecheck:
        dire += 1
        if dire > (dirs):
            dire -= dirs
    posx += size*math.cos(math.radians((dire/dirs)*360))
    posy += size*math.sin(math.radians((dire/dirs)*360))
    if posx > screenX:
        posx -= screenX
    if posy > screenY:
        posy -= screenY
    if posx < 0:
        posx += screenX
    if posy < 0:
        posy += screenY
    screen.blit(consolas.render(str(i),True,white),(0,0))
    pygame.display.update()
    clock.tick(400)