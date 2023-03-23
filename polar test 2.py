import pygame as pg
import numpy as np
import sys
import time
import threading

pg.init()

# Screen Variables
screen_width = 1400
screen_height = 1100

scalef = .01

# Colors
white = (255, 255, 255)
grey = (25, 25, 25)
black = (0, 0, 0)
teal = (0, 255, 255)

consolas = pg.font.SysFont('Consolas',15,True)

# Create Window
screen = pg.display.set_mode((screen_width, screen_height))
pg.display.set_caption('Prime Polars')

clock = pg.time.Clock()

locationp = np.array([[0*np.cos(0),0*np.sin(0)], [1*np.cos(1),1*np.sin(1)]])
locationo = np.empty((0,2))

i = 1
m = 1
q = 1
crashed = False
stop_flag = False
brange = 100000000
pnums = np.array([True]*(brange+1))
pnums[0:2] = False

start = time.perf_counter()

liste = np.arange(0,brange)
cos_vals = np.cos(liste)*liste
sin_vals = np.sin(liste)*liste

left = (brange-i)*((time.perf_counter() - start)/i)
left_str = "{:.0f}h {:.0f}m {:.0f}s".format(left // 3600, (left % 3600) // 60, left % 60)

start = time.perf_counter()
elapsed_time = 0
elapsed_time2 = 0
left = 0

TIMER_EVENT = pg.USEREVENT + 1
pg.time.set_timer(TIMER_EVENT, 200)

left_str = ""

def compute_values():
    global stop_flag, locationp, locationo, i, brange, pnums, start, elapsed_time
    while i < brange and not stop_flag:
        if pnums[i]:
            pnums[i*i::i] = False
        elapsed_time = time.perf_counter() - start
        i += 1

def append_pvalues():
    global stop_flag, locationp, locationo, i, m, brange, pnums, cos_vals, sin_vals, start, elapsed_time2
    while m < brange and not stop_flag:
        if pnums[m] and m < i:
            locationp = np.vstack((locationp,[cos_vals[m], sin_vals[m]]))
            m += 1
        elif m < i:
            m += 1
        elapsed_time2 = time.perf_counter() - start

def display_values():
    global stop_flag, locationp, locationo, i, m, screen, black, grey, teal, consolas, screen_width, screen_height, scalef, elapsed_time, elapsed_time2
    clock = pg.time.Clock()
    while not stop_flag:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                # signal separate thread to exit
                stop_flag = True
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if pg.key.get_pressed()[pg.K_ESCAPE]:
                    # signal separate thread to exit
                    stop_flag = True
                    pg.quit()
                    sys.exit()
        screen.fill(black)
        scalef = m/(screen_height/2)*1.05
        #for l in locationo:
        #    pg.draw.circle(screen,grey,((l[0]*1/scalef)+(screen_width/2),-1*(l[1]*1/scalef)+(screen_height/2)),1)
        #if elapsed_time2 > 120:
        for p in locationp:
            pg.draw.circle(screen,teal,((p[0]*1/scalef)+(screen_width/2),-1*(p[1]*1/scalef)+(screen_height/2)),1)
        screen.blit(consolas.render(f"Elapsed Time: {elapsed_time:.2f}s", True, white), (0, 0))
        screen.blit(consolas.render(f"i: {i:.0f}", True, white), (0, 15))
        screen.blit(consolas.render(f"Elapsed Time: {elapsed_time2:.2f}s", True, white), (0, 30))
        screen.blit(consolas.render(f"m: {m:.0f}", True, white), (0, 45))
        pg.display.update()
        clock.tick(10)

t1 = threading.Thread(target=compute_values)
t2 = threading.Thread(target=display_values)
t3 = threading.Thread(target=append_pvalues)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()

pg.quit()
sys.exit()