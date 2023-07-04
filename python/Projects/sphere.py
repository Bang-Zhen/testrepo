from vpython import *
import random

colors = [color.green, color.blue, color.red, color.orange, color.purple, color.cyan]

def colorChange():
    sleep(1)
    sphereA.color = colors[random.randint(0, len(colors)-1)]
    sleep(1)
    sphereB.color = colors[random.randint(0, len(colors)-1)]
    sleep(1)
    sphereA.color = colors[random.randint(0, len(colors)-1)]
    sleep(1)
    sphereB.color = colors[random.randint(0, len(colors)-1)]

sphereA = sphere(pos = vector(-1, 2, 0),
            size = vector(1, 1, 1),
            color = vector(15, 1, 1))

sphereB = sphere(pos = vector(1, -2, 0), 
            size = vector(3, 3, 3),
            color = vector(5, 10, 1))

for i in range(0, 10000):
    colorChange()