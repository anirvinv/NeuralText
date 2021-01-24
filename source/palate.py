import pygame 
from pygame.locals import K_SPACE
import numpy as np
import tensorflow as tf

loaded_model = tf.keras.models.load_model('model')

SCREENWIDTH = 28
SCREENHEIGHT = 28


pygame.init()

screen = pygame.display.set_mode((SCREENWIDTH,SCREENHEIGHT))
screen.fill((0, 0,0))
pygame.display.set_caption("Draw a Number")
clock = pygame.time.Clock()

loop = True
while loop:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            loop = False

    keys = pygame.key.get_pressed()
    
    pixels = []
    
    if keys[K_SPACE]:
        for i in range(SCREENWIDTH):
            row = []
            for j in range(SCREENHEIGHT):
                row.append(screen.get_at((i, j))[0]) 
            pixels.append(row)

        pixels = np.array(pixels)
        loop = False

    px, py = pygame.mouse.get_pos()
    if pygame.mouse.get_pressed() == (1,0,0):
        pygame.draw.rect(screen, (255, 255, 255),(px,py,3,3))

    pygame.display.update()
    clock.tick(1000)

pygame.quit()

pixels = pixels.reshape(1, 28,28, 1)
pixels = pixels/255.0

prediction = loaded_model.predict(pixels)

print(prediction)

