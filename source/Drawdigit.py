import pygame 
from pygame.locals import K_SPACE
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
        pygame.draw.rect(screen, (192,192,192),(px,py,2,2))

    pygame.display.update()
    clock.tick(1000)

pygame.quit()

pixels = pixels.reshape(1, 28,28, 1)
pixels = pixels/255.0

pixels = pixels.transpose()

predictions = loaded_model.predict(pixels)[0]

max = 0
index = 0

for i, pred in enumerate(predictions):
    if pred > max:
        max = pred
        index = i

# print(predictions)

print("Prediction:", np.argmax(predictions), max)

from keras.datasets import mnist

(train_X, train_y), (val_X, val_y) = mnist.load_data()

maps = [pixels[0], train_X[0]]

for i in range(2):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(maps[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

# print(pixels.shape)


