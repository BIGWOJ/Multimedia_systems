import matplotlib.pyplot as plt
import numpy as np
import cv2

def scale_nearest_neighbour(img, scale):
    height, width, RGB = img.shape
    new_width = np.ceil(width * scale).astype(int)
    new_height = np.ceil(height * scale).astype(int)

    X = np.linspace(0, width - 1, new_width).astype(int)
    Y = np.linspace(0, height - 1, new_height).astype(int)

    new_img = np.zeros((new_height, new_width, RGB), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            new_img[i, j, :] = img[Y[i], X[j], :]

    plt.subplot(1,2,1)
    plt.title("Oryginalny obraz")
    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.title(f"Obraz po skalowaniu o {scale}")
    plt.imshow(new_img)
    plt.show()

def scale_bilinear(img, scale):
    height, width, RGB = img.shape
    new_width = np.ceil(width * scale).astype(int)
    new_height = np.ceil(height * scale).astype(int)

    X = np.linspace(0, width - 1, new_width).astype(int)
    Y = np.linspace(0, height - 1, new_height).astype(int)

    new_img = np.zeros((new_height, new_width, RGB), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            xx = X[j]
            yy = Y[i]

            x1 = np.floor(xx).astype(int)
            x2 = np.ceil(xx).astype(int)

            y1 = np.floor(yy).astype(int)
            y2 = np.ceil(yy).astype(int)





img = cv2.imread('IMG_SMALL/SMALL_0001.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
scale = 2
scale_nearest_neighbour(img, scale)