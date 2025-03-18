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

    return new_img

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

            x_fraction = xx - x1
            y_fraction = yy - y1

            if x1 == x2:
                x2 = x1 + 1 if x1 + 1 < width else x1
            if y1 == y2:
                y2 = y1 + 1 if y1 + 1 < height else y1

            for color in range(RGB):
                R1 = (1 - x_fraction) * img[y1, x1, color] + x_fraction * img[y1, x2, color]
                R2 = (1 - x_fraction) * img[y2, x1, color] + x_fraction * img[y2, x2, color]
                new_img[i, j, color] = (1 - y_fraction) * R1 + y_fraction * R2

    return new_img

def test_scale_img(img, scale, method="nearest"):
    if method == "nearest":
        scaled_img = scale_nearest_neighbour(img, scale)
    elif method == "bilinear":
        scaled_img = scale_bilinear(img, scale)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Oryginalny obraz")
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title(f"Obraz po skalowaniu o {scale} metodą {method}")
    plt.imshow(scaled_img)
    plt.tight_layout()
    plt.show()

img = cv2.imread('IMG_SMALL/SMALL_0001.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
scale = 2
test_scale_img(img, scale, method="nearest")
test_scale_img(img, scale, method="bilinear")



