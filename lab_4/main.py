import matplotlib.pyplot as plt
import numpy as np
import cv2

def color_fit(pixel_color, color_palette):
    distances = np.linalg.norm(color_palette - pixel_color, axis=1)
    return color_palette[np.argmin(distances)]

def quantize_image(img, color_palette):
    # Dla wektora szarości zrobić, żeby reshape wtedy robić
    if color_palette.shape[1] != 3:
        color_palette.reshape(color_palette.shape[0], 3)
    img_normalized = img / 255.0
    out_img = np.zeros_like(img_normalized)
    for row in range(img_normalized.shape[0]):
        for col in range(img_normalized.shape[1]):
            out_img[row, col] = color_fit(img_normalized[row, col], color_palette)

    return (out_img * 255).astype(np.uint8)

pallet8 = np.array([
        [0.0, 0.0, 0.0,],
        [0.0, 0.0, 1.0,],
        [0.0, 1.0, 0.0,],
        [0.0, 1.0, 1.0,],
        [1.0, 0.0, 0.0,],
        [1.0, 0.0, 1.0,],
        [1.0, 1.0, 0.0,],
        [1.0, 1.0, 1.0,],
])

pallet16 =  np.array([
        [0.0, 0.0, 0.0,],
        [0.0, 1.0, 1.0,],
        [0.0, 0.0, 1.0,],
        [1.0, 0.0, 1.0,],
        [0.0, 0.5, 0.0,],
        [0.5, 0.5, 0.5,],
        [0.0, 1.0, 0.0,],
        [0.5, 0.0, 0.0,],
        [0.0, 0.0, 0.5,],
        [0.5, 0.5, 0.0,],
        [0.5, 0.0, 0.5,],
        [1.0, 0.0, 0.0,],
        [0.75, 0.75, 0.75,],
        [0.0, 0.5, 0.5,],
        [1.0, 1.0, 1.0,],
        [1.0, 1.0, 0.0,]
])

pallet_gray = np.linspace(0,1, 4)
print(pallet_gray.shape[1])
print(pallet16.shape)
# exit()
img_small = cv2.imread('IMG_SMALL/SMALL_0006.jpg')
img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

plt.imshow(img_small)
plt.show()

img_small_quantized = quantize_image(img_small, pallet_gray)
plt.imshow(img_small_quantized)
plt.show()
