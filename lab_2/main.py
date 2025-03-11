import numpy as np
import matplotlib.pyplot as plt
import cv2

def task_1():
    img_imread = plt.imread('IMG_INTRO/A3.png')

    print(img_imread.dtype)
    print(img_imread.shape)
    print(np.min(img_imread),np.max(img_imread))

    # img = plt.imread('IMG_INTRO/A2.jpg')
    #
    # print(img.dtype)
    # print(img.shape)
    # print(np.min(img),np.max(img))

    def img_to_uint8(img):
        if np.issubdtype(img.dtype, np.unsignedinteger):
            return img
        else:
            return (img * 255).astype('uint8')

    def img_to_float(img):
        if np.issubdtype(img.dtype, np.floating):
            return img
        else:
            return img / 255

    plt.imshow(img_imread)
    plt.show()

    R_color = img_imread[:,:,0]
    G_color = img_imread[:,:,1]
    B_color = img_imread[:,:,2]

    plt.imshow(R_color, cmap=plt.cm.gray)
    plt.show()

    img_gray = 0.2126 * R_color + 0.7152 * G_color + 0.0722 * B_color

    img_cv = cv2.imread('IMG_INTRO/A3.png')
    plt.imshow(img_cv)
    plt.show()

    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    plt.imshow(img_cv_rgb)
    plt.show()

    img_cv_bgr = cv2.cvtColor(img_cv_rgb, cv2.COLOR_RGB2BGR)
    plt.imshow(img_cv_bgr)
    plt.show()

def task_2(img):
    plt.subplot(3,3,1)
    plt.imshow(img)

    R_color = img[:,:,0]
    plt.subplot(3,3,2)
    plt.imshow(R_color, cmap=plt.cm.gray)

    G_color = img[:,:,1]
    B_color = img[:,:,2]
    img_y2 = 0.2126 * R_color + 0.7152 * G_color + 0.0722 * B_color
    plt.subplot(3,3,3)
    plt.imshow(img_y2, cmap=plt.cm.gray)

    plt.subplot(3,3,4)
    plt.imshow(R_color)

    plt.subplot(3, 3, 5)
    plt.imshow(G_color)

    plt.subplot(3, 3, 6)
    plt.imshow(B_color)

    plt.subplot(3, 3, 7)
    img_copy = img.copy()
    img_copy[:, :, 1] = 0
    img_copy[:, :, 2] = 0
    plt.imshow(img_copy)

    plt.subplot(3, 3, 8)
    img_copy = img.copy()
    img_copy[:, :, 0] = 0
    img_copy[:, :, 2] = 0
    plt.imshow(img_copy)

    plt.subplot(3, 3, 9)
    img_copy = img.copy()
    img_copy[:, :, 0] = 0
    img_copy[:, :, 1] = 0
    plt.imshow(img_copy)
    plt.show()

def task_3():
    pass




# task_1()
img = plt.imread('IMG_INTRO/B01.png')
# task_2(img)
task_3()