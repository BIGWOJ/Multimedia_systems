import cv2
import numpy as np

def run_length_encoding(img):
    x = [len(img.shape)]
    x += list(img.shape)
    og_shape = x[1:int(x[0] + 1)]
    img_1d = img.flatten()
    img_nd = img_1d.reshape(og_shape)

    # Zmienić z output = [] na np.zeros jak w pdfie, bo dodawanie appendem więcej pamięci niż zrobienie od razu na x2 większą
    # względem inputu
    # output = np.zeros(np.prod(img_1d.shape)*2)
    output = []
    count = 1
    index = 0
    for i in range(len(img_1d) - 1):
        if img_1d[i] == img_1d[i + 1]:
            count += 1
        else:
            output.append((count, img_1d[i]))
            index += 1
            count = 1

    return output, og_shape

def run_length_decode(encoded_img, og_shape):
    decoded_img = []
    for count, value in encoded_img:
        decoded_img.extend([value] * count)
    return np.array(decoded_img).reshape(og_shape)





images = ['IMAGES/blueprint.png', 'IMAGES/document.png', 'IMAGES/mountains_panorama.png']
img = cv2.imread(images[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

encoded_img, og_shape = run_length_encoding(img)
decoded_img = run_length_decode(encoded_img, og_shape)



# for img in images:
#     img = cv2.imread(img)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


