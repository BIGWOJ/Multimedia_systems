import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity
from PIL import Image

def water_mark(img,mask,alpha=0.25):
    assert (img.shape[0]==mask.shape[0]) and (img.shape[1]==mask.shape[1]), "Wrong size"
    if len(img.shape)<3:
        flag=True
        t_img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGBA)
    else:
        flag=False
        t_img=cv2.cvtColor(img,cv2.COLOR_RGB2RGBA)
    if (mask.dtype==bool):
        t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
    elif (mask.dtype==np.uint8):
        if len(mask.shape)<3:
            t_mask=cv2.cvtColor((mask).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
        else:
            t_mask=cv2.cvtColor((mask).astype(np.uint8),cv2.COLOR_RGB2RGBA)
    else:
        if len(mask.shape)<3:
            t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_GRAY2RGBA)
        else:
            t_mask=cv2.cvtColor((mask*255).astype(np.uint8),cv2.COLOR_RGB2RGBA)
    t_out=cv2.addWeighted(t_img,1,t_mask,alpha,0)
    if flag:
        out=cv2.cvtColor(t_out,cv2.COLOR_RGBA2GRAY)
    else:
        out=cv2.cvtColor(t_out,cv2.COLOR_RGBA2RGB)
    return out

def put_data(img,data,binary_mask=np.uint8(1)):
    assert img.dtype==np.uint8 , "img wrong data type"
    assert binary_mask.dtype==np.uint8, "binary_mask wrong data type"
    un_binary_mask=np.unpackbits(binary_mask)
    if data.dtype!=bool:
        unpacked_data=np.unpackbits(data)
    else:
        unpacked_data=data
    dataspace=img.shape[0]*img.shape[1]*np.sum(un_binary_mask)
    print("dataspace: ",dataspace)
    print("unpacked_data: ",unpacked_data.size)
    assert (dataspace>=unpacked_data.size) , "too much data"
    # Obliczenie ilości potrzebnych bitów
    total_bits_needed = img.shape[0] * img.shape[1] * np.sum(un_binary_mask)

    # Dopasowanie danych do wymaganej długości
    if unpacked_data.size < total_bits_needed:
        # Tworzymy nową tablicę z danymi + zerami
        padded_data = np.zeros(total_bits_needed, dtype=np.uint8)
        padded_data[:unpacked_data.size] = unpacked_data
        unpacked_data = padded_data
    else:
        # Albo obcinamy jeśli danych jest za dużo
        unpacked_data = unpacked_data[:total_bits_needed]

    # Reshape do odpowiedniego kształtu
    prepered_data = unpacked_data.reshape(img.shape[0], img.shape[1], np.sum(un_binary_mask)).astype(np.uint8)
    mask=np.full((img.shape[0],img.shape[1]),binary_mask)
    img=np.bitwise_and(img,np.invert(mask))
    bv=0
    for i,b in enumerate(un_binary_mask[::-1]):
        if b:
            temp=prepered_data[:,:,bv]
            temp=np.left_shift(temp,i)
            img=np.bitwise_or(img,temp)
            bv+=1
    return img

def pop_data(img, binary_mask=np.uint8(1), out_shape=None):
    un_binary_mask = np.unpackbits(binary_mask)
    num_bits = np.sum(un_binary_mask)
    data = np.zeros((img.shape[0], img.shape[1], num_bits), dtype=np.uint8)

    bv = 0
    for i, b in enumerate(un_binary_mask[::-1]):  # Od MSB do LSB
        if b:
            mask = np.full_like(img, 2 ** i)
            bit_plane = np.bitwise_and(img, mask)
            # Sprowadzenie bitu na pozycję LSB
            bit_plane = np.right_shift(bit_plane, i)
            data[:, :, bv] = bit_plane
            bv += 1

    if out_shape is not None:
        flat = data.flatten()
        total_bytes_needed = np.prod(out_shape)
        # Packujemy bity do bajtów
        packed = np.packbits(flat[:total_bytes_needed * 8])
        # Obcinamy do potrzebnej liczby bajtów
        packed = packed[:total_bytes_needed]
        recovered = packed.reshape(out_shape)
        return recovered

    return data

def calculate_metrics(original, modified):
    mse = np.mean((original - modified) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse)
    ssim = structural_similarity(original, modified, multichannel=True, channel_axis=2)

    return {'PSNR': psnr, 'SSIM': ssim}

def task_1(img):
    blue_channel = img[:, :, 0]
    with open("text.txt", "r") as f:
        text = f.read()

    binary_mask = np.uint8(1)
    text_data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    encoded = put_data(blue_channel, text_data, binary_mask)
    encoded_img = img.copy()
    encoded_img[:, :, 0] = encoded
    retrieved_bits = pop_data(encoded, binary_mask, out_shape=text_data.shape)
    retrieved_text = retrieved_bits.tobytes().decode('utf-8', errors='ignore')

    print("Odtworzony tekst:", retrieved_text)

def task_2(img, img_hidden):
    def hide_image(carrier_img, data_img):
        stego_img = np.copy(carrier_img)

        masks = {
            'R': np.uint8(0b00000011),  # 2 bits
            'G': np.uint8(0b00000011),  # 2 bits
            'B': np.uint8(0b00000111),  # 3 bits
        }

        for channel_idx, (channel_name, mask) in enumerate(masks.items()):
            channel_data = data_img[:, :, channel_idx]
            stego_img[:, :, channel_idx] = put_data(carrier_img[:, :, channel_idx], channel_data, mask)

        return stego_img

    def recover_image(stego_img, shape, dtype=np.uint8):
        recovered = np.zeros(shape, dtype=dtype)

        masks = {
            'R': np.uint8(0b00000011),  # 2 bits
            'G': np.uint8(0b00000011),  # 2 bits
            'B': np.uint8(0b00000111),  # 3 bits
        }

        for channel_idx, (channel_name, mask) in enumerate(masks.items()):
            recovered[:, :, channel_idx] = pop_data(stego_img[:, :, channel_idx], mask, out_shape=shape[:2])

        return recovered

    carrier_img = np.array(Image.open('images/img_1.png'))
    data_img = np.array(Image.open('images/img_2.png'))

    stego_img = hide_image(carrier_img, data_img)

    Image.fromarray(stego_img).save('stego_image.png')

    recovered_img = recover_image(stego_img, data_img.shape)[:, :, :3]
    plt.imshow(recovered_img)
    plt.show()
    print(recovered_img)

    Image.fromarray(recovered_img).save('recovered_image.png')

def task_4(img):
    mask = cv2.imread('images/binary.png', 0).astype(np.uint8)
    alpha_values = [0.1, 0.25, 0.5]

    for alpha in alpha_values:
        watermarked = water_mark(img, mask, alpha=alpha)
        psnr = calculate_metrics(img, watermarked)['PSNR']
        ssim = calculate_metrics(img, watermarked)['SSIM']

        cv2.imwrite(f'watermarked_alpha_{alpha}.png', watermarked)
        print(f'Alpha {alpha}: PSNR = {psnr:.2f}, SSIM = {ssim:.2f}')

img = cv2.imread('images/img_1.png')
img_hidden = cv2.imread('images/img_2.png')

task_1(img)
# task_2(None, None)
# task_4(img)

