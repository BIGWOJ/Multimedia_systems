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
    print(dataspace)
    print(unpacked_data.size)
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

def pop_data(img,binary_mask=np.uint8(1),out_shape=None):
    un_binary_mask=np.unpackbits(binary_mask)
    data=np.zeros((img.shape[0],img.shape[1],np.sum(un_binary_mask))).astype(np.uint8)
    bv=0
    for i,b in enumerate(un_binary_mask[::-1]):
        if b:
            mask=np.full((img.shape[0],img.shape[1]),2**i)
            temp=np.bitwise_and(img,mask)
            data[:,:,bv]=temp[:,:].astype(np.uint8)
            bv+=1
    if out_shape!=None:
        tmp=np.packbits(data.flatten())
        tmp=tmp[:np.prod(out_shape)]
        data=tmp.reshape(out_shape)
    return data

def calculate_metrics(original, modified):
    mse = np.mean((original - modified) ** 2)
    psnr = 10 * np.log10((255 ** 2) / mse)
    ssim = structural_similarity(original, modified, multichannel=True, channel_axis=2)

    return psnr, ssim

def task_1(img):
    blue_channel = img[:, :, 0]
    with open("text.txt", "r") as file:
        text = file.read()

    binary_mask = np.uint8(1)
    text_data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    encoded = put_data(blue_channel, text_data, binary_mask)
    encoded_img = img.copy()
    encoded_img[:, :, 0] = encoded
    restored_bits = pop_data(encoded, binary_mask, out_shape=text_data.shape)
    restored_text = restored_bits.tobytes().decode('utf-8')

    print("Restored text:", restored_text)

def task_2_3(carrier_path, img_hidden_path):
    def recover_image(stego_img, shape, RG_mask, B_mask, dtype=np.uint8):
        recovered = np.zeros(shape, dtype=dtype)

        masks = {
            'R': RG_mask,  # 2 bits
            'G': RG_mask,  # 2 bits
            'B': B_mask,  # 3 bits
            # 'R': np.uint8(0b00000011),  # 2 bits
            # 'G': np.uint8(0b00000011),  # 2 bits
            # 'B': np.uint8(0b00000111),  # 3 bits
        }

        for channel_idx, (channel_name, mask) in enumerate(masks.items()):
            recovered[:, :, channel_idx] = pop_data(stego_img[:, :, channel_idx], mask, out_shape=shape[:2])

        return recovered

    # img_carrier = cv2.cvtColor(cv2.imread(carrier_path), cv2.COLOR_BGR2RGB)
    # img_hidden = cv2.cvtColor(cv2.imread(img_hidden_path), cv2.COLOR_BGR2RGB)
    #
    # R = img_carrier[:, :, 0].copy()
    # G = img_carrier[:, :, 1].copy()
    # B = img_carrier[:, :, 2].copy()
    # R2 = img_hidden[:, :, 0].copy()
    # G2 = img_hidden[:, :, 1].copy()
    # B2 = img_hidden[:, :, 2].copy()
    #
    # carrier_with_hidden = np.copy(img_carrier)
    # carrier_with_hidden[:, :, 0] = put_data(R, R2, binary_mask=np.uint8(0b00000011))
    # carrier_with_hidden[:, :, 1] = put_data(G, G2, binary_mask=np.uint8(0b00000011))
    # carrier_with_hidden[:, :, 2] = put_data(B, B2, binary_mask=np.uint8(0b00000111))
    #
    # recovered_img = recover_image(carrier_with_hidden, img_hidden.shape, dtype=img_hidden.dtype)
    #
    # plt.subplot(3,1,1)
    # plt.imshow(img_carrier)
    # plt.title('Carrier image')
    # plt.axis('off')
    #
    # plt.subplot(3,1,2)
    # plt.imshow(carrier_with_hidden)
    # plt.title('Image with hidden image')
    # plt.axis('off')
    #
    # plt.subplot(3,1,3)
    # plt.imshow(recovered_img)
    # plt.title('Hidden image')
    # plt.subplots_adjust(hspace=0.5)
    # plt.axis('off')
    # plt.show()


    # exit()
    RG_masks = [np.uint8(0b00000011), np.uint8(0b00000011), np.uint8(0b00000111), np.uint8(0b00001111), np.uint8(0b00011111), np.uint8(0b00111111)]
    B_masks = [np.uint8(0b00000111), np.uint8(0b00001111), np.uint8(0b00011111), np.uint8(0b00111111), np.uint8(0b01111111)]

    for RG_mask in RG_masks:
        for B_mask in B_masks:

            print(f'Using RG mask: {RG_mask:08b}, B mask: {B_mask:08b}')
            img_carrier = cv2.cvtColor(cv2.imread(carrier_path), cv2.COLOR_BGR2RGB)
            img_hidden = cv2.cvtColor(cv2.imread(img_hidden_path), cv2.COLOR_BGR2RGB)

            R = img_carrier[:, :, 0].copy()
            G = img_carrier[:, :, 1].copy()
            B = img_carrier[:, :, 2].copy()
            R2 = img_hidden[:, :, 0].copy()
            G2 = img_hidden[:, :, 1].copy()
            B2 = img_hidden[:, :, 2].copy()

            carrier_with_hidden = np.copy(img_carrier)
            carrier_with_hidden[:, :, 0] = put_data(R, R2, RG_mask)
            carrier_with_hidden[:, :, 1] = put_data(G, G2, RG_mask)
            carrier_with_hidden[:, :, 2] = put_data(B, B2, B_mask)

            recovered_img = recover_image(carrier_with_hidden, img_hidden.shape, RG_mask, B_mask, img_hidden.dtype)

            plt.subplot(3, 1, 1)
            plt.imshow(img_carrier)
            plt.title('Carrier image')
            plt.axis('off')

            plt.subplot(3, 1, 2)
            plt.imshow(carrier_with_hidden)
            plt.title('Image with hidden image')
            plt.axis('off')

            plt.subplot(3, 1, 3)
            plt.imshow(recovered_img)
            plt.title('Hidden image')
            plt.subplots_adjust(hspace=0.5)
            plt.axis('off')
            # plt.show()
            plt.savefig(f'images/RG_{RG_mask:08b}_B_{B_mask:08b}.png')


def task_4(img):
    mask = cv2.imread('images/binary.png', 0).astype(np.uint8)
    alpha_values = [0.1, 0.25, 0.5]

    for img_counter, alpha in enumerate(alpha_values):
        watermarked = water_mark(img, mask, alpha=alpha)
        watermarked = cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB)
        psnr, ssim = calculate_metrics(img, watermarked)

        plt.subplot(3, 1, img_counter + 1)
        plt.subplots_adjust(hspace=0.75)
        plt.imshow(watermarked)
        plt.title(f'Alpha:{alpha}, PSNR:{psnr:.2f}, SSIM:{ssim:.4f}')
    plt.show()

img_path = 'images/img_1.png'
img_hidden_path = 'images/img_2.png'
img = cv2.imread(img_path)
img_hidden = cv2.imread(img_hidden_path)

# task_1(img)
task_2_3(img_path, img_hidden_path)
# task_4(img)

