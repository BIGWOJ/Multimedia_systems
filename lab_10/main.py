import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd


def compute_metrics(original, modified):
    original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    modified = cv2.cvtColor(modified, cv2.COLOR_RGB2GRAY)

    mse = np.mean((original - modified) ** 2)
    nmse = mse / (np.mean(modified ** 2))
    psnr = 10 * np.log10((255 ** 2) / mse)
    if_rating = np.sum((modified - original) ** 2) / (np.sum(modified * original))

    if mse == 0:
        if_ssim = 1.0
        if_sim = 1.0
    else:
        if_sim = np.corrcoef(original.flatten(), modified.flatten())[0][1]
        if_ssim = ssim(original, modified, data_range=modified.max() - modified.min())

    return {
        "MSE": mse,
        "NMSE": nmse,
        "PSNR": psnr,
        "IF_SIM": if_sim,
        "SSIM": if_ssim
    }


def compress_jpeg(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def blur_image(img, kernel_size):
    return cv2.blur(img, (kernel_size, kernel_size))


def noise_image(img, alpha=0.5, sigma=25):
    gauss = np.random.normal(0, sigma, img.shape).astype(np.int16)
    noisy = (img + alpha * gauss).clip(0,255).astype(np.uint8)
    return noisy



results = []

images_dir = 'images'
output_dir = 'modified'
os.makedirs(output_dir, exist_ok=True)

methods = ['JPEG', 'Blur', 'GaussNoise']
params = {
    'JPEG': [10, 30, 50, 70, 90],
    'Blur': [3, 5, 7, 9, 11],
    'GaussNoise': [0.1, 0.3, 0.5, 0.7, 1.0]
}

for image_file in os.listdir(images_dir):
    original_img = cv2.imread(os.path.join(images_dir, image_file))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    for method in methods:
        for param in params[method]:
            print(f"Processing {image_file} with {method} at param {param}")
            if method == 'JPEG':
                degraded_img = compress_jpeg(original_img, param)
            elif method == 'Blur':
                degraded_img = blur_image(original_img, param)
            elif method == 'GaussNoise':
                degraded_img = noise_image(original_img, alpha=param)

            metrics = compute_metrics(original_img, degraded_img)
            metrics.update({
                "Image": image_file,
                "Method": method,
                "Param": param
            })
            results.append(metrics)


            filename = f"{os.path.splitext(image_file)[0]}_{method}_{param}.png"
            cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR))


df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)