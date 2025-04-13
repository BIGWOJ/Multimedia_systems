import numpy as np
import cv2

def run_length_encode(img):
    img_flatten = img.flatten()
    output = np.zeros(np.prod(img_flatten.shape) * 2, dtype=object)
    index = 0
    count = 1
    for i in range(1, len(img_flatten)):
        if img_flatten[i] == img_flatten[i - 1]:
            count += 1
        else:
            # new_pair = (count, img_flatten[i - 1])
            output[index] = (count, img_flatten[i - 1])
            index += 1
            count = 1

    return (img.shape, output[:index])

def run_length_decode(encoded_data):
    shape, data = encoded_data
    flat = []
    for count, value in data:
        flat.extend([value] * count)
    return np.array(flat, dtype=np.uint8).reshape(shape)

def byte_run_encode(img: np.ndarray):
    flat = img.flatten()
    encoded = []
    i = 0
    while i < len(flat):
        run_len = 1
        while i + run_len < len(flat) and flat[i] == flat[i + run_len] and run_len < 127:
            run_len += 1
        if run_len > 1:
            encoded.append((-(run_len - 1), flat[i]))
            i += run_len
        else:
            literals = [flat[i]]
            i += 1
            while i < len(flat) and (len(literals) < 127):
                if i + 1 < len(flat) and flat[i] == flat[i + 1]:
                    break
                literals.append(flat[i])
                i += 1
            encoded.append((len(literals) - 1, literals))

    return (img.shape, encoded)

def byte_run_decode(encoded_data):
    shape, data = encoded_data
    flat = []
    for length, value in data:
        if length < 0:
            flat.extend([value] * (1 - length))
        else:
            flat.extend(value)
    return np.array(flat, dtype=np.uint8).reshape(shape)

def compression_ratio(original, compressed):
    original_size = original.size
    compressed_size = 0
    for item in compressed:
        if isinstance(item[1], list):
            compressed_size += 1 + len(item[1])
        else:
            compressed_size += 2
    return original_size / compressed_size, (compressed_size / original_size) * 100

images = ['IMAGES/blueprint.png', 'IMAGES/document.png', 'IMAGES/mountains_panorama.png']
import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj,np.ndarray):
        size=obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

for img_path in images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = np.array(img, dtype=np.uint8)

    print(f"\nTesting image: {img_path}")

    rle_encoded = run_length_encode(img)
    # print(rle_encoded[0])
    print(rle_encoded[1])
    print(get_size(img))
    exit()
    rle_decoded = run_length_decode(rle_encoded)
    print("RLE correct:", np.array_equal(img, rle_decoded))
    ratio_rle, perc_rle = compression_ratio(img, rle_encoded[1])
    print(f"RLE compression ratio: {ratio_rle:.2f}, size = {perc_rle:.2f}%")

    # br_encoded = byte_run_encode(img)
    # br_decoded = byte_run_decode(br_encoded)
    # print("ByteRun correct:", np.array_equal(img, br_decoded))
    # ratio_br, perc_br = compression_ratio(img, br_encoded[1])
    # print(f"ByteRun compression ratio: {ratio_br:.2f}, size = {perc_br:.2f}%")
