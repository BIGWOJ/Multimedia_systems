import numpy as np
import cv2


def run_length_encode(img):
    flat = np.array(img).flatten()
    encoded = []
    count = 1

    for i in range(1, len(flat)):
        if flat[i] == flat[i - 1]:
            count += 1
        else:
            encoded.append(count)
            encoded.append(flat[i - 1])
            count = 1
    encoded.append(count)
    encoded.append(flat[-1])

    return img.shape, np.array(encoded)

def run_length_decode(encoded_data):
    og_shape, data = encoded_data
    decoded = []

    for i in range(0, len(data), 2):
        count = data[i]
        value = data[i + 1]
        decoded.extend([value] * count)

    return np.array(decoded).reshape(og_shape)

def count_repeats(data, start):
    value = data[start]
    count = 1

    for i in range(start + 1, len(data)):
        if data[i] == value:
            count += 1
            if count == 128:
                break
        else:
            break

    return count

def count_uniques(data, start):
    count = 1

    for i in range(start + 1, len(data)):
        if data[i] == data[i - 1]:
            break
        count += 1
        if count == 128:
            break

    return count

def byte_run_encode(img):
    flat = img.flatten()
    encoded = []
    i = 0
    while i < len(flat):
        if i + 1 < len(flat) and flat[i] == flat[i + 1]:
            repeat_count = count_repeats(flat, i)
            while repeat_count > 128:
                encoded.append(-127)
                encoded.append(flat[i])
                repeat_count -= 128
            encoded.append(-(repeat_count - 1))
            encoded.append(flat[i])
            i += repeat_count
        else:
            unique_count = count_uniques(flat, i)
            while unique_count > 128:
                encoded.append(127)
                encoded.extend(flat[i:i+128])
                i += 128
                unique_count -= 128
            encoded.append(unique_count - 1)
            encoded.extend(flat[i:i+unique_count])
            i += unique_count

    return img.shape, np.array(encoded)

def byte_run_decode(encoded_data):
    og_shape, encoded = encoded_data
    decoded = []
    i = 0
    while i < len(encoded):
        header = encoded[i]
        if header < 0:
            count = -header + 1
            value = encoded[i + 1]
            decoded.extend([value] * count)
            i += 2
        else:
            count = header + 1
            decoded.extend(encoded[i + 1:i + 1 + count])
            i += 1 + count

    return np.array(decoded).reshape(og_shape)

def compression_ratio(original, compressed):
    before_compression = get_size(original.flatten())
    after_compression = get_size(compressed)
    CR = before_compression / after_compression
    PR = after_compression / before_compression * 100

    return CR, PR

def get_size(obj, seen=None):
    import sys
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

def test():
    images = ['IMAGES/blueprint.png', 'IMAGES/document.png', 'IMAGES/mountains_panorama.png']

    for img_path in images:
        print(img_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(int)

        encoded = run_length_encode(img)
        decoded = run_length_decode(encoded)
        print("\tPoprany proces kodowania -> dekodowania:", np.array_equal(img, decoded))
        CR, PR = compression_ratio(img, encoded[1])
        print(f"\tRLE stopień kompresji: {CR:.2f}, czyli {PR:.2f}%")

        encoded = byte_run_encode(img)
        decoded = byte_run_decode(encoded)
        print("\tPoprany proces kodowania -> dekodowania ByteRun :", np.array_equal(img, decoded))
        CR, PR = compression_ratio(img, encoded)
        print(f"\tRLE stopień kompresji: {CR:.2f}, czyli {PR:.2f}%\n")

def generate_report():
    from docx import Document
    from docx.shared import Inches
    from io import BytesIO
    from docx2pdf import convert
    import matplotlib.pyplot as plt

    document = Document()
    document.add_heading('Kompresja bezstratna\nWojciech Latos', 0)

    imgs_path = ['IMAGES/blueprint.png', 'IMAGES/document.png', 'IMAGES/mountains_panorama.png']

    for img_path in imgs_path:
        print(img_path)
        img_imread = cv2.imread(img_path)
        img = cv2.cvtColor(img_imread, cv2.COLOR_RGB2GRAY).astype(int)

        plt.imshow(cv2.cvtColor(img_imread, cv2.COLOR_BGR2RGB))
        memfile = BytesIO()
        plt.savefig(memfile)
        memfile.seek(0)

        document.add_picture(memfile, width=Inches(6))
        plt.close()

        memfile.close()

        rle_encoded = run_length_encode(img)
        rle_decoded = run_length_decode(rle_encoded)
        document.add_paragraph('Poprawność kompresji/dekompresji: {}'.format(np.array_equal(img, rle_decoded)))
        CR, PR = compression_ratio(img, rle_encoded[1])

        document.add_paragraph('RLE: stopień kompresji: {:.2f}, czyli {:.2f}%'.format(CR, PR))

        br_encoded = byte_run_encode(img)
        br_decoded = byte_run_decode(br_encoded)
        document.add_paragraph('Poprawność kompresji/dekompresji: {}'.format(np.array_equal(img, br_decoded)))
        CR, PR = compression_ratio(img, br_encoded[1])
        document.add_paragraph('ByteRun: stopień kompresji: {:.2f}, czyli {:.2f}%'.format(CR, PR))

    docx_path = 'report.docx'
    document.save(docx_path)
    # convert(docx_path, 'report.pdf')

test()
# generate_report()