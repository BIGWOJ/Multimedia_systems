import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

kat = '.'  # katalog z plikami wideo
plik = "video.mp4"  # nazwa pliku
ile = 15  # ile klatek odtworzyć? <0 - całość
key_frame_counter = 4  # co która klatka ma być kluczowa i nie podlegać kompresji
plot_frames = np.array([1, 10])  # automatycznie wyrysuj wykresy
auto_pause_frames = np.array([25])  # automatycznie za pauzuj dla klatki
# subsampling = "4:2:0"  # parametry dla chroma subsampling
dzielnik = 8  # dzielnik przy zapisie różnicy
wyswietlaj_kaltki = False  # czy program ma wyświetlać klatki
ROI = [[0, 100, 0, 100]]  # wyświetlane fragmenty (można podać kilka )


##############################################################################
####     Kompresja i dekompresja    ##########################################
##############################################################################
class data:
    def init(self):
        self.Y = None
        self.Cb = None
        self.Cr = None


def Chroma_subsampling(L, subsampling):
    # uzupełnić
    if subsampling == "4:2:2":
        L = L[:, ::2]
    elif subsampling == "4:2:0":
        L = L[::2, ::2]
    elif subsampling == "4:1:1":
        L = L[:, ::4]
    elif subsampling == "4:1:0":
        L = L[::2, ::4]
    elif subsampling == "4:4:0":
        L = L[::2, ::2]
    elif subsampling == "4:4:4":
        # No changes
        pass

    return L


def Chroma_resampling(L, subsampling):
    # uzupełnić
    if subsampling == "4:2:2":
        L = np.repeat(L, 2, axis=1)
    elif subsampling == "4:2:0":
        L = np.repeat(L, 2, axis=1)
        L = np.repeat(L, 2, axis=0)
    elif subsampling == "4:1:1":
        L = np.repeat(L, 4, axis=1)
    elif subsampling == "4:1:0":
        L = np.repeat(L, 4, axis=1)
        L = np.repeat(L, 2, axis=0)
    elif subsampling == "4:4:0":
        L = np.repeat(L, 2, axis=1)
        L = np.repeat(L, 2, axis=0)
    elif subsampling == "4:4:4":
        # No changes
        pass
    return L

def frame_image_to_class(frame, subsampling):
    Frame_class = data()
    Frame_class.Y = frame[:, :, 0].astype(int)
    Frame_class.Cb = Chroma_subsampling(frame[:, :, 2].astype(int), subsampling)
    Frame_class.Cr = Chroma_subsampling(frame[:, :, 1].astype(int), subsampling)
    return Frame_class


def frame_layers_to_image(Y, Cr, Cb, subsampling):
    Cb = Chroma_resampling(Cb, subsampling)
    Cr = Chroma_resampling(Cr, subsampling)
    return np.dstack([Y, Cr, Cb]).clip(0, 255).astype(np.uint8)

def run_length_encode(img):
    flat = np.array(img).flatten()
    encoded = [img.shape[0], img.shape[1]]
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

    return np.array(encoded)

def compress_KeyFrame(Frame_class):
    KeyFrame = data()
    KeyFrame.Y = Frame_class.Y.copy()
    KeyFrame.Cb = Frame_class.Cb.copy()
    KeyFrame.Cr = Frame_class.Cr.copy()
    return KeyFrame

def decompress_KeyFrame(KeyFrame, subsampling):
    Y = KeyFrame.Y
    Cb = KeyFrame.Cb
    Cr = KeyFrame.Cr
    return frame_layers_to_image(Y, Cr, Cb, subsampling)

def compress_not_KeyFrame(Frame_class, KeyFrame, inne_paramerty_do_dopisania=None):
    Compress_data = data()
    Compress_data.Y = (Frame_class.Y - KeyFrame.Y) // dzielnik
    Compress_data.Cb = (Frame_class.Cb - KeyFrame.Cb) // dzielnik
    Compress_data.Cr = (Frame_class.Cr - KeyFrame.Cr) // dzielnik
    return Compress_data

def decompress_not_KeyFrame(Compress_data, KeyFrame, subsampling):
    Y = Compress_data.Y * dzielnik + KeyFrame.Y
    Cb = Compress_data.Cb * dzielnik + KeyFrame.Cb
    Cr = Compress_data.Cr * dzielnik + KeyFrame.Cr
    return frame_layers_to_image(Y, Cr, Cb, subsampling)


def plotDiffrence(ReferenceFrame, DecompressedFrame, ROI, frame_counter):
    # Convert back to RGB for difference visualization
    RefRGB = cv2.cvtColor(ReferenceFrame, cv2.COLOR_YCrCb2RGB)
    DecRGB = cv2.cvtColor(DecompressedFrame, cv2.COLOR_YCrCb2RGB)

    r1, r2, c1, c2 = ROI
    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.suptitle(f'Frame {frame_counter}')
    fig.set_size_inches(16, 5)

    axs[0].imshow(RefRGB[r1:r2, c1:c2])
    axs[0].set_title("Original (RGB)")

    diff = RefRGB[r1:r2, c1:c2].astype(float) - DecRGB[r1:r2, c1:c2].astype(float)
    axs[1].imshow(np.abs(diff).astype(np.uint8))
    axs[1].set_title("Difference (RGB)")

    axs[2].imshow(DecRGB[r1:r2, c1:c2])
    axs[2].set_title("Decompressed (RGB)")
    # plt.show()

    print("RGB diff range:", np.min(diff), np.max(diff))

def generate_report(ile):
    from docx import Document
    from docx.shared import Inches
    from io import BytesIO
    from docx2pdf import convert
    from os import remove

    ##############################################################################
    ####     Głowna pętla programu      ##########################################
    ##############################################################################

    cap = cv2.VideoCapture(os.path.join(kat, plik))

    if ile < 0:
        ile = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # cv2.namedWindow('Normal Frame')
    # cv2.namedWindow('Decompressed Frame')

    compression_information = np.zeros((3, ile))
    compression_information_rle = np.zeros((3, ile))

    document = Document()
    document.add_heading('Kompresja video\nWojciech Latos', 0)

    for subsampling in ['4:4:4', '4:2:2', '4:4:0', '4:2:0', '4:1:1', '4:1:0']:
        for i in range(ile):
            ret, frame = cap.read()
            if wyswietlaj_kaltki:
                cv2.imshow('Normal Frame', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Frame_class = frame_image_to_class(frame, subsampling)
            if (i % key_frame_counter) == 0:  # pobieranie klatek kluczowych
                KeyFrame = compress_KeyFrame(Frame_class)
                cR = KeyFrame.Y
                cG = KeyFrame.Cb
                cB = KeyFrame.Cr
                Decompresed_Frame = decompress_KeyFrame(KeyFrame, subsampling)
            else:  # kompresja
                Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame)
                cR = Compress_data.Y
                cG = Compress_data.Cb
                cB = Compress_data.Cr
                Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame, subsampling)

            compression_information[0, i] = (frame[:, :, 0].size - cR.size) / frame[:, :, 0].size
            compression_information[1, i] = (frame[:, :, 1].size - cG.size) / frame[:, :, 1].size
            compression_information[2, i] = (frame[:, :, 2].size - cB.size) / frame[:, :, 2].size

            compression_information_rle[0, i] = (run_length_encode(frame[:, :, 0]).size - cR.size) / frame[:, :, 0].size
            compression_information_rle[1, i] = (run_length_encode(frame[:, :, 1]).size - cG.size) / frame[:, :, 1].size
            compression_information_rle[2, i] = (run_length_encode(frame[:, :, 2]).size - cB.size) / frame[:, :, 2].size
            if wyswietlaj_kaltki:
                cv2.imshow('Decompressed Frame', cv2.cvtColor(Decompresed_Frame, cv2.COLOR_RGB2BGR))

            if np.any(plot_frames == i):  # rysuj wykresy
                for r in ROI:
                    plotDiffrence(frame, Decompresed_Frame, r, i)
                    memfile = BytesIO()
                    plt.savefig(memfile)
                    memfile.seek(0)
                    document.add_picture(memfile, width=Inches(6))
                    memfile.close()
                    plt.close()

            # if np.any(auto_pause_frames == i):
            #     cv2.waitKey(-1)  # wait until any key is pressed
            #
            # k = cv2.waitKey(1) & 0xff
            #
            # if k == ord('q'):
            #     break
            # elif k == ord('p'):
            #     cv2.waitKey(-1)  # wait until any key is pressed


    plt.figure()
    plt.plot(np.arange(0, ile), compression_information[0, :] * 100, label='R')
    plt.plot(np.arange(0, ile), compression_information[1, :] * 100, label='G')
    plt.plot(np.arange(0, ile), compression_information[2, :] * 100, label='B')
    plt.title("File:{}, subsampling={}, divider={}, KeyFrame={} bez RLE".format(plik, subsampling, dzielnik, key_frame_counter))
    plt.legend()

    memfile = BytesIO()
    plt.savefig(memfile)
    memfile.seek(0)
    document.add_picture(memfile, width=Inches(6))
    memfile.close()
    plt.close()

    plt.figure()
    plt.plot(np.arange(0, ile), compression_information_rle[0, :] * 100, label='R')
    plt.plot(np.arange(0, ile), compression_information_rle[1, :] * 100, label='G')
    plt.plot(np.arange(0, ile), compression_information_rle[2, :] * 100, label='B')
    plt.title("File:{}, subsampling={}, divider={}, KeyFrame={} z RLE".format(plik, subsampling, dzielnik, key_frame_counter))
    plt.legend()

    memfile = BytesIO()
    plt.savefig(memfile)
    memfile.seek(0)
    document.add_picture(memfile, width=Inches(6))
    memfile.close()
    plt.close()

    docx_path = 'report.docx'
    document.save(docx_path)

generate_report(ile)