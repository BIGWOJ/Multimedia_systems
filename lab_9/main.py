import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

##############################################################################
######   Konfiguracja       ##################################################
##############################################################################

kat = '.'  # katalog z plikami wideo
plik = "video.mp4"  # nazwa pliku
ile = 100  # ile klatek odtworzyć? <0 - całość
key_frame_counter = 4  # co która klatka ma być kluczowa i nie podlegać kompresji
plot_frames = np.array([30, 45])  # automatycznie wyrysuj wykresy
auto_pause_frames = np.array([25])  # automatycznie za pauzuj dla klatki
subsampling = "4:4:4"  # parametry dla chroma subsampling
dzielnik = 1  # dzielnik przy zapisie różnicy
wyswietlaj_kaltki = True  # czy program ma wyświetlać klatki
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

def run_length_decode(encoded_data):
    og_shape = encoded_data[:2]
    data = encoded_data[2:]
    decoded = []

    for i in range(0, len(data), 2):
        count = data[i]
        value = data[i + 1]
        decoded.extend([value] * count)

    return np.array(decoded).reshape(og_shape)

def compress_KeyFrame(Frame_class):
    KeyFrame = data()
    ## TO DO



    KeyFrame.Y = Frame_class.Y
    KeyFrame.Cb = Frame_class.Cb
    KeyFrame.Cr = Frame_class.Cr
    return KeyFrame


def decompress_KeyFrame(KeyFrame):
    Y = KeyFrame.Y
    Cb = KeyFrame.Cb
    Cr = KeyFrame.Cr
    ## TO DO
    frame_image = frame_layers_to_image(Y, Cr, Cb, subsampling)
    return frame_image


def compress_not_KeyFrame(Frame_class, KeyFrame, inne_paramerty_do_dopisania=None):
    Compress_data = data()
    ## TO DO

    frame_difference_y = Frame_class.Y - KeyFrame.Y
    frame_difference_cb = Frame_class.Cb - KeyFrame.Cb
    frame_difference_cr = Frame_class.Cr - KeyFrame.Cr

    Compress_data.Y = run_length_encode(frame_difference_y / 4)
    Compress_data.Cb = run_length_encode(frame_difference_cb / 4)
    Compress_data.Cr = run_length_encode(frame_difference_cr / 4)

    return Compress_data

def decompress_not_KeyFrame(Compress_data, KeyFrame, inne_paramerty_do_dopisania=None):
    Y = Compress_data.Y.astype(int)
    Cb = Compress_data.Cb.astype(int)
    Cr = Compress_data.Cr.astype(int)
    ## TO DO

    Y = run_length_decode(Y)*4 + KeyFrame.Y
    Cb = run_length_decode(Cb)*4 + KeyFrame.Cb
    Cr = run_length_decode(Cr)*4 + KeyFrame.Cr
    # Cb = Cb * 4 + KeyFrame.Cb
    # Cr = Cr * 4 + KeyFrame.Cr

    return frame_layers_to_image(Y, Cr, Cb, subsampling)


def plotDiffrence(ReferenceFrame, DecompressedFrame, ROI):
    # bardzo słaby i sztuczny przykład wykorzystania tej opcji
    # przerobić żeby porównanie było dokonywane w RGB nie YCrCb i/lub zastąpić innym porównaniem
    # ROI - Region of Insert współrzędne fragmentu który chcemy przybliżyć i ocenić w formacie [w1,w2,k1,k2]
    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(16, 5)

    axs[0].imshow(ReferenceFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]])
    axs[2].imshow(DecompressedFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]])
    diff = ReferenceFrame[ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(float) - DecompressedFrame[ROI[0]:ROI[1],
                                                                        ROI[2]:ROI[3]].astype(float)
    print(np.min(diff), np.max(diff))
    axs[1].imshow(diff, vmin=np.min(diff), vmax=np.max(diff))


##############################################################################
####     Głowna pętla programu      ##########################################
##############################################################################

cap = cv2.VideoCapture(os.path.join(kat, plik))

if ile < 0:
    ile = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('Normal Frame')
cv2.namedWindow('Decompressed Frame')

compression_information = np.zeros((3, ile))

for i in range(ile):
    ret, frame = cap.read()
    if wyswietlaj_kaltki:
        cv2.imshow('Normal Frame', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Frame_class = frame_image_to_class(frame, subsampling)
    if (i % key_frame_counter) == 0:  # pobieranie klatek kluczowych
        KeyFrame = compress_KeyFrame(Frame_class)
        cY = KeyFrame.Y
        cCb = KeyFrame.Cb
        cCr = KeyFrame.Cr
        Decompresed_Frame = decompress_KeyFrame(KeyFrame)
    else:  # kompresja
        Compress_data = compress_not_KeyFrame(Frame_class, KeyFrame)
        cY = Compress_data.Y
        cCb = Compress_data.Cb
        cCr = Compress_data.Cr
        Decompresed_Frame = decompress_not_KeyFrame(Compress_data, KeyFrame)

    compression_information[0, i] = (frame[:, :, 0].size - cY.size) / frame[:, :, 0].size
    compression_information[1, i] = (frame[:, :, 0].size - cCb.size) / frame[:, :, 0].size
    compression_information[2, i] = (frame[:, :, 0].size - cCr.size) / frame[:, :, 0].size
    if wyswietlaj_kaltki:
        cv2.imshow('Decompressed Frame', cv2.cvtColor(Decompresed_Frame, cv2.COLOR_YCrCb2BGR))

    if np.any(plot_frames == i):  # rysuj wykresy
        for r in ROI:
            plotDiffrence(frame, Decompresed_Frame, r)

    if np.any(auto_pause_frames == i):
        cv2.waitKey(-1)  # wait until any key is pressed

    k = cv2.waitKey(1) & 0xff

    if k == ord('q'):
        break
    elif k == ord('p'):
        cv2.waitKey(-1)  # wait until any key is pressed

plt.figure()
plt.plot(np.arange(0, ile), compression_information[0, :] * 100)
plt.plot(np.arange(0, ile), compression_information[1, :] * 100)
plt.plot(np.arange(0, ile), compression_information[2, :] * 100)
plt.title("File:{}, subsampling={}, divider={}, KeyFrame={} ".format(plik, subsampling, dzielnik, key_frame_counter))