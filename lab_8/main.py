#wymiary obrazu, żeby były podzielne przez 8
import numpy as np
import cv2
import scipy.fftpack

class ver1:
    Y=np.array([])
    Cb=np.array([])
    Cr=np.array([])
    ChromaRatio="4:4:4"
    QY=np.ones((8,8))
    QC=np.ones((8,8))
    shape=(0,0,3)

data1=ver1()
data1.shape=(1,1)

YCrCb=cv2.cvtColor(RGB,cv2.COLOR_RGB2YCrCb).astype(int)
RGB=cv2.cvtColor(YCrCb.astype(np.uint8),cv2.COLOR_YCrCb2RGB)

def CompressBlock(block,Q):
    ###
    return vector

def DecompressBlock(vector,Q):
    ###
    return block

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a.astype(float), axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a.astype(float), axis=0 , norm='ortho'), axis=1 , norm='ortho')

def zigzag(A):
    template= np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(A.shape)==1:
        B=np.zeros((8,8))
        for r in range(0,8):
            for c in range(0,8):
                B[r,c]=A[template[r,c]]
    else:
        B=np.zeros((64,))
        for r in range(0,8):
            for c in range(0,8):
                B[template[r,c]]=A[r,c]
    return B

## podział na bloki
# L - warstwa kompresowana
# S - wektor wyjściowy
def CompressLayer(L,Q):
    S=np.array([])
    for w in range(0,L.shape[0],8):
        for k in range(0,L.shape[1],8):
            block=L[w:(w+8),k:(k+8)]
            S=np.append(S, CompressBlock(block,Q))

## wyodrębnianie bloków z wektora
# L - warstwa o oczekiwanym rozmiarze
# S - długi wektor zawierający skompresowane dane
def DecompressLayer(S,Q):
    L= # zadeklaruj odpowiedniego rozmiaru macierzy
    for idx,i in enumerate(range(0,S.shape[0],64)):
        vector=S[i:(i+64)]
        m=L.shape[1]/8
        k=int((idx%m)*8)
        w=int((idx//m)*8)
        L[w:(w+8),k:(k+8)]=DecompressBlock(vector,Q)

def CompressJPEG(RGB,Ratio="4:4:4",QY=np.ones((8,8)),QC=np.ones((8,8))):
    # RGB -> YCrCb
    JPEG= ver1()
    # zapisać dane z wejścia do klasy
    # Tu chroma subsampling
    JPEG.Y=CompressLayer(JPEG.Y,JPEG.QY)
    JPEG.Cr=CompressLayer(JPEG.Cr,JPEG.QC)
    JPEG.Cb=CompressLayer(JPEG.Cb,JPEG.QC)
    # tu dochodzi kompresja bezstratna

    if Ratio == "4:2:2":
        pass
    else:  # defalut "4:4:4"
        pass

    return JPEG

def DecompressJPEG(JPEG):
    # dekompresja bezstratna
    Y=DecompressLayer(JPEG.Y,JPEG.QY)
    Cr=DecompressLayer(JPEG.Cr,JPEG.QC)
    Cb=DecompressLayer(JPEG.Cb,JPEG.QC)
    # Tu chroma resampling
    # tu rekonstrukcja obrazu
    # YCrCb -> RGB
    return RGB