import numpy as np
import matplotlib.pyplot as plt

def a():
    R=np.random.rand(5,5)
    A=np.zeros(R.shape)
    B=np.zeros(R.shape)
    C=np.zeros(R.shape)

    idx=R<0.25
    A[idx]=1 # <-
    B[idx]+=0.25 # <-
    C[idx]=2*R[idx]-0.25 # <-
    C[np.logical_not(idx)]=4*R[np.logical_not(idx)]-0.5 # <-
    print(R)
    print(A)
    print(B)
    print(C)

def A_law_encode(data):
    A = 87.6
    indexes = np.abs(data) < 1/A
    denominator = 1 + np.log(A)
    data[indexes] = np.sign(data[indexes]) * (A * np.abs(data[indexes]) / denominator)
    data[~indexes] = np.sign(data[~indexes]) * (1 + np.log(A * np.abs(data[~indexes])) / denominator)

    return data

def A_law_decode(encoded):
    A = 87.6
    indexes = np.abs(encoded) < 1/(1 + np.log(A))
    encoded[indexes] = np.sign(encoded[indexes]) * (np.abs(encoded[indexes]) * (1 + np.log(A)) / A)
    encoded[~indexes] = np.sign(encoded[~indexes]) * (np.exp(np.abs(encoded[~indexes]) * (1 + np.log(A)) - 1) / A)

    return encoded

def mu_law_encode(data):
    mu = 255
    indexes = -1 <= data <= 1
    data[indexes] = np.log(1 + mu * np.abs(data[indexes])) / (np.log(1 + mu))

    return data

def mu_law_decode(encoded):
    mu = 255
    indexes = -1 <= encoded <= 1
    encoded[indexes] = np.sign(encoded[indexes]) * (1 / mu) * ((1 + mu) ** np.abs(encoded[indexes]) -1)

    return encoded

def DPCM_encode(data, bit):
    y = np.zeros(x.shape)
    e = 0
    for i in range(0, x.shape[0]):
        y[i] = kwant(x[i] - e, bit)
        e += y[i]
    return y

plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
x=np.linspace(-1,1,1000)
y=0.9*np.sin(np.pi*x*4)
plt.plot(x, y)

# plt.subplot(2, 1, 2)
# y_encoded = A_law_encode(y)
# y_encoded = np.round((y_encoded - y_encoded.min()) / (y_encoded.max() - y_encoded.min()) * (2**8 - 1)) / (2**8 - 1) * (y_encoded.max() - y_encoded.min()) + y_encoded.min()
# plt.plot(x, y_encoded)
plt.show()


# a()