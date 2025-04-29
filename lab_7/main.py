import numpy as np

def A_law_encode(data):
    A = 87.6
    indexes = np.abs(data) < 1/A
    denominator = 1 + np.log(A)
    data[indexes] = np.sign(data[indexes]) * (A * np.abs(data[indexes]) / denominator)
    data[~indexes] = np.sign(data[~indexes]) * ((1 + np.log(A * np.abs(data[~indexes]))) / denominator)

    return data

def A_law_decode(encoded):
    A = 87.6
    indexes = np.abs(encoded) < 1 / (1 + np.log(A))
    encoded[indexes] = np.sign(encoded[indexes]) * (np.abs(encoded[indexes]) * (1 + np.log(A)) / A)
    encoded[~indexes] = np.sign(encoded[~indexes]) * (np.exp(np.abs(encoded[~indexes]) * (1 + np.log(A)) - 1) / A)

    return encoded

def mu_law_encode(data):
    mu = 255
    indexes = (-1 <= data) & (data <= 1)
    data[indexes] = np.sign(data[indexes]) * (np.log(1 + mu * np.abs(data[indexes])) / (np.log(1 + mu)))

    return data

def mu_law_decode(encoded):
    mu = 255
    indexes = (-1 <= encoded) & (encoded <= 1)
    encoded[indexes] = np.sign(encoded[indexes]) * (1 / mu) * ((1 + mu) ** np.abs(encoded[indexes]) -1)

    return encoded

def DPCM_encode(data, bit):
    y = np.zeros(data.shape)
    e = 0
    for i in range(0, data.shape[0]):
        y[i] = quantization(data[i] - e, bit)
        e += y[i]
    return y

def DPCM_decode(encoded):
    decoded = np.zeros(encoded.shape)
    e = 0
    for i in range(encoded.shape[0]):
        decoded[i] = e + encoded[i]
        e = decoded[i]

    return decoded

def DPCM_encode_prediction(x, bit, n):
    encoded = np.zeros(x.shape)
    reconstructed_signal = np.zeros(x.shape)
    e = 0
    for i in range(0, x.shape[0]):
        encoded[i] = quantization(x[i] - e, bit)
        reconstructed_signal[i] = encoded[i] + e
        if i > 0:
            e = np.mean(reconstructed_signal[max(0, i - n):i])
        else:
            e = 0

    return encoded

def DPCM_decode_prediction(encoded, n):
    decoded = np.zeros(encoded.shape)
    reconstructed_signal = np.zeros(encoded.shape)

    e = 0
    for i in range(encoded.shape[0]):
        decoded[i] = e + encoded[i]
        reconstructed_signal[i] = decoded[i]
        if i > 0:
            e = np.mean(reconstructed_signal[max(0, i - n):i])
        else:
            e = 0

    return decoded

def quantization(data, bit):
    levels = 2 ** bit
    step = 2 / (levels - 1)
    quantized = np.round((data + 1) / step) * step - 1

    return quantized

def generate_report():
    from docx import Document
    from docx.shared import Inches
    from io import BytesIO
    from docx2pdf import convert
    from os import remove
    import matplotlib.pyplot as plt

    x_ranges = [np.linspace(-1, 1, 1000), np.linspace(-0.9, -0.8, 1000), np.linspace(-0.01, 0.01, 1000)]

    document = Document()
    document.add_heading('Kompresja stratna audio\nWojciech Latos', 0)

    for x_range in x_ranges:
        encoded = A_law_encode(x_range.copy())
        encoded_quantized = quantization(encoded, 8)
        fig = plt.figure(figsize=(8, 10))
        plt.plot(x_range, encoded_quantized, label='Sygnal po kompresji a-law po kwantyzacji do 8-bitów')
        plt.plot(x_range, encoded, label='Sygnal po kompresji a-law bez kwantyzacji')

        encoded = mu_law_encode(x_range.copy())
        ecoded_quantized = quantization(encoded, 8)
        plt.plot(x_range, ecoded_quantized, label='Sygnal po kompresji mu-law po kwantyzacji do 8-bitów')
        plt.plot(x_range, encoded, label='Sygnal po kompresji mu-law bez kwantyzacji')
        plt.title("Krzywa kompresji")
        plt.xlabel("Wartość syngału wejściowego")
        plt.ylabel("Wartość sygnału wyjściowego")
        plt.legend(loc='upper left')

        fig.tight_layout()
        memfile = BytesIO()
        fig.savefig(memfile)
        memfile.seek(0)
        document.add_picture(memfile, width=Inches(6))
        memfile.close()
        plt.close(fig)

        fig = plt.figure(figsize=(8, 10))
        plt.plot(x_range, x_range, label='Sygnał oryginalny')
        plt.plot(x_range, A_law_decode(encoded_quantized.copy()), label='Sygnał po dekompresji z a-law (kwantyzacja 8-bitów)')
        plt.plot(x_range, mu_law_decode(ecoded_quantized.copy()), label='Sygnał po dekompresji z mu-law (kwantyzacja 8-bitów)')
        plt.plot(x_range, quantization(x_range.copy(), 8), label='Sygnał oryginalny po kwantyzacji do 8 bitów')
        plt.title("Krzywa po dekompresji")
        plt.xlabel("Wartość syngału wejściowego")
        plt.ylabel("Wartość sygnału wyjściowego")
        plt.legend(loc='upper left')

        fig.tight_layout()
        memfile = BytesIO()
        fig.savefig(memfile)
        memfile.seek(0)
        document.add_picture(memfile, width=Inches(6))
        memfile.close()
        plt.close(fig)

        document.add_page_break()

    for x_range in [np.linspace(-1, 1, 1000), np.linspace(-0.5, -0.25, 1000)]:
        signal = 0.9 * np.sin(np.pi * x_range * 4)
        fig = plt.figure(figsize=(8, 7))
        plt.suptitle("Przykład A kwantyzacja do 6 bitów")

        plt.subplot(5, 1, 1)
        plt.title("Sygnal oryginalny")
        plt.plot(x_range, signal)

        plt.subplot(5, 1, 2)
        plt.title("Kompresja A-law")
        plt.plot(x_range, A_law_decode(quantization(A_law_encode(signal.copy()), 6)), label='Kompresja A-law')

        plt.subplot(5, 1, 3)
        plt.title("Kompresja mu-law")
        plt.plot(x_range, mu_law_decode(quantization(mu_law_encode(signal.copy()), 6)), label='Kompresja mu-law')

        plt.subplot(5, 1, 4)
        plt.title("Kompresja DPCM bez predykcji")
        plt.plot(x_range, DPCM_decode(DPCM_encode(signal, 6)), label='Kompresja DPCM bez predykcji')

        plt.subplot(5, 1, 5)
        plt.title("Kompresja DPCM z predykcją")
        plt.plot(x_range, DPCM_decode_prediction(DPCM_encode_prediction(signal.copy(), 6, n=3), n=3), label='Kompresja DPCM z predykcją')

        plt.tight_layout()
        memfile = BytesIO()
        fig.savefig(memfile)
        memfile.seek(0)
        document.add_picture(memfile, width=Inches(6))
        memfile.close()
        plt.close(fig)

        document.add_page_break()

        fig = plt.figure(figsize=(12, 6))
        plt.title('Przykład B kwantyzacja do 6 bitów')
        plt.plot(x_range, signal, label='Sygnał oryginalny')
        plt.plot(x_range, A_law_decode(quantization(A_law_encode(signal.copy()), 6)), label='Sygnał po dekompresji z a-law')
        plt.plot(x_range, mu_law_decode(quantization(mu_law_encode(signal.copy()), 6)), label='Sygnał po dekompresji z mu-law')
        plt.plot(x_range, DPCM_decode(DPCM_encode(signal, 6)), label='Sygnał po dekompresji z DPCM bez predykcji')
        plt.plot(x_range, DPCM_decode_prediction(DPCM_encode_prediction(signal.copy(), 6, n=3), n=3), label='Sygnał po dekompresji z DPCM z predykcją')
        plt.legend(loc='upper left')

        plt.tight_layout()
        memfile = BytesIO()
        fig.savefig(memfile)
        memfile.seek(0)
        document.add_picture(memfile, width=Inches(6))
        memfile.close()
        plt.close(fig)

    document.add_heading("Opis metod", level=1)
    document.add_paragraph('A-law i mu-law to nieliniowe metody kompresji amplitudy sygnału audio, które zmieniają jego zakres dynamiczny przed kwantyzacją. '
                           'Dzięki temu uzyskuje się mniejszą ilość danych przy zachowaniu dobrej jakości dźwięku. '
                           'Ciche dźwięki są wzmocniane, a głośne – tłumione. '
                           'Kompresji podlega bezpośrednio wartość każdej próbki audio, przekształcana zgodnie z odpowiednim wzorem matematycznym. '
                           'DPCM natomiast nie koduje wartości bezwzględnych próbek, ale różnice między rzeczywistą wartością próbki a jej prognozą. '
                           'W moim przypadku prognoza została wykonana jako średnia trzech ostatnich próbek. Pozwala to znacznie ograniczyć ilość danych niezbędnych do zapisania sygnału, szczególnie wtedy, gdy zmiany między kolejnymi próbkami są niewielkie i przewidywalne.'
                           )

    document.add_heading("Jakość dźwięku po kompresji do 8 bitów", level=1)
    document.add_paragraph('Jakość dźwięku po kompresji do 8 bitów jest zbliżona do oryginalnego. Są słyszalne szumy, aczkolwiek są one niewielkie, niewarte odnotowania.')

    document.add_page_break()
    document.add_heading("Jakość dźwięku dla x bitów", level=1)
    table = document.add_table(rows=4, cols=7)
    table.style = 'Table Grid'
    cell_texts = [
        'Plik / ilość bitów', '7', '6', '5', '4', '3', '2',
        'sing_low1', 'Jakość bardzo dobra, brak artefaktów, czysty dźwięk, pełne odwzorowanie', 'Bardzo dobra jakość odwzorowania, czysty dźwięk, znikome zniekształcenia', 'Dźwięk słyszalnie zniekształcony, nadal dobre odwzorowanie, słyszalne szumy', 'Bardzo duże zniekształcenia, artefakty, znaczące trudności w rozpoznaniu', 'Bardzo duże zniekształcenia, daleki od oryginału, mocne szumy', 'Dźwięk najgorszy ze wszystkich próbek, ogromne szumy, niezrozumiały',
        'sing_medium1', 'Jak powyżej', 'Bardzo czysty dźwięk, minimalne zniekształcenia', 'Wyraźnie gorsze odwzorowanie, natomiast nadal do zrozumienia', 'Nadal możliwość zrozumienia, bardzo duże zniekształcenia dźwięku, znaczące szumy', 'Bardzo ciężki do zrozumienia, duże szumy', 'Praktycznie brak możliwości rozpoznania',
        'sing_high1', 'Jak powyżej', 'Bardzo czysty dźwięk, słyszalne zakłócenia/zniekształcenia wprzy górnych częstotliwościach', 'Najbardziej zniekształcony przy górnych częstotliwościach, najgorsze efekty względem low oraz medium, jednakże nadal do zrozumienia', 'Przy najwyższych częstotliwościach duże trudności ze zrozumieniem, ogólny zarys dźwięku zrozumiany, irytujące dźwięki, szumy', 'Ogromne zniekształcenia, ledwo do zrozumienia', 'Można powiedzieć, że tylko sam szum, brak możliwości interpretacji'
    ]
    document.add_paragraph('W skrócie można powiedzieć, że im mniej bitów, tym gorsza jakość dźwięku. Im wyższe częstotliwości dźwięku tym większe zniekształcenia, szybciej pojawiają się zakłócenia względem innych próbek.')

    for row in range(len(table.rows)):
        for col in range(len(table.columns)):
            cell = table.cell(row, col)
            cell.text = cell_texts[row * len(table.columns) + col]

    docx_path = 'report.docx'
    document.save(docx_path)
    convert(docx_path, 'report.pdf')
    remove(docx_path)


generate_report()
