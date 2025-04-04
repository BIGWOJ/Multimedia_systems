import numpy as np
import matplotlib.pyplot as plt
from useful_modules import generate_figures
import soundfile as sf
from scipy.interpolate import interp1d

def quantize(data, bit, m=0, n=1):
    d = 2 ** bit - 1

    if not np.issubdtype(data.dtype, np.floating):
        DataF = data.astype(float)
    else:
        DataF = data

    DataF = (DataF - m) / (n - m)
    DataF = np.round(DataF * d) / d
    DataF = DataF * (n - m) + m

    return DataF.astype(data.dtype)

def decimation(data, factor, fs):
    new_fs = fs / factor
    return data[::factor], new_fs

def interpolation(data, fs, fs_new, method='linear'):
    n = len(data)
    duration = n / fs
    n_new = int(duration * fs_new)

    t = np.linspace(0, duration, n)
    t_new = np.linspace(0, duration, n_new)

    if method == 'linear':
        interpolated = interp1d(t, data)
    elif method == 'cubic':
        interpolated = interp1d(t, data, kind='cubic')

    data_new = interpolated(t_new)

    return data_new.astype(data.dtype)

def test_quantize(signal, fs):
    generate_figures(signal, fs)
    generate_figures(quantize(signal, 1), fs)
    generate_figures(quantize(signal, 2), fs)
    generate_figures(quantize(signal, 4), fs)
    generate_figures(quantize(signal, 8), fs)

def test_decimation(signal, fs):
    # generate_figures(signal, fs)
    # decimated_signal, decimated_fs = decimation(signal, 2, fs)
    # generate_figures(decimated_signal, decimated_fs)
    # decimated_signal, decimated_fs = decimation(signal, 4, fs)
    # generate_figures(decimated_signal, decimated_fs)
    # decimated_signal, decimated_fs = decimation(signal, 5, fs)
    # generate_figures(decimated_signal, decimated_fs, time_margin=[0, 0.002])
    # decimated_signal, decimated_fs = decimation(signal, 10, fs)
    # generate_figures(decimated_signal, decimated_fs, [0, 0.002])
    decimated_signal, decimated_fs = decimation(signal, 2, fs)
    generate_figures(decimated_signal, decimated_fs)

def test_interpolation(signal, fs):
    fs_new = 2000
    signal_lin = interpolation(signal, fs, fs_new, method='linear')
    signal_cubic = interpolation(signal, fs, fs_new, method='cubic')
    # generate_figures(signal, fs)
    generate_figures(signal_lin, fs_new, [0, 1])
    generate_figures(signal_cubic, fs_new, [0, 1])

def test_sing_files():
    import sounddevice as sd

    files = ['SING/sing_low1.wav', 'SING/sing_low2.wav',
             'SING/sing_medium1.wav', 'SING/sing_medium2.wav'
             'SING/sing_high1.wav', 'SING/sing_high2.wav']

    bits = [4, 8]
    decimation_steps = [4, 6, 10, 24]
    interpolation_fs = [4000, 8000, 11999, 16000, 16953]

    for file in files:
        print(f"\n===== Original sound of {file} =====")
        signal, fs = sf.read(file)
        sd.play(signal, fs)
        status = sd.wait()

        for bit in bits:
            print(f"\t=== Quantization {bit} bits ===")
            quantized_signal = quantize(signal, bit)
            sd.play(quantized_signal, fs)
            status = sd.wait()

        for step in decimation_steps:
            print(f"\t=== Decimation {step} steps ===")
            decimated_signal, decimated_fs = decimation(signal, step, fs)
            sd.play(decimated_signal, decimated_fs)
            status = sd.wait()

        for fs_new in interpolation_fs:
            print(f"\t=== Interpolation to {fs_new} Hz ===")
            for method in ['linear', 'cubic']:
                print(f"\t\t=== Method {method} ===")
                interpolated_signal = interpolation(signal, fs, fs_new, method)
                sd.play(interpolated_signal, fs_new)
                status = sd.wait()
def generate_report():
    from docx import Document
    from docx.shared import Inches
    from io import BytesIO
    from docx2pdf import convert
    from os import remove

    files = ['SIN/sin_60Hz.wav', 'SIN/sin_440Hz.wav', 'SIN/sin_8000Hz.wav', 'SIN/sin_combined.wav']
    bits = [4, 8, 16, 24]
    decimation_steps = [2, 4, 6, 10, 24]
    interpolation_fs = [2000, 4000, 8000, 11999, 16000, 16953, 24000, 41000]
    interpolation_methods = ['linear', 'cubic']
    time_margins = [[0, 0.05], [0, 0.006], [0, 0.0004], [0, 0.0085]]
    time_margins_interpolation_8000 = [[0, 1], [0, 1], [0, 1], [0, 0.001], [0, 0.01], [0, 0.005], [0, 0.0005], [0, 0.001]]

    document = Document()
    document.add_heading('Kwantyzacja i próbkowanie dźwięku oraz re-sampling\nWojciech Latos', 0)

    for time_margin_index, file in enumerate(files):
        print(file)
        time_margin = time_margins[time_margin_index]
        document.add_heading('Sygnał - {}'.format(file), 2)
        signal, fs = sf.read(file)

        document.add_heading('Sygnał oryginalny', 3)
        fig = generate_figures(signal, fs, time_margin, True)
        fig_memfile = BytesIO()
        fig.savefig(fig_memfile)
        fig_memfile.seek(0)
        document.add_picture(fig_memfile, width=Inches(7))
        fig_memfile.close()
        plt.close(fig)

        for bit in bits:
            document.add_heading('Kwantyzacja {} bit'.format(bit), 3)
            fig = generate_figures(quantize(signal, bit), fs, time_margin, True)
            fig_memfile = BytesIO()
            fig.savefig(fig_memfile)
            fig_memfile.seek(0)

            document.add_picture(fig_memfile, width=Inches(7))

            fig_memfile.close()
            plt.close(fig)

        for step in decimation_steps:
            if file.__contains__('8000'):
                if step in [4, 10]:
                    time_margin = np.array(time_margin) * 3
                document.add_heading('Decymacja {} kroki'.format(step), 3)
                decimated_signal, decimated_fs = decimation(signal, step, fs)
                fig = generate_figures(decimated_signal, decimated_fs, time_margin, True)
                fig_memfile = BytesIO()
                fig.savefig(fig_memfile)
                fig_memfile.seek(0)

                document.add_picture(fig_memfile, width=Inches(7))

                fig_memfile.close()
                plt.close(fig)

        for fs_new in interpolation_fs:
            document.add_heading('Interpolacja do {}Hz'.format(fs_new), 3)
            for method in interpolation_methods:
                if file.__contains__('8000'):
                    time_margin = time_margins_interpolation_8000[interpolation_fs.index(fs_new)]
                document.add_heading('Metoda {}'.format(method), 4)
                interpolated_signal = interpolation(signal, fs, fs_new, method)
                fig = generate_figures(interpolated_signal, fs_new, time_margin, generate_report=True)
                fig_memfile = BytesIO()
                fig.savefig(fig_memfile)
                fig_memfile.seek(0)

                document.add_picture(fig_memfile, width=Inches(7))

                fig_memfile.close()
                plt.close(fig)

    docx_path = 'report.docx'
    document.save(docx_path)
    convert(docx_path, 'report.pdf')
    remove(docx_path)

# signal, fs = sf.read('SIN/sin_8000Hz.wav')
# test_quantize(signal, fs)
# test_decimation(signal, fs)
# test_interpolation(signal, fs)
# decimated_signal, decimated_fs = decimation(signal, 24, fs)
# generate_figures(decimated_signal, decimated_fs)
# generate_report()

test_sing_files()
