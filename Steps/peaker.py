import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import fft, signal
import scipy

def plot_fft(folder_path):
    # Create a directory to store the FFT images
    output_folder = os.path.join(folder_path, "fft_plots")
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)

            # Load audio file
            y, sr = librosa.load(file_path)

            # Perform FFT
            fft_result = np.fft.fft(y)
            magnitude = np.abs(fft_result)
            frequency = np.fft.fftfreq(len(magnitude), d=1/sr)

            all_peaks, props = signal.find_peaks(magnitude)
            peaks, props = signal.find_peaks(magnitude, prominence=0, distance=10000)
            n_peaks = 15
            # Get the n_peaks largest peaks from the prominences
            # This is an argpartition
            # Useful explanation: https://kanoki.org/2020/01/14/find-k-smallest-and-largest-values-and-its-indices-in-a-numpy-array/
            largest_peaks_indices = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
            largest_peaks = peaks[largest_peaks_indices]
            plt.plot(frequency, magnitude, label="Spectrum")
            plt.scatter(frequency[largest_peaks], magnitude[largest_peaks], color="r", zorder=10, label="Constrained Peaks")
            plt.xlim(0, 3000)
            plt.show()

            # Plot the FFT
            plt.figure(figsize=(12, 4))
            plt.plot(frequency, magnitude)
            plt.xlim(0, 2000)
            plt.title('FFT for {}'.format(filename))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.savefig(os.path.join(output_folder, filename.replace(".mp3", "_fft_plot.png")))
            plt.show()

if __name__ == "__main__":
    folder_path = "../Songs"
    plot_fft(folder_path)