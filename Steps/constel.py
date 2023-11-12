import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def plot_stft(folder_path):
    # Create a directory to store the STFT images
    output_folder = os.path.join(folder_path, "stft_plots")
    os.makedirs(output_folder, exist_ok=True)

    # Parameters for STFT
    n_fft = 2048
    hop_length = 512

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)

            # Load audio file
            y, sr = librosa.load(file_path)

            # Compute Short-Time Fourier Transform (STFT)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

            # Extract time and frequency information
            times = librosa.times_like(D)
            frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Find peaks in each time frame
            all_peaks = []
            for frame in range(D.shape[1]):
                magnitude = D[:, frame]
                peaks, _ = find_peaks(magnitude, prominence=0, distance=100)
                all_peaks.append(peaks)

            # Flatten the list of peak indices
            flattened_peaks = [peak for sublist in all_peaks for peak in sublist]

            # Plot the STFT with identified peaks
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.scatter(times[flattened_peaks], frequencies[flattened_peaks], color='r', s=5)
            plt.title('STFT with Peaks for {}'.format(filename))
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            #plt.savefig(os.path.join(output_folder, filename.replace(".mp3", "_stft_plot.png")))
            plt.show()

if __name__ == "__main__":
    folder_path = "/home/nr55/Desktop/Projects/SaSProj/Song_ss"
    plot_stft(folder_path)
