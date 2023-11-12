import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal

def plot_fft_with_stft(folder_path):
    # Create a directory to store the FFT and STFT images
    #output_folder = os.path.join(folder_path, "fft_stft_plots")
    #os.makedirs(output_folder, exist_ok=True)

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

            # Plot the FFT
            plt.figure(figsize=(12, 4))
            plt.plot(frequency, magnitude)
            plt.xlim(0, 2000)
            plt.title('FFT for {}'.format(filename))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            #plt.savefig(os.path.join(output_folder, filename.replace(".mp3", "_fft_plot.png")))
            plt.show()

            # Parameters for STFT
            window_length_seconds = 3
            window_length_samples = int(window_length_seconds * sr)
            window_length_samples += window_length_samples % 2

            # Perform a short time Fourier transform
            # frequencies and times are references for plotting/analysis later
            # the stft is a NxM matrix
            frequencies, times, stft = signal.stft(
                y, sr, nperseg=window_length_samples,
                nfft=window_length_samples, return_onesided=True
            )

            # Code segment for peak identification in STFT
            constellation_map = []

            for time_idx, window in enumerate(stft.T):
                # Spectrum is by default complex.
                # We want real values only
                spectrum = abs(window)
                # Find peaks - these correspond to interesting features
                # Note the distance - want an even spread across the spectrum
                peaks, props = signal.find_peaks(spectrum, prominence=0, distance=200)

                # Only want the most prominent peaks
                # With a maximum of 5 per time slice
                n_peaks = 5
                # Get the n_peaks largest peaks from the prominences
                largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
                for peak in peaks[largest_peaks]:
                    frequency = frequencies[peak]
                    constellation_map.append([time_idx, frequency])

            # Transform [(x, y), ...] into ([x1, x2...], [y1, y2...]) for plotting using zip
            plt.scatter(*zip(*constellation_map))
            plt.title('STFT Peaks for {}'.format(filename))
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            #plt.savefig(os.path.join(output_folder, filename.replace(".mp3", "_stft_plot.png")))
            plt.show()

if __name__ == "__main__":
    folder_path = "/home/nr55/Desktop/Projects/SaSProj/Song_ss"
    plot_fft_with_stft(folder_path)
