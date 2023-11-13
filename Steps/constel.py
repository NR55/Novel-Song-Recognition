import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from IPython.display import Audio, display
import sounddevice as sd

def plot_sound_wave_and_stft():

    example_file = librosa.example('trumpet')

    y, sr = librosa.load(example_file)

    # Plot the sound wave
    plt.figure(figsize=(6, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Sound Wave for Example Song')
    plt.show()

    # Perform STFT
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot the STFT
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT for Example Song')
    plt.show()

    # Perform constellation map
    constellation_map = []
    stft = librosa.stft(y)
    frequencies = librosa.fft_frequencies(sr=sr)
    
    for time_idx, window in enumerate(stft.T):
        spectrum = np.abs(window)
        peaks, props = signal.find_peaks(spectrum, prominence=0, distance=200)
        n_peaks = min(5, len(peaks))  # Ensure n_peaks is within bounds
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])

    plt.scatter(*zip(*constellation_map))
    plt.title('Constellation Map for Example Song')
    plt.xlabel('Time Index')
    plt.ylabel('Frequency (Hz)')
    plt.show()

    sd.play(y, sr)
    sd.wait()


if __name__ == "__main__":
    plot_sound_wave_and_stft()