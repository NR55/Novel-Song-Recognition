import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal

def plot_sound_wave_and_stft():

    filename="Eminem - Rap God"
    file_path = "Songs/"+filename+".mp3"

    # Load audio file
    y, sr = librosa.load(file_path)

    # Plot the sound wave
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Sound Wave for {}'.format(filename))
    plt.show()

    # Perform STFT
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot the STFT
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT for {}'.format(filename))
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
    plt.title('Constellation Map for {}'.format(filename))
    plt.xlabel('Time Index')
    plt.ylabel('Frequency (Hz)')
    plt.show()

if __name__ == "__main__":
    plot_sound_wave_and_stft()
