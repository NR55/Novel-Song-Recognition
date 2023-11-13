import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_sound_wave_and_stft():

    example_file = librosa.example('trumpet')

    y, sr = librosa.load(example_file)

    # Plot the sound wave
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Sound Wave for Example Song')
    plt.show()

    # Perform STFT
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot the STFT
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT for Example Song')
    plt.show()

if __name__ == "__main__":
    plot_sound_wave_and_stft()