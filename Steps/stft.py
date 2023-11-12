import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_sound_wave_and_stft():

    filename="Ijazat"
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

if __name__ == "__main__":
    plot_sound_wave_and_stft()
