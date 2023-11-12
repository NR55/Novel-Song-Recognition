import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_sound_wave():

    filename="Ijazat"
    file_path = "Songs/"+filename+".mp3"

    # Load audio file
    y, sr = librosa.load(file_path)

    # Plot the sound wave
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Sound Wave for {}'.format(filename))
    plt.show()

if __name__ == "__main__":
    plot_sound_wave()
