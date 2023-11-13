import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_sound_wave():

    example_file = librosa.example('trumpet')

    y, sr = librosa.load(example_file)

    # Plot the sound wave
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Sound Wave for Example Song')
    plt.show()

if __name__ == "__main__":
    plot_sound_wave()

    plot_sound_wave()