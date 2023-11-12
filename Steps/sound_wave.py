import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_sound_wave(folder_path):
    # Create a directory to store the sound wave images
    output_folder = os.path.join(folder_path, "sound_waves")
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)

            # Load audio file
            y, sr = librosa.load(file_path)

            # Plot the sound wave
            plt.figure(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr)
            plt.title('Sound Wave for {}'.format(filename))
            plt.savefig(os.path.join(output_folder, filename.replace(".mp3", "_sound_wave.png")))
            plt.show()

if __name__ == "__main__":
    folder_path = "/home/nr55/Desktop/Projects/SaSProj/Song_ss"
    plot_sound_wave(folder_path)
