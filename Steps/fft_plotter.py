import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def process_and_plot_mp3_folder(folder_path):
    # Create a directory to store the spectrogram images
    output_folder = os.path.join(folder_path, "spectrograms")
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)

            # Load audio file
            y, sr = librosa.load(file_path)

            # Compute Short-Time Fourier Transform (STFT)
            D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

            # Plot the spectrogram
            plt.figure(figsize=(12, 8))
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram for {}'.format(filename))
            plt.savefig(os.path.join(output_folder, filename.replace(".mp3", "_spectrogram.png")))
            plt.show()
            #plt.close()

if __name__ == "__main__":
    folder_path = "../Songs"
    process_and_plot_mp3_folder(folder_path)
