import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import pickle
import glob

folder_path = "./Songs"

def create_constellation(audio, Fs):
    # Parameters
    window_length_seconds = 0.5
    window_length_samples = int(window_length_seconds * Fs)
    window_length_samples += window_length_samples % 2
    num_peaks = 15

    # Pad the song to divide evenly into windows
    amount_to_pad = window_length_samples - audio.size % window_length_samples
    song_input = np.pad(audio, (0, amount_to_pad))

    # Perform a short time Fourier transform
    frequencies, times, stft = signal.stft(
        song_input, Fs, nperseg=window_length_samples, nfft=window_length_samples, return_onesided=True
    )

    constellation_map = []

    for time_idx, window in enumerate(stft.T):
        spectrum = abs(window)
        peaks, props = signal.find_peaks(spectrum, prominence=0, distance=200)
        n_peaks = min(num_peaks, len(peaks))
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])

    return constellation_map

def create_hashes(constellation_map, song_id=None):
    hashes = {}
    # Use this for binning - 23_000 is slightly higher than the maximum
    # frequency that can be stored in the .wav files, 22.05 kHz
    upper_frequency = 23_000 
    frequency_bits = 10

    # Iterate the constellation
    for idx, (time, freq) in enumerate(constellation_map):
        # Iterate the next 100 pairs to produce the combinatorial hashes
        # When we produced the constellation before, it was sorted by time already
        # So this finds the next n points in time (though they might occur at the same time)
        for other_time, other_freq in constellation_map[idx : idx + 100]: 
            diff = other_time - time
            # If the time difference between the pairs is too small or large
            # ignore this set of pairs
            if diff <= 1 or diff > 10:
                continue

            # Place the frequencies (in Hz) into 1024 bins
            freq_binned = freq / upper_frequency * (2 ** frequency_bits)
            other_freq_binned = other_freq / upper_frequency * (2 ** frequency_bits)

            # Produce a 32-bit hash
            # Use bit shifting to move the bits to the correct location
            hash_val = int(freq_binned) | (int(other_freq_binned) << 10) | (int(diff) << 20)
            hashes[hash_val] = (time, song_id)
    return hashes

def plot_fft_with_stft_constellation_hashes_and_database(folder_path):
    # Create a directory to store the FFT, STFT, constellation, hash, and database images
    output_folder = os.path.join(folder_path, "Database")
    os.makedirs(output_folder, exist_ok=True)

    # Song Database
    song_name_index = {}
    database = {}

    # Go through each song, using where they are alphabetically as an id
    songs = glob.glob(os.path.join(folder_path, '*.mp3'))
    for index, filename in enumerate(tqdm(sorted(songs))):
        song_name_index[index] = filename
        # Load audio file
        y, sr = librosa.load(filename)

        # Perform FFT
        fft_result = np.fft.fft(y)
        magnitude = np.abs(fft_result)
        frequency = np.fft.fftfreq(len(magnitude), d=1/sr)

        # Plot the FFT
        plt.figure(figsize=(12, 4))
        plt.plot(frequency, magnitude)
        plt.xlim(0, 2000)
        plt.title('FFT for {}'.format(os.path.basename(filename)))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        # plt.savefig(os.path.join(output_folder, os.path.basename(filename).replace(".mp3", "_fft_plot.png")))
        # plt.show()

        # Create Constellation Map
        constellation_map = create_constellation(y, sr)

        # Transform [(x, y), ...] into ([x1, x2...], [y1, y2...]) for plotting using zip
        plt.scatter(*zip(*constellation_map))
        plt.title('Constellation Map for {}'.format(os.path.basename(filename)))
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        # Create and Investigate Hashes
        hashes = create_hashes(constellation_map, index)
        for i, (hash_val, (time, _)) in enumerate(hashes.items()):
            if i > 10:
                break

        # Update the Database
        for hash_val, time_index_pair in hashes.items():
            if hash_val not in database:
                database[hash_val] = []
            database[hash_val].append(time_index_pair)

    # Dump the database and list of songs as pickles
    with open(os.path.join(output_folder, "database.pickle"), 'wb') as db:
        pickle.dump(database, db, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_folder, "song_index.pickle"), 'wb') as songs:
        pickle.dump(song_name_index, songs, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    plot_fft_with_stft_constellation_hashes_and_database(folder_path)