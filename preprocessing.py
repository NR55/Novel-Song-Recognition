import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import pickle
import glob

folder_path = "Songs"

def constel_gen(audio, Fs):

    win_len_seconds = 0.5
    win_len_samples = int(win_len_seconds * Fs)
    win_len_samples += win_len_samples % 2
    num_peaks = 15
    amount_to_pad = win_len_samples - audio.size % win_len_samples
    song_input = np.pad(audio, (0, amount_to_pad))
    frequencies, times, stft = signal.stft(
        song_input, Fs, nperseg=win_len_samples, nfft=win_len_samples, return_onesided=True
    )

    conste_map = []

    for time_idx, window in enumerate(stft.T):
        spectrum = abs(window)
        peaks, props = signal.find_peaks(spectrum, prominence=0, distance=200)
        n_peaks = min(num_peaks, len(peaks))
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            freq = frequencies[peak]
            conste_map.append([time_idx, freq])

    return conste_map

def hash_gen(conste_map, song_id=None):
    hashes = {}
    upper_freq = 23_000 
    freq_bits = 10
    for idx, (time, freq) in enumerate(conste_map):
        for other_time, other_freq in conste_map[idx : idx + 100]: 
            diff = other_time - time
            if diff <= 1 or diff > 10:
                continue
            freq_binned = freq / upper_freq * (2 ** freq_bits)
            other_freq_binned = other_freq / upper_freq * (2 ** freq_bits)
            hash_val = int(freq_binned) | (int(other_freq_binned) << 10) | (int(diff) << 20)
            hashes[hash_val] = (time, song_id)
    return hashes

def generator(folder_path):
    output_folder = "Database"
    os.makedirs(output_folder, exist_ok=True)

    song_name_index = {}
    dtbs = {}
    songs = glob.glob(os.path.join(folder_path, '*.mp3'))
    
    for index, filename in enumerate(tqdm(sorted(songs))):

        song_name_index[index] = filename
        y, sr = librosa.load(filename)

        conste_map = constel_gen(y, sr)
        hashes = hash_gen(conste_map, index)
        for i, (hash_val, (time, _)) in enumerate(hashes.items()):
            if i > 10:
                break

        for hash_val, time_index_pair in hashes.items():
            if hash_val not in dtbs:
                dtbs[hash_val] = []
            dtbs[hash_val].append(time_index_pair)

    with open(os.path.join(output_folder, "database.pickle"), 'wb') as db:
        pickle.dump(dtbs, db, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_folder, "song_index.pickle"), 'wb') as songs:
        pickle.dump(song_name_index, songs, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    generator(folder_path)