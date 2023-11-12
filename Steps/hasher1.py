import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal

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
        # Spectrum is by default complex.
        # We want real values only
        spectrum = abs(window)
        # Find peaks - these correspond to interesting features
        # Note the distance - want an even spread across the spectrum
        peaks, props = signal.find_peaks(spectrum, prominence=0, distance=200)

        # Only want the most prominent peaks
        # With a maximum of 15 per time slice
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

def plot_fft_with_stft_constellation_and_hashes(folder_path):
    # Create a directory to store the FFT, STFT, constellation, and hash images
    #output_folder = os.path.join(folder_path, "fft_stft_constellation_hashes_plots")
    #os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)

            # Load audio file
            y, sr = librosa.load(file_path)

            # Perform FFT
            fft_result = np.fft.fft(y)
            magnitude = np.abs(fft_result)
            frequency = np.fft.fftfreq(len(magnitude), d=1/sr)

            # Plot the FFT
            plt.figure(figsize=(12, 4))
            plt.plot(frequency, magnitude)
            plt.xlim(0, 2000)
            plt.title('FFT for {}'.format(filename))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            #plt.savefig(os.path.join(output_folder, filename.replace(".mp3", "_fft_plot.png")))
            plt.show()

            # Create Constellation Map
            constellation_map = create_constellation(y, sr)

            # Transform [(x, y), ...] into ([x1, x2...], [y1, y2...]) for plotting using zip
            plt.scatter(*zip(*constellation_map))
            plt.title('Constellation Map for {}'.format(filename))
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            #plt.savefig(os.path.join(output_folder, filename.replace(".mp3", "_constellation_plot.png")))
            plt.show()

            # Create and Investigate Hashes
            hashes = create_hashes(constellation_map, song_id=filename)
            for i, (hash_val, (time, _)) in enumerate(hashes.items()):
                if i > 10: 
                    break
                print(f"Hash {hash_val} occurred at {time} in {filename}")

if __name__ == "__main__":
    folder_path = "../Songs"
    plot_fft_with_stft_constellation_and_hashes(folder_path)