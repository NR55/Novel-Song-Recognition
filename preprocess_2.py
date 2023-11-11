import os
import librosa
import hashlib
import json
import numpy as np
from scipy.ndimage import maximum_filter

# Specify the input folder containing audio files
input_folder = "./Songs"


def extract_features(audio_file, chunk_size=10, num_bands=6):
    try:
        # Convert stereo to mono
        y, sr = librosa.load(audio_file, mono=True)

        # Resample the signal to 16384 Hz
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16384)

        # Apply hamming window (window length 1024) and extract features for each chunk
        window_size = 1024
        hop_size = 32
        features = []

        for chunk in librosa.util.frame(y_resampled, frame_length=chunk_size * 16384, hop_length=chunk_size * 16384, axis=0):
            chunk_windowed = np.pad(chunk, pad_width=window_size // 2, mode='reflect')
            chunk_windowed = librosa.effects.preemphasis(chunk_windowed)
            stft_result = np.abs(librosa.stft(chunk_windowed, hop_length=hop_size, window='hamming'))

            # Divide the STFT result into logarithmic bands
            num_bins = stft_result.shape[0]
            bins_per_band = num_bins // num_bands
            bands = [stft_result[i * bins_per_band:(i + 1) * bins_per_band, :] for i in range(num_bands)]

            # Find local maxima in each band
            maxima_per_band = [np.where((bands[i] == np.max(bands[i], axis=0)) & (bands[i] > 0.8 * np.max(bands[i]))) for i in range(num_bands)]

            # Flatten the array before further processing
            flat_maxima = np.concatenate(maxima_per_band, axis=1).flatten()

            # Extract peaks
            peaks = np.where(flat_maxima == 1)[0]
            features.extend(peaks)

        # Input normalization
        features = np.array(features)
        features = (features - np.mean(features)) / np.std(features) if np.std(features) != 0 else features

        return features

    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return None

def generate_hash(features):
    if features is not None:
        # Convert the list to a NumPy array
        features_array = np.array(features)

        # Convert the array to bytes
        features_bytes = features_array.tobytes()

        # Use SHA-256 for hashing
        hash_object = hashlib.sha256(features_bytes)

        return hash_object.hexdigest()
    else:
        return None


def preprocess_audio_files(input_folder, output_file):
    hash_table = {}

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            audio_file_path = os.path.join(input_folder, filename)

            features = extract_features(audio_file_path)
            audio_hash = generate_hash(features)

            if audio_hash is not None:
                hash_table[audio_file_path] = audio_hash

    try:
        with open(output_file, 'w') as json_file:
            json.dump(hash_table, json_file)
    except Exception as e:
        print(f"Error saving hash table to {output_file}: {e}")

if __name__ == "__main__":
    output_file = "hash_table_2.json"

    preprocess_audio_files(input_folder, output_file)