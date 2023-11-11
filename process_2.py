import os
import sounddevice as sd
import librosa
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play

# Specify the input folder containing audio files
input_folder = "./Songs"

def extract_features(audio_signal, sample_rate, chunk_size=10, num_bands=6):
    try:
        # Convert stereo to mono
        if len(audio_signal.shape) > 1:
            audio_signal = audio_signal[:, 0]

        # Resample the signal to 16384 Hz
        audio_signal_resampled = librosa.resample(audio_signal, orig_sr=sample_rate, target_sr=16384)

        # Apply hamming window (window length 1024) and extract features for each chunk
        window_size = 1024
        hop_size = 32
        features = []

        for chunk in librosa.util.frame(audio_signal_resampled, frame_length=chunk_size * 16384, hop_length=chunk_size * 16384, axis=0):
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
        print(f"Error extracting features from audio signal: {e}")
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
    
def hamming_distance(hash1, hash2):
    # Calculate the Hamming distance between two hash values
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def load_hash_table(json_file):
    # Load the precomputed hash table from the JSON file
    with open(json_file, 'r') as file:
        hash_table = json.load(file)

    return hash_table


def remove_noise(audio_signal, threshold=0.02):
    # Apply thresholding to remove noise
    audio_signal[np.abs(audio_signal) < threshold] = 0.0
    return audio_signal

def find_most_similar_song(input_features, hash_table):
    if input_features is None:
        print("Error: Input features are None.")
        return None

    # Generate a hash for the input features
    input_hash = generate_hash(input_features)

    # Compare the input hash with the hashes in the hash table
    similarity_scores = {}

    for key in hash_table:
        hash_value = hash_table[key]

        # Skip if the hash value is None
        if hash_value is None:
            continue

        # Ensure hashes have the same length
        min_len = min(len(input_hash), len(hash_value))
        hamming_dist = hamming_distance(input_hash[:min_len], hash_value[:min_len])


        # Calculate similarity score
        similarity_score = 1 - (hamming_dist / len(input_hash))  # Use the length of the input hash for normalization
        similarity_scores[key] = similarity_score


    # Find the audio file with the highest similarity score
    most_similar_song = max(similarity_scores, key=similarity_scores.get)

    return most_similar_song, similarity_scores


def play_audio(audio_signal, sample_rate):
    # Convert the NumPy array to PyDub AudioSegment
    audio_segment = AudioSegment(audio_signal.tobytes(), frame_rate=sample_rate, sample_width=audio_signal.dtype.itemsize, channels=1)

    # Play the audio
    play(audio_segment)

def plot_similarity(similarity_scores):
    # Extract the substring of the name from the last occurrence of '/' to the end
    song_names = [key.rsplit('/', 1)[-1] for key in similarity_scores.keys()]

    # Plot the similarity scores
    plt.bar(song_names, list(similarity_scores.values()))
    plt.xlabel('Songs')
    plt.ylabel('Similarity Score')
    plt.title('Similarity Scores with Different Songs')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.show()

def main():

    # Specify the precomputed hash table file
    hash_table_file = "hash_table_2.json"

    # Load the hash table
    hash_table = load_hash_table(hash_table_file)

    # Set the sample rate and duration for real-time audio input
    sample_rate = 44100
    duration = 20  # seconds

    # Record audio from the microphone
    print("Recording audio...")
    audio_signal = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()

    # Remove noise from the recorded audio
    audio_signal = remove_noise(audio_signal[:, 0])

    # Extract features from the denoised audio
    input_features = extract_features(audio_signal, sample_rate)

    # Generate a hash for the input features
    most_similar_song, similarity_scores = find_most_similar_song(input_features, hash_table)

    print("The most similar song is:", most_similar_song)

    # Play the recorded audio as output
    play_audio(audio_signal, sample_rate)

    # Plot the similarity scores with shortened song names
    plot_similarity(similarity_scores)

    

if __name__ == "__main__":
    main()