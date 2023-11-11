import os
import sounddevice as sd
import librosa
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play

def load_hash_table(json_file):
    # Load the precomputed hash table from the JSON file
    with open(json_file, 'r') as file:
        hash_table = json.load(file)
    return hash_table

def remove_noise(audio_signal, threshold=0.02):
    # Apply thresholding to remove noise
    audio_signal[np.abs(audio_signal) < threshold] = 0.0
    return audio_signal

def extract_features(audio_signal, sample_rate, chunk_size=10, num_bands=6):
    try:
        # Convert stereo to mono
        if len(audio_signal.shape) > 1:
            audio_signal = audio_signal[:, 0]

        # Resample the signal to 8192 Hz
        audio_signal_resampled = librosa.resample(audio_signal, orig_sr=sample_rate, target_sr=8192)

        # Apply hamming window (window length 1024) and extract features for each chunk
        window_size = 1024
        hop_size = 32
        features = []

        for chunk in librosa.util.frame(audio_signal_resampled, frame_length=chunk_size * 8192, hop_length=chunk_size * 8192, axis=0):
            chunk_windowed = np.pad(chunk, pad_width=window_size // 2, mode='reflect')
            chunk_windowed = librosa.effects.preemphasis(chunk_windowed)
            stft_result = np.abs(librosa.stft(chunk_windowed, hop_length=hop_size, window='hamming'))

            # Divide the STFT result into logarithmic bands
            num_bins = stft_result.shape[0]
            bins_per_band = num_bins // num_bands
            bands = [stft_result[i * bins_per_band:(i + 1) * bins_per_band, :] for i in range(num_bands)]

            # Find local maxima in each band
            maxima_per_band = [np.where((bands[i] == np.max(bands[i], axis=0)) & (bands[i] > 0.1 * np.max(bands[i]))) for i in range(num_bands)]

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
    
def generate_hash(anchor_point, target_point, time_change):
    # Convert zero-dimensional arrays to one-dimensional arrays
    anchor_point = np.atleast_1d(anchor_point)
    target_point = np.atleast_1d(target_point)

    # Combine anchor point, target point, and time change for hash generation
    combined_features = np.concatenate([anchor_point, target_point, np.array([time_change])])

    if combined_features is not None:
        # Convert the array to bytes
        hash_object = hashlib.md5(combined_features.tobytes())
        return hash_object.hexdigest()
    else:
        return None


def generate_hash_table(features, target_zone=50):
    hash_table = {}

    # Iterate over anchor points
    for anchor_index, anchor_point in enumerate(features):
        # Iterate over target points within the target zone
        for target_index in range(anchor_index, min(anchor_index + target_zone, len(features))):
            target_point = features[target_index]
            time_change = target_index - anchor_index

            # Generate hash using anchor point, target point, and time change
            hash_value = generate_hash(anchor_point, target_point, time_change)

            # Store the hash value in the hash table
            if anchor_index not in hash_table:
                hash_table[anchor_index] = []
            hash_table[anchor_index].append((target_index, hash_value))

    return hash_table

def analyze_offset_times(client_hash_table, server_hash_tables):
    client_offset_times = []
    server_offset_times = []

    # Iterate over each element in the client hash table
    for client_anchor, client_targets in client_hash_table.items():
        # Check if the element is also present in the server hash tables
        if client_anchor in server_hash_tables:
            server_targets = server_hash_tables[client_anchor]

            # Iterate over client targets for the given anchor
            for client_target_index, client_hash in client_targets:
                # Iterate over server targets for the given anchor
                for server_target_index, server_hash in server_targets:
                    if server_target_index == client_target_index:
                        server_offset_times.append(server_target_index)
                        client_offset_times.append(client_target_index)

    return client_offset_times, server_offset_times

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

def find_most_similar_song(input_features, hash_tables):
    if input_features is None:
        print("Error: Input features are None.")
        return None

    # Create a hash table for the input features
    client_hash_table = generate_hash_table(input_features)

    # Compare the client hash table with hash tables from the database
    matching_counts = compare_hash_tables(client_hash_table, hash_tables)

    # Analyze offset times for matched elements
    client_offset_times, server_offset_times = analyze_offset_times(client_hash_table, hash_tables)

    # Identify the most similar song based on offset times
    most_similar_song = max(matching_counts, key=matching_counts.get)

    return most_similar_song, matching_counts, client_offset_times, server_offset_times

def compare_hash_tables(client_hash_table, server_hash_tables):
    matching_counts = {}

    # Iterate over each hash table in the server
    for song_id, server_hash_table in server_hash_tables.items():
        # Count of matched elements for each song
        matching_count = 0

        # Iterate over each element in the client hash table
        for client_anchor, client_targets in client_hash_table.items():
            # Check if the element is also present in the server hash table
            if str(client_anchor) in server_hash_tables:
                server_targets = server_hash_tables[str(client_anchor)]

                # Compare the hash values for each target point
                for client_target_index, client_hash in client_targets:
                    for server_target_index, server_hash in server_targets:
                        if client_target_index == server_target_index and client_hash == server_hash:
                            matching_count += 1

        matching_counts[song_id] = matching_count

    return matching_counts

def main():
    # Specify the precomputed hash table file
    hash_table_file = "hash_tables_1.json"

    # Load the hash table
    hash_tables = load_hash_table(hash_table_file)

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

    most_similar_song, matching_counts, client_offset_times, server_offset_times = find_most_similar_song(input_features, hash_tables)

    print("The most similar song is:", most_similar_song)
    print("Matching counts:", matching_counts)

    # Plot the similarity scores with shortened song names
    plot_similarity(matching_counts)

    # Plot histogram of offset times
    plt.hist(client_offset_times, bins=range(min(client_offset_times), max(client_offset_times) + 2), alpha=0.5, label='Client')
    plt.hist(server_offset_times, bins=range(min(server_offset_times), max(server_offset_times) + 2), alpha=0.5, label='Server')
    plt.xlabel('Offset Times')
    plt.ylabel('Frequency')
    plt.title('Histogram of Offset Times')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

