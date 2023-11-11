import os
import sounddevice as sd
import librosa
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
from sklearn.metrics.pairwise import cosine_similarity

# Specify the input folder containing audio files
input_folder = "./Songs"

# Specify the precomputed hash table file
hash_table_file = "hash_table_cens.json"

def extract_features(audio_signal, sample_rate):
    # Extract Chroma Energy Normalized (CENS) features from the audio signal using Constant-Q Transform (CQT)
    features = librosa.feature.chroma_cens(y=audio_signal, sr=sample_rate)
    features_normalized = librosa.util.normalize(features)
    return features_normalized

def generate_hash(features):
    # Convert the features to a hash using hashlib
    hash_object = hashlib.md5(features.tobytes())
    return hash_object.hexdigest()

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
    # Generate a hash for the input features
    input_hash = generate_hash(input_features)

    # Convert hex string to binary array
    binary_input_hash = bin(int(input_hash, 16))[2:]

    # Ensure the binary string has the same length as other hashes in the table by padding with zeros
    binary_input_hash = binary_input_hash.zfill(128)

    # Calculate cosine similarity with each hash in the hash table
    similarity_scores={}
    for key in hash_table:
        try:
            hash_in_table = hash_table[key]
            binary_hash = bin(int(hash_in_table, 16))[2:]
            binary_hash = binary_hash.zfill(128)
            similarity_scores[key] = cosine_similarity([np.array([int(bit) for bit in binary_input_hash])],
                                                    [np.array([int(bit) for bit in binary_hash])])[0][0]
        except ValueError as e:
            print(f"Error processing hash for key '{key}': {e}")
            similarity_scores[key] = 0.0


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

    # Extract features from the denoised audio using CQT and CENS
    input_features = extract_features(audio_signal, sample_rate)

    # Generate a hash for the input features and find the most similar song
    most_similar_song, similarity_scores = find_most_similar_song(input_features, hash_table)

    print("The most similar song is:", most_similar_song)

    # Play the recorded audio as output
    play_audio(audio_signal, sample_rate)

    # Plot the similarity scores with shortened song names
    plot_similarity(similarity_scores)

if __name__ == "__main__":
    main()
