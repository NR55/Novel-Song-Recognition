import os
import sounddevice as sd
import librosa
import hashlib
import json
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

# Specify the input folder containing audio files
input_folder = "/home/nr55/Desktop/Projects/SaSProj/Songs"

def extract_features(audio_signal, sample_rate):
    # Extract features (e.g., chroma features) from the audio signal
    features = librosa.feature.chroma_stft(y=audio_signal, sr=sample_rate)
    return features

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

    # Compare the input hash with the hashes in the hash table
    similarity_scores = {key: np.sum(np.array(list(hash_table[key])) == np.array(list(input_hash)))
                         for key in hash_table}

    # Find the audio file with the highest similarity score
    most_similar_song = max(similarity_scores, key=similarity_scores.get)

    return most_similar_song

def play_audio(audio_signal, sample_rate):
    # Convert the NumPy array to PyDub AudioSegment
    audio_segment = AudioSegment(audio_signal.tobytes(), frame_rate=sample_rate, sample_width=audio_signal.dtype.itemsize, channels=1)

    # Play the audio
    play(audio_segment)

def main():

    # Specify the precomputed hash table file
    hash_table_file = "hash_table.json"

    # Load the hash table
    hash_table = load_hash_table(hash_table_file)

    # Set the sample rate and duration for real-time audio input
    sample_rate = 44100
    duration = 10  # seconds

    # Record audio from the microphone
    print("Recording audio...")
    audio_signal = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()

    # Remove noise from the recorded audio
    audio_signal = remove_noise(audio_signal[:, 0])

    # Extract features from the denoised audio
    input_features = extract_features(audio_signal, sample_rate)

    # Generate a hash for the input features
    most_similar_song = find_most_similar_song(input_features, hash_table)

    print("The most similar song is:", most_similar_song)

    # Play the recorded audio as output
    play_audio(audio_signal, sample_rate)

if __name__ == "__main__":
    main()
