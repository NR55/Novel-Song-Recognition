import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
import pickle
import sounddevice as sd
from preprocessing import create_constellation, create_hashes

# Load the database
database = pickle.load(open('Database/database.pickle', 'rb'))
song_name_index = pickle.load(open("Database/song_index.pickle", "rb"))

# Function to remove noise from audio signal
def remove_noise(audio_signal, threshold=0.00):
    # Apply thresholding to remove noise
    audio_signal[np.abs(audio_signal) < threshold] = 0.0
    return audio_signal

# Function to record audio from the microphone
def record_audio(duration=10, fs=44100):
    print(f"Recording audio for {duration} seconds...")
    audio_input = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    print("Recording complete.")
    return fs, audio_input.flatten()

# Function to perform audio matching with the database
def audio_matching(database, song_name_index, audio_input, threshold=0.00):
    # Create the constellation and hashes
    constellation = create_constellation(audio_input, Fs)
    hashes = create_hashes(constellation, None)

    # For each hash in the song, check if there's a match in the database
    # There could be multiple matching tracks, so for each match:
    #   Increment a counter for that song ID by one
    matches_per_song = {}
    for hash, (sample_time, _) in hashes.items():
        if hash in database:
            matching_occurrences = database[hash]
            for source_time, song_id in matching_occurrences:
                if song_id not in matches_per_song:
                    matches_per_song[song_id] = 0
                matches_per_song[song_id] += 1

    # Print the top matches
    top_matches = sorted(matches_per_song.items(), key=lambda x: x[1], reverse=True)[:10]
    for song_id, num_matches in top_matches:
        print(f"Song: {song_name_index[song_id]} - Matches: {num_matches}")

if __name__ == "__main__":

    # Record audio from the microphone
    duration = 10  # Set the duration of recording (in seconds)
    Fs, audio_input = record_audio(duration)

    # Plot the recorded audio
    plt.figure(figsize=(12, 4))
    plt.plot(audio_input)
    plt.title('Recorded Audio')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

    # Remove noise from the recorded audio
    audio_input = remove_noise(audio_input)

    # Plot the processed audio without noise
    plt.figure(figsize=(12, 4))
    plt.plot(audio_input)
    plt.title('Processed Audio (Noise Removed)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

    # Perform audio matching with the database
    audio_matching(database, song_name_index, audio_input, threshold=0.00)
