import os
import librosa
import hashlib
import json
    
# Specify the input folder containing audio files
input_folder = "/home/nr55/Desktop/Projects/SaSProj/Songs"

def extract_features(audio_file):
    # Use librosa to extract features (e.g., chroma features) from the audio file
    y, sr = librosa.load(audio_file)
    features = librosa.feature.chroma_stft(y=y, sr=sr)
    return features

def generate_hash(features):
    # Convert the features to a hash using hashlib
    hash_object = hashlib.md5(features.tobytes())
    return hash_object.hexdigest()

def preprocess_audio_files(input_folder, output_file):
    # Dictionary to store audio file paths and their corresponding hashes
    hash_table = {}

    # Iterate through each audio file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            audio_file_path = os.path.join(input_folder, filename)

            # Extract features from the audio file
            features = extract_features(audio_file_path)

            # Generate a hash for the features
            audio_hash = generate_hash(features)

            # Store the file path and hash in the hash table
            hash_table[audio_file_path] = audio_hash

    # Save the hash table to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(hash_table, json_file)

if __name__ == "__main__":

    # Specify the output file for the hash table
    output_file = "hash_table.json"

    # Preprocess audio files and generate the hash table
    preprocess_audio_files(input_folder, output_file)
