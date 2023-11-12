import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import numpy as np
import sounddevice as sd
import pickle
from preprocessing import create_constellation, create_hashes  # Import preprocessing functions

class SongRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Song Recognition App")

        # Load the background image
        image_path = "./AppBackground/Music_Wallpaper1.jpg"  # Replace with the actual path to your image
        background_image = Image.open(image_path)

        # Set the size of the Tkinter window to match the image
        self.root.geometry(f"{background_image.width}x{background_image.height}")

        # Create a label to display the background image
        background_photo = ImageTk.PhotoImage(background_image)
        background_label = tk.Label(root, image=background_photo)
        background_label.image = background_photo
        background_label.place(relx=0.5, rely=0.5, anchor="center")

        # Create a style for the record button
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 14))

        # Add the record button and place it in the center
        self.record_button = ttk.Button(root, text="Record Audio", command=self.record_and_recognize, style="TButton")
        self.record_button.place(relx=0.5, rely=0.8, anchor="center")

        # Add the result label and place it in the center with rounded corners and a custom background color
        self.result_label = tk.Label(root, text="", font=("Helvetica", 16), bg="#FF5733", fg="black", borderwidth=2, relief="solid", padx=10, pady=5, anchor="center")
        self.result_label.place(relx=0.5, rely=0.9, anchor="center")

        # Load the database
        self.database = pickle.load(open('./Database/database.pickle', 'rb'))
        self.song_name_index = pickle.load(open("./Database/song_index.pickle", "rb"))

        # Load the folder containing song cover images
        self.cover_images_folder = "./CoverImages"

    def record_and_recognize(self):
        # Record audio from the microphone
        duration = 10  # Set the duration of recording (in seconds)
        Fs, audio_input = self.record_audio(duration)

        # Remove noise from the recorded audio
        audio_input = self.remove_noise(audio_input)

        # Perform audio matching with the database and score against the database
        result = self.audio_matching(audio_input, Fs, threshold=0.00)

        # Display the result on the GUI
        self.display_result(result)

    def record_audio(self, duration=10, fs=44100):
        print(f"Recording audio for {duration} seconds...")
        audio_input = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
        sd.wait()
        print("Recording complete.")
        return fs, audio_input.flatten()

    def remove_noise(self, audio_signal, threshold=0.00):
        # Apply thresholding to remove noise
        audio_signal[np.abs(audio_signal) < threshold] = 0.0
        return audio_signal

    def score_hashes_against_database(self, hashes):
        matches_per_song = {}
        for hash, (sample_time, _) in hashes.items():
            if hash in self.database:
                matching_occurrences = self.database[hash]
                for source_time, song_index in matching_occurrences:
                    if song_index not in matches_per_song:
                        matches_per_song[song_index] = []
                    matches_per_song[song_index].append((hash, sample_time, source_time))

        # Calculate and visualize matching scores for the recorded audio
        scores = {}
        for song_index, matches in matches_per_song.items():
            song_scores_by_offset = {}
            for hash, sample_time, source_time in matches:
                delta = source_time - sample_time
                if delta not in song_scores_by_offset:
                    song_scores_by_offset[delta] = 0
                song_scores_by_offset[delta] += 1

            max_offset = max(song_scores_by_offset, key=song_scores_by_offset.get)
            max_score = song_scores_by_offset[max_offset]

            scores[song_index] = (max_offset, max_score)

        # Sort the scores for better readability
        sorted_scores = list(sorted(scores.items(), key=lambda x: x[1][1], reverse=True))

        return sorted_scores

    def audio_matching(self, audio_input, Fs, threshold=0.00):
        # Create the constellation and hashes
        constellation = create_constellation(audio_input, Fs)
        hashes = create_hashes(constellation, None)

        # Call the function to score hashes against the database
        scores = self.score_hashes_against_database(hashes)

        # Print the top matching songs from the audio input
        result = []
        for song_index, (offset, score) in scores[:5]:
            result.append((self.song_name_index[song_index], score))

        return result

    def display_result(self, result):
        # Get the top-matched song and its score
        top_song, score = result[0]

        # Extract the filename without the path or extension
        song_filename = os.path.splitext(os.path.basename(top_song))[0]

        # Update the result label with the top-matched song and score
        self.result_label.config(text=f"Top matching song: {song_filename} (Score: {score})")

        # Load and display the cover image for the top-matched song
        cover_image_path = os.path.join(self.cover_images_folder, f"{song_filename}.jpeg")
        print(f"Cover image path: {cover_image_path}")

        if os.path.exists(cover_image_path):
            cover_image = Image.open(cover_image_path)
            cover_photo = ImageTk.PhotoImage(cover_image)

        else:
            cover_image = Image.open("./CoverImages/Default.jpg")
            cover_photo = ImageTk.PhotoImage(cover_image)
        
        # Create a label to display the cover image
        cover_label = tk.Label(self.root, image=cover_photo)
        cover_label.image = cover_photo
        cover_label.place(relx=0.5, rely=0.6, anchor="center")


if __name__ == "__main__":
    root = tk.Tk()
    app = SongRecognitionApp(root)
    root.mainloop()
