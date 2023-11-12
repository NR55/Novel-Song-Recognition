import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import numpy as np
import sounddevice as sd
import pickle
from preprocessing import constel_gen, hash_gen  # Import preprocessing functions

background_img_path = "./AppBackground/Music_Wallpaper2.jpg"
default_img_path="./CoverImages/Default.jpg"

class SongRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Song Recognition App")

        background_image = Image.open(background_img_path)

        self.root.geometry(f"{background_image.width}x{background_image.height}")

        background_photo = ImageTk.PhotoImage(background_image)
        background_label = tk.Label(root, image=background_photo)
        background_label.image = background_photo
        background_label.place(relx=0.5, rely=0.5, anchor="center")

        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 14))

        self.record_button = ttk.Button(root, text="Record Audio", command=self.record_and_recognize, style="TButton")
        self.record_button.place(relx=0.5, rely=0.8, anchor="center")

        self.result_label = tk.Label(root, text="", font=("Helvetica", 16), bg="#808080", fg="white", borderwidth=2, relief="solid", padx=10, pady=5, anchor="center")
        self.result_label.place(relx=0.5, rely=0.9, anchor="center")

        self.database = pickle.load(open('./Database/database.pickle', 'rb'))
        self.song_name_index = pickle.load(open("./Database/song_index.pickle", "rb"))

        self.cover_images_folder = "./CoverImages"

    def record_and_recognize(self):
        duration = 10
        Fs, audio_input = self.record_audio(duration)

        audio_input = self.remove_noise(audio_input)

        result = self.audio_matching(audio_input, Fs, threshold=0.00)

        self.display_result(result)

    def record_audio(self, duration=10, fs=44100):
        print(f"Recording audio for {duration} seconds...")
        audio_input = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
        sd.wait()
        print("Recording complete.")
        return fs, audio_input.flatten()

    def remove_noise(self, audio_signal, threshold=0.00):
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
        sorted_scores = list(sorted(scores.items(), key=lambda x: x[1][1], reverse=True))
        return sorted_scores

    def audio_matching(self, audio_input, Fs, threshold=0.00):
        constellation = constel_gen(audio_input, Fs)
        hashes = hash_gen(constellation, None)
        scores = self.score_hashes_against_database(hashes)
        result = []
        for song_index, (offset, score) in scores[:5]:
            result.append((self.song_name_index[song_index], score))
        return result

    def display_result(self, result):
        top_song, score = result[0]
        song_filename = os.path.splitext(os.path.basename(top_song))[0]
        self.result_label.config(text=f"Best-matching Song : {song_filename}")
        cover_image_path = os.path.join(self.cover_images_folder, f"{song_filename}.jpeg")
        print(f"Cover image path: {cover_image_path}")
        if os.path.exists(cover_image_path):
            cover_image = Image.open(cover_image_path)
            cover_photo = ImageTk.PhotoImage(cover_image)
        else:
            cover_image = Image.open(default_img_path)
            cover_photo = ImageTk.PhotoImage(cover_image)

        cover_label = tk.Label(self.root, image=cover_photo)
        cover_label.image = cover_photo
        cover_label.place(relx=0.5, rely=0.55, anchor="center")


if __name__ == "__main__":
    root = tk.Tk()
    app = SongRecognitionApp(root)
    root.mainloop()
