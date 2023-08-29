import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.ttk import Progressbar
import pyaudio
import wave
import os
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack
from threading import Thread
from utils import load_data, split_data, create_model
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd


THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
SILENCE = 30

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * RATE))])
    return r

def record(progress_var):

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

        # Update the progress bar
        progress = min(100, int((len(r) / RATE) * 100))
        progress_var.set(progress)

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path, progress_var):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record(progress_var)
    data = pack('<' + ('h' * len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def extract_feature(file_name, **kwargs):

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
        result = np.hstack((result, tonnetz))
    return result

def classify_gender(progress_var):
    file = "test.wav"
    record_to_file(file, progress_var)
    features = extract_feature(file, mel=True).reshape(1, -1)
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "Male" if male_prob > female_prob else "Female"
    probabilities = f"Probabilities:\nMale: {male_prob * 100:.2f}%\nFemale: {female_prob * 100:.2f}%"
    show_info("Gender Classification Result", f"Result: {gender}\n{probabilities}")

def play_audio():
    file = "test.wav"
    if os.path.exists(file):
        audio = AudioSegment.from_wav(file)
        play(audio)
    else:
        show_info("Audio Playback", "No recorded audio file found.")

def load_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        classify_gender_from_file(file_path)

def play_local_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        data, fs = librosa.load(file_path, sr=None)
        sd.play(data, fs)
        sd.wait()
    else:
        show_info("Audio Playback", "No audio file selected.")

def classify_gender_from_file(file_path):
    if os.path.exists(file_path):
        features = extract_feature(file_path, mel=True).reshape(1, -1)
        male_prob = model.predict(features)[0][0]
        female_prob = 1 - male_prob
        gender = "Male" if male_prob > female_prob else "Female"
        probabilities = f"Probabilities:\nMale: {male_prob * 100:.2f}%\nFemale: {female_prob * 100:.2f}%"
        show_info("Gender Classification Result", f"Result: {gender}\n{probabilities}")
    else:
        show_info("File Error", "Invalid audio file.")

def show_info(title, message):
    result_window = tk.Toplevel(window)
    result_window.title(title)
    result_window.geometry("300x300")

    # Calculate the screen width and height
    screen_width = result_window.winfo_screenwidth()
    screen_height = result_window.winfo_screenheight()

    # Calculate the x and y coordinates for centering the result window
    x = int((screen_width / 2) - (300 / 2))
    y = int((screen_height / 2) - (300 / 2))

    # Set the result window position
    result_window.geometry(f"300x300+{x}+{y}")

    result_label = tk.Label(result_window, text=message, font=("Helvetica", 20), wraplength=280, justify='center')
    result_label.pack(pady=(30, 10), fill='both', expand=True)

    ok_button = tk.Button(result_window, text="OK", command=result_window.destroy, width=20)
    ok_button.pack(pady=(10, 30))

    result_window.transient(window)
    result_window.grab_set()
    window.wait_window(result_window)



def exit_application():
    window.destroy()

# Load the model
model_path = "results/model.h5"
model = create_model()
model.load_weights(model_path)

# Create the main window
window = tk.Tk()
window.title("Gender Classification")
window.geometry("600x600")

# Create a label for the heading
heading_label = tk.Label(window, text="Voice Based Gender Detection Model by Ayush Pundir", font=("Helvetica", 16, "bold"))
heading_label.pack(pady=10)

# Create a progress bar
progress_var = tk.IntVar()
progress_bar = Progressbar(window, variable=progress_var, maximum=100, mode="determinate")
progress_bar.pack(pady=10)

# Create a label to display the recording status
status_label = tk.Label(window, text="Recording in-progress", font=("Helvetica", 12))
status_label.pack(pady=5)

# Create a record button
record_button = tk.Button(window, text="Record", command=lambda: classify_gender(progress_var), width=20)
record_button.pack(pady=(20, 5))

# Create an audio playback button with a play symbol
play_button_text = u"\u25B6"  # Play symbol unicode character
play_button = tk.Button(window, text=play_button_text, command=play_audio, width=20)
play_button.pack(pady=10)

# Create a load audio button
load_button = tk.Button(window, text="Load Audio", command=load_audio, width=20)
load_button.pack(pady=10)

#Local File button to play local file
play_local_button = tk.Button(window, text="Play Local File", command=play_local_file, width=20)
play_local_button.pack(pady=10)


# Create a label to display the result
result_label = tk.Label(window, text="", font=("Helvetica", 25))
result_label.pack()

# Create an exit button
exit_button = tk.Button(window, text="Exit", command=exit_application, width=20)
exit_button.pack(pady=(5))

def update_status_label():
    progress = progress_var.get()
    if progress == 100:
        status_label.config(text="Recording completed")
    else:
        status_label.config(text="Recording in-progress")
    window.after(100, update_status_label)

# Start updating the status label
update_status_label()

# Run the Tkinter event loop
window.mainloop()
