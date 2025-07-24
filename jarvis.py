import os
import subprocess
import cv2
import google.generativeai as genai
import numpy as np
import openwakeword as oww
import pyaudio
import pyttsx3
import speech_recognition as sr
import spotipy
from dotenv import load_dotenv
from openwakeword.model import Model
from spotipy.oauth2 import SpotifyOAuth

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Wake Word Configuration
WAKE_WORD_MODEL = "hey_jarvis" # Use the standard model name from openwakeword

# Audio Input Configuration (based on your MacBook Pro Microphone)
AUDIO_DEVICE_INDEX = int(os.getenv("AUDIO_DEVICE_INDEX", 2)) # Default to Device ID 2
CHANNELS = 1 # Your microphone requires 1 channel (mono)
RATE = 16000
CHUNK_SIZE = 1280
FORMAT = pyaudio.paInt16

# Text-to-Speech Configuration
TTS_RATE = int(os.getenv("TTS_RATE", 172))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", 0.9))

# --- Initialization ---

# Download wake word model if needed
print("Downloading wake word model...")
oww.utils.download_models([WAKE_WORD_MODEL])

# Initialize OpenWakeWord model
print("Initializing wake word model...")
oww_model = Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")

# Initialize Google Gemini API
print("Initializing Google Gemini API...")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize text-to-speech engine
print("Initializing text-to-speech engine...")
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", TTS_RATE)
tts_engine.setProperty("volume", TTS_VOLUME)

# Initialize audio stream for wake word detection
print("Initializing audio stream...")
audio = pyaudio.PyAudio()
try:
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=AUDIO_DEVICE_INDEX,
    )
except Exception as e:
    print(f"FATAL ERROR: Could not initialize microphone. {e}")
    print("Please run the check_devices.py script to find the correct AUDIO_DEVICE_INDEX.")
    raise

# Initialize Spotify API client
print("Initializing Spotify client...")
try:
    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            redirect_uri=os.getenv("REDIRECT_URI"),
            scope="user-library-read user-read-playback-state user-modify-playback-state",
            cache_path=os.getenv("SPOTIFY_CACHE_PATH"),
        )
    )
    # Perform a test API call to trigger authentication if needed
    sp.current_user()
except Exception as e:
    print(f"Could not initialize Spotify client: {e}")
    sp = None

print("\n--- Jarvis is ready ---")

# --- Core Functions ---

def speak(text):
    """Convert text to speech."""
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error in speak function: {e}")

def pause_playback():
    """Pause Spotify playback if a song is playing."""
    if not sp:
        return
    try:
        playback_state = sp.current_playback()
        if playback_state and playback_state["is_playing"]:
            sp.pause_playback()
    except Exception as e:
        print(f"Error while pausing playback: {e}")

def detect_wake_word():
    """Detect the wake word using OpenWakeWord."""
    print(f"\nListening for wake word '{WAKE_WORD_MODEL}'...")
    while True:
        raw_data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        prediction = oww_model.predict(audio_data)

        # Check for wake word detection
        if prediction[WAKE_WORD_MODEL] > 0.5:
            print("Wake word detected!")
            oww_model.reset()
            pause_playback()
            return

def capture_image():
    """Capture an image using the device camera."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("I could not open the camera.")
        return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        image_path = os.path.join(os.getcwd(), "captured_image.jpg")
        cv2.imwrite(image_path, frame)
        return image_path
    else:
        speak("I failed to capture an image.")
        return None

def is_visual_query(query):
    """Determine if the query requires visual input."""
    print(f"Checking if visual information is needed for query: '{query}'")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"Does the following user query require an image to be answered accurately? Respond with only 'Yes' or 'No'.\nQuery: {query}"
        response = model.generate_content(prompt)
        response_text = response.text.strip().lower()
        print(f"Visual query decision: {response_text}")
        return "yes" in response_text
    except Exception as e:
        print(f"Error checking visual query with Gemini: {e}")
        return False

def process_query(query, image_path=None):
    """Process the query using Google Gemini API."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = None # Initialize response variable

        if image_path:
            print("Processing query with image...")
            with open(image_path, 'rb') as f:
                img_data = f.read()
            img_part = {'mime_type': 'image/jpeg', 'data': img_data}
            response = model.generate_content([query, img_part])
            os.remove(image_path) # Clean up image
        else:
            print("Processing text-only query...")
            response = model.generate_content(query)

        # Unified response handling
        if response:
            response_text = response.text
            print(f"Response: {response_text}")
            speak(response_text)
        else:
            # This case would only happen if something went wrong before the API call
            speak("I was unable to get a response. Please try again.")

    except Exception as e:
        error_message = f"Error generating response from Gemini: {e}"
        print(error_message)
        speak("I'm sorry, I encountered an error while processing that request.")

def open_spotify_if_not_running():
    """Opens the Spotify app if it's not already running (for macOS)."""
    try:
        # This command checks if a process named "Spotify" exists
        subprocess.run(["pgrep", "-x", "Spotify"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Spotify is not running. Opening it now...")
        # This command opens the Spotify application on macOS
        subprocess.Popen(["open", "/Applications/Spotify.app"])
        # Give Spotify a moment to launch before we try to use it
        import time
        time.sleep(5)


def search_and_play(song_name):
    """Search and play a song using Spotify, activating a device if necessary."""
    if not sp:
        speak("Spotify is not configured. I can't play music.")
        return

    # 1. Make sure Spotify is running
    open_spotify_if_not_running()

    # 2. Search for the song
    print(f"Searching for song: {song_name}")
    results = sp.search(q=song_name, limit=1, type="track")
    if not results["tracks"]["items"]:
        speak(f"Sorry, I couldn't find a song called {song_name}.")
        return

    track = results["tracks"]["items"][0]
    track_uri = track["uri"]
    track_name = track["name"]
    track_artist = track["artists"][0]["name"]

    # 3. Find an available device and start playback
    try:
        devices = sp.devices()
        if not devices['devices']:
            speak("No Spotify devices found. Please open Spotify on one of your devices.")
            return

        # Prefer a computer, otherwise take the first available device
        device_id = next((d['id'] for d in devices['devices'] if d['type'] == 'Computer'), None)
        if not device_id:
            device_id = devices['devices'][0]['id']

        # 4. Start playback on the chosen device
        sp.start_playback(device_id=device_id, uris=[track_uri])
        speak(f"Playing {track_name} by {track_artist}.")

    except Exception as e:
        print(f"Error with Spotify playback: {e}")
        speak("I had trouble playing the song. Please ensure Spotify is running and you are logged in.")


def is_song_query(query):
    """Check if the query is a request to play a song."""
    query = query.lower()
    return query.startswith("play") or "play the song" in query

def main():
    """Main loop for the assistant."""
    while True:
        try:
            detect_wake_word()
            speak("Yes?")

            recognizer = sr.Recognizer()
            with sr.Microphone(device_index=AUDIO_DEVICE_INDEX) as source:
                print("Listening for your command...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    audio_command = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    query = recognizer.recognize_google(audio_command)
                    print(f"Query: {query}")

                    if is_song_query(query):
                        song_name = query.lower().replace("play", "").strip()
                        search_and_play(song_name)
                    elif is_visual_query(query):
                        speak("Let me take a look.")
                        image_path = capture_image()
                        if image_path:
                            process_query(query, image_path)
                    else:
                        process_query(query)

                except sr.WaitTimeoutError:
                    speak("I didn't hear anything. Please try again.")
                except sr.UnknownValueError:
                    speak("I'm sorry, I didn't catch that.")
                except sr.RequestError as e:
                    speak("I'm having trouble connecting to the speech service.")
                    print(f"Speech recognition request error: {e}")

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            speak("Something went wrong. I'll need to restart.")
            # In a real-world scenario, you might want to break or add a delay
            # break

if __name__ == "__main__":
    main()