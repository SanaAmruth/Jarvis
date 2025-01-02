import os
import base64
import pyttsx3
import cv2
import numpy as np
from dotenv import load_dotenv
import openwakeword as oww
from openwakeword.model import Model
import pyaudio
import unify
import speech_recognition as sr
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import subprocess

# Load environment variables
load_dotenv()

# Download models needed for tests
oww.utils.download_models(["Hey Jarvis"])

# Access the API key
api_key = os.getenv("UNIFY_API_KEY")

# Configuration
UNIFY_API_KEY = api_key
WAKE_WORD = "Hey Jarvis"
AUDIO_DEVICE_INDEX = int(
    os.getenv("AUDIO_DEVICE_INDEX", 0)
)  # Default to system default device

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", int(os.getenv("TTS_RATE", 172)))
tts_engine.setProperty("volume", float(os.getenv("TTS_VOLUME", 0.9)))

# Set model path
model_path = "hey jarvis"

# Initialize OpenWakeWord model
oww_model = Model(wakeword_models=[model_path], inference_framework="onnx")

# Setup for audio streaming
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1280
audio = pyaudio.PyAudio()

try:
    mic_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=AUDIO_DEVICE_INDEX if AUDIO_DEVICE_INDEX >= 0 else None,
    )
except Exception as e:
    print(f"Error initializing microphone: {e}")
    raise

# Initialize Spotify API client
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
SCOPE = "user-library-read user-read-playback-state user-modify-playback-state"

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=os.getenv("SPOTIFY_CACHE_PATH", None),
    )
)


def speak(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()


def pause_playback():
    """Pause playback if a song is playing."""
    try:
        playback_state = sp.current_playback()

        if playback_state and playback_state["is_playing"]:
            sp.pause_playback()
    except Exception as e:
        print(f"Error while pausing playback: {e}")


def detect_wake_word():
    """Detect the wake word using OpenWakeWord."""
    print("Listening for wake word...")
    while True:
        raw_data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        oww_model.predict(audio_data)

        del raw_data
        del audio_data

        for mdl in oww_model.prediction_buffer.keys():
            if oww_model.prediction_buffer[mdl][-1] > 0.4:
                oww_model.prediction_buffer[mdl].clear()
                print("Wake word detected!")

                # Pause playback when wake word is detected
                pause_playback()

                return


def capture_image():
    """Capture an image using the device camera."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        image_path = os.path.join(os.getcwd(), "captured_image.jpg")
        cv2.imwrite(image_path, frame)
        cap.release()
        return image_path
    else:
        cap.release()
        raise RuntimeError("Failed to capture image.")


def is_visual_query(query):
    """Determine if the query requires visual input using Unify API."""
    print(f"Checking if visual information is needed for query: {query}")
    try:
        client = unify.Unify("gpt-4o-mini@openai", api_key=UNIFY_API_KEY)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"You have the capability to get the image if needed to answer the query. Does the following query need an image to answer?\nQuery: {query}\nRespond with 'Yes' or 'No'",
                    }
                ],
            }
        ]

        response = client.generate(messages=messages, max_completion_tokens=1)
        print(f"Visual query decision: {response}")
        return "Yes" in response
    except Exception as e:
        print(f"Error checking visual query: {e}")
        return False


def process_query(query, image_path=None):
    """Process the query using Unify API."""
    client = unify.Unify("gpt-4o-mini@openai", api_key=UNIFY_API_KEY)
    messages = [{"role": "system", "content": "Provide precise and accurate answers."}]

    if image_path:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        )
        os.remove(image_path)
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})

    try:
        response = client.generate(messages=messages, max_completion_tokens=150)
        print(f"Response: {response}")
        speak(response)
        return response
    except Exception as e:
        return f"Error processing query: {e}"


def search_and_play(song_name):
    """Search and play song using Spotify API."""
    try:
        subprocess.run(
            ["pgrep", "-x", "Spotify"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        print("Spotify is not running. Opening Spotify...")
        subprocess.Popen(["open", os.getenv("SPOTIFY_PATH", "Spotify")])

    results = sp.search(q=song_name, limit=10, type="track")
    if results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        track_uri = track["uri"]
        track_name = track["name"]
        track_artist = track["artists"][0]["name"]

        devices = sp.devices()
        active_device = next(
            (device for device in devices["devices"] if device["is_active"]), None
        )

        if active_device:
            device_id = active_device["id"]
            sp.start_playback(device_id=device_id, uris=[track_uri])
            print(f"Playing: {track_name} by {track_artist}")
        else:
            print("No active device found.")
    else:
        print("No song found.")


def is_song_query(query):
    """Check if the query mentions a song."""
    return query.lower().split()[0] == "play"


def main():
    global mic_stream
    while True:
        try:
            detect_wake_word()

            recognizer = sr.Recognizer()
            with sr.Microphone(device_index=AUDIO_DEVICE_INDEX) as source:
                print("Listening for query...")
                audio = recognizer.listen(source)
                try:
                    query = recognizer.recognize_google(audio)
                    print(f"Query: {query}")

                    if is_song_query(query):
                        search_and_play(query)
                    elif is_visual_query(query):
                        image_path = capture_image()
                        process_query(query, image_path)
                    else:
                        process_query(query)
                except Exception as e:
                    print(f"Error recognizing speech: {e}")
        except Exception as e:
            print(f"Error in main loop: {e}")


if __name__ == "__main__":
    main()
