import os
import base64
import pyttsx3
import cv2
import numpy as np
from dotenv import load_dotenv
import openwakeword as oww
from openwakeword.model import Model
import pyaudio
import speech_recognition as sr
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import subprocess

# Import Google GenAI SDK
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Download models needed for tests (for OpenWakeWord)
oww.utils.download_models(["Hey Jarvis"])

# Access the API key for Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Configure Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Configuration
WAKE_WORD = "Hey Jarvis"
AUDIO_DEVICE_INDEX = int(
    os.getenv("AUDIO_DEVICE_INDEX", 0)
)  # Default to system default device

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", int(os.getenv("TTS_RATE", 172)))
tts_engine.setProperty("volume", float(os.getenv("TTS_VOLUME", 0.9)))

# Set model path for OpenWakeWord
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
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

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
    """Determine if the query requires visual input using Google Gemini (text-only model)."""
    print(f"Checking if visual information is needed for query: {query}")
    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash-lite-preview-06-17')
        prompt = f"Given the following query, does it require an image to answer? Respond with 'Yes' or 'No'.\nQuery: {query}"
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.1,  # Keep temperature low for direct answers
            max_output_tokens=5 # Just enough for "Yes" or "No"
        ))
        
        # Accessing the text from the response object
        response_text = response.text.strip().lower()
        print(f"Visual query decision: {response_text}")
        return "yes" in response_text
    except Exception as e:
        print(f"Error checking visual query with Gemini: {e}")
        return False


def process_query(query, image_path=None):
    """Process the query using Google Gemini API."""
    messages = []

    if image_path:
        model = genai.GenerativeModel('models/gemini-2.5-flash-lite-preview-06-17')
        try:
            # Read the image and create a BytesPart
            img = {
                'mime_type': 'image/jpeg',
                'data': open(image_path, 'rb').read()
            }
            messages.append(img)
            messages.append(query) # Text part of the multimodal prompt
            
            response = model.generate_content(messages, generation_config=genai.types.GenerationConfig(
                max_output_tokens=150
            ))
            
            # Remove the captured image after processing
            os.remove(image_path)
        except Exception as e:
            print(f"Error processing image with Gemini: {e}")
            # Fallback to text-only if image processing fails
            response = genai.GenerativeModel('models/gemini-2.5-flash-lite-preview-06-17').generate_content(query, generation_config=genai.types.GenerationConfig(
                max_output_tokens=150
            ))
            speak("I had trouble processing the image, but I'll try to answer based on your words.")
    else:
        model = genai.GenerativeModel('models/gemini-2.5-flash-lite-preview-06-17')
        response = model.generate_content(query, generation_config=genai.types.GenerationConfig(
            max_output_tokens=150
        ))

    try:
        response_text = response.text
        print(f"Response: {response_text}")
        speak(response_text)
        return response_text
    except Exception as e:
        error_message = f"Error generating response from Gemini: {e}"
        print(error_message)
        speak("I'm sorry, I couldn't process that request right now.")
        return error_message


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
        # Adjust this command for your OS if needed. 'open' is for macOS.
        # For Windows: 'start spotify:'
        # For Linux: 'spotify' or 'xdg-open spotify:'
        subprocess.Popen(["open", os.getenv("SPOTIFY_PATH", "/Applications/Spotify.app")]) # Example for macOS

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
            speak(f"Playing: {track_name} by {track_artist}")
            print(f"Playing: {track_name} by {track_artist}")
        else:
            speak("No active Spotify device found. Please make sure Spotify is open and playing on a device.")
            print("No active device found.")
    else:
        speak(f"Sorry, I couldn't find any song called {song_name}.")
        print("No song found.")


def is_song_query(query):
    """Check if the query mentions a song."""
    # A more robust check might involve NLP, but for simplicity, checking the first word.
    return query.lower().startswith("play")


def main():
    global mic_stream
    while True:
        try:
            detect_wake_word()

            recognizer = sr.Recognizer()
            with sr.Microphone(device_index=AUDIO_DEVICE_INDEX) as source:
                print("Listening for query...")
                # Adjust for ambient noise once before listening for the command
                recognizer.adjust_for_ambient_noise(source) 
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5) # Add timeout and phrase_time_limit
                try:
                    query = recognizer.recognize_google(audio)
                    print(f"Query: {query}")

                    if is_song_query(query):
                        # Extract song name (simple approach, can be improved with more NLP)
                        song_name = query[len("play"):].strip() 
                        search_and_play(song_name)
                    elif is_visual_query(query):
                        speak("Please give me a moment to capture the image.")
                        image_path = capture_image()
                        if image_path:
                            process_query(query, image_path)
                        else:
                            speak("I couldn't capture an image. Please try again.")
                    else:
                        process_query(query)
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                    speak("Sorry, I didn't catch that. Could you please repeat?")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                    speak("I'm having trouble connecting to the speech recognition service.")
                except Exception as e:
                    print(f"An unexpected error occurred during query processing: {e}")
                    speak("Something went wrong. Please try again.")
        except Exception as e:
            print(f"Error in main loop: {e}")
            # Consider a short delay before retrying the main loop to prevent rapid error looping
            # import time
            # time.sleep(1)


if __name__ == "__main__":
    main()