import os
import base64
import time
import pyttsx3
import cv2
import numpy as np
from dotenv import load_dotenv
import openwakeword as oww
from openwakeword.model import Model
import pyaudio
import argparse
import unify
import speech_recognition as sr

# Load .env file
load_dotenv()

# Download models needed for tests
oww.utils.download_models(['Hey Jarvis'])

# Access the API key
api_key = os.getenv('API_KEY')

# Configuration
UNIFY_API_KEY = api_key

WAKE_WORD = "Hey Jarvis"
AUDIO_DEVICE_INDEX = None  # Set this to None for default microphone (laptop mic)
BLUETOOTH_DEVICE_NAME = "Sana's mac microphone"

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 155)
tts_engine.setProperty('volume', 0.9)
# tts_engine.setProperty('voice', 'com.apple.speech.synthesis.voice.Samantha')

# Set model path
model_path = "hey jarvis"

# Initialize OpenWakeWord model
oww_model = Model(wakeword_models=[model_path], inference_framework='onnx')

# Setup for audio streaming
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

def speak(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def detect_wake_word():
    """Detect the wake word using OpenWakeWord."""
    print("🔊 Listening for wake word, listening for your query...")
    try:
        while True:
            # Read audio data from the stream
            raw_data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # Convert the audio data into a NumPy array
            audio_data = np.frombuffer(raw_data, dtype=np.int16)

            del raw_data

            # Feed to OpenWakeWord model
            oww_model.predict(audio_data)

            # Check if the wake word is detected
            for mdl in oww_model.prediction_buffer.keys():
                scores = list(oww_model.prediction_buffer[mdl])
                # print(f"Wake word detection scores: {scores}")
                if scores[-1] > 0.3:  # Wake word detected with score > 0.3
                    oww_model.prediction_buffer[mdl].clear()
                    print("🎤 Wake word detected!")
                    return
    except Exception as e:
        print(f"❌ Error while detecting wake word: {e}")

def capture_image():
    """Capture an image using the device camera."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        image_path = "captured_image.jpg"
        cv2.imwrite(image_path, frame)
        cap.release()
        print("📸 Image captured successfully.")
        return image_path
    else:
        cap.release()
        raise RuntimeError("❌ Failed to capture image.")

def is_visual_query(query):
    """Determine if the query requires visual input using Unify API."""
    return False
    # print(f"🔍 Checking if visual information is needed for query: {query}")
    return False
    client = unify.Unify("gpt-4o@openai", api_key=UNIFY_API_KEY)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Does the following query need an image to answer?\nQuery: {query}\nRespond with 'Yes' or 'No'"}
            ],
        }
    ]
    response = client.generate(messages=messages, max_completion_tokens=3)
    print(f"🤖 Visual query decision: {response}")
    return "yes" in response

def process_query(query, image_path=None):
    """Process the query using Unify API, optionally with an image."""
    print(f"🤖 Processing query: {query}")
    client = unify.Unify("gpt-4o@openai", api_key=UNIFY_API_KEY)  # Specify the model name
    if image_path:
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare the messages payload
        messages = [
            {
                "role": "system",
                "content": "Provide precise and accurate answers without unnecessary elaboration."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    },
                ],
            }
        ]
    else:
        # Prepare the messages payload for text-only queries
        messages = [
            {
                "role": "system",
                "content": "Provide precise and accurate answers without unnecessary elaboration."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
                ],
            }
        ]
    
    # Make the API call
    try:
        response = client.generate(
            messages=messages,
            max_completion_tokens=150  # Adjust token limit for shorter responses
        )
        print(f"💬 Response: {response}")
        return response
    except Exception as e:
        print(f"❌ Error while processing query: {e}")
        return "❌ Sorry, I encountered an error while processing your request."

def main():
    global mic_stream  # Use the global mic_stream to manage its state

    while True:  # Outer loop to keep listening for wake word
        try:
            # Detect the wake word first
            detect_wake_word()
            # print("🎧 Wake word detected! \n❗ Listening for your query...")

            time.sleep(1.5)

            recognizer = sr.Recognizer()
            with sr.Microphone(device_index=AUDIO_DEVICE_INDEX) as source:
                try:
                    # Listen for the query
                    audio = recognizer.listen(source)
                    query = recognizer.recognize_google(audio)
                    print(f"📝 Query received: {query}")

                    # Check if query requires an image
                    if is_visual_query(query):
                        print("🔲 Visual input required. Capturing image...")
                        image_path = capture_image()
                        print("💡 Processing query with image...")
                        response = process_query(query, image_path)
                    else:
                        print("💬 Processing query with text only...")
                        response = process_query(query)

                    # Speak the response
                    # print(f"💬 Response: {response}")
                    speak(response)

                except sr.UnknownValueError:
                    print("❌ Could not understand audio, listening again.")
                    continue
                except Exception as e:
                    print(f"❌ Error during query handling: {e}")
                    continue

        except OSError as e:
            # Handle stream errors and reset if necessary
            if e.args[0] == -9988:  # Stream closed
                print("🔄 Stream closed. Reinitializing...")
                mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
            else:
                print(f"❌ Unexpected audio error: {e}")
                break

if __name__ == "__main__":
    main()
