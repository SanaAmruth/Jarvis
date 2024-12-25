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
AUDIO_DEVICE_INDEX = None
BLUETOOTH_DEVICE_NAME = "Sana's mac microphone"

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 172)
tts_engine.setProperty('volume', 0.9)

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
    print("Listening for wake word...")
    while True:
        raw_data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        oww_model.predict(audio_data)

        del raw_data

        for mdl in oww_model.prediction_buffer.keys():
            if oww_model.prediction_buffer[mdl][-1] > 0.3:
                oww_model.prediction_buffer[mdl].clear()
                print("Wake word detected!")
                return

def capture_image():
    """Capture an image using the device camera."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        image_path = "captured_image.jpg"
        cv2.imwrite(image_path, frame)
        cap.release()
        return image_path
    else:
        cap.release()
        raise RuntimeError("Failed to capture image.")

def is_visual_query(query):
    """Determine if the query requires visual input."""
    return False

def process_query(query, image_path=None):
    """Process the query using Unify API."""
    client = unify.Unify("gpt-4o@openai", api_key=UNIFY_API_KEY)
    messages = [{"role": "system", "content": "Provide precise and accurate answers."}]
    
    if image_path:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": query}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}],
        })
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
    
    try:
        response = client.generate(messages=messages, max_completion_tokens=150)
        return response
    except Exception as e:
        return f"Error processing query: {e}"

def main():
    global mic_stream
    while True:
        try:
            # Detect the wake word
            detect_wake_word()

            # Use the recognizer to listen for the query
            recognizer = sr.Recognizer()
            with sr.Microphone(device_index=AUDIO_DEVICE_INDEX) as source:
                print("Listening for query...")
                audio = recognizer.listen(source)
                try:
                    # Recognize the speech
                    query = recognizer.recognize_google(audio)
                    print(f"Query: {query}")

                    # Check if the query requires an image
                    if is_visual_query(query):
                        image_path = capture_image()
                        response = process_query(query, image_path)
                    else:
                        response = process_query(query)

                    # Respond and speak
                    print(f"Response: {response}")
                    speak(response)
                except Exception as e:
                    continue
        except Exception as e:
            continue


if __name__ == "__main__":
    main()
