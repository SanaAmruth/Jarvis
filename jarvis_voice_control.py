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
model_path = "hey_jarvis_v0.1.onnx"

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
        audio_data = np.frombuffer(mic_stream.read(CHUNK_SIZE), dtype=np.int16)
        
        # Feed to OpenWakeWord model
        prediction = oww_model.predict(audio_data)
        
        # Check if the wake word is detected
        for mdl in oww_model.prediction_buffer.keys():
            scores = list(oww_model.prediction_buffer[mdl])
            # print(scores)
            if scores[-1] > 0.3:  # Wake word detected with score > 0.5
                print("Wake word detected.")
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
    """Determine if the query requires visual input using Unify API."""
    # client = unify.Unify("gpt-4o-mini@openai", api_key=UNIFY_API_KEY)
    print("Checking if visual information is needed or not")
    # client = unify.Unify("gemini-1.5-flash@openai", api_key=UNIFY_API_KEY)bH5V13s+e8JVZPv3+i4TRZOsNVTsriJvXkEwlSsA2-Q=
    # client = unify.Unify("gemini-1.5-flash@openai", api_key="bH5V13s+e8JVZPv3+i4TRZOsNVTsriJvXkEwlSsA2-Q=")
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
    print(response)
    # decision = response.get("choices", [{}])[0].get("text", "").strip().lower()
    return "yes" in response

def process_query(query, image_path=None):
    """Process the query using Unify API, optionally with an image."""
    client = unify.Unify("gpt-4o@openai", api_key=UNIFY_API_KEY)  # Specify the model name
    if image_path:
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare the messages payload
        messages = [
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
            max_completion_tokens=300
        )
        print(response)
        # Extract the generated response text
        # return response.get("choices", [{}])[0].get("text", "").strip()
        return response
    except Exception as e:
        print(f"Error while processing query: {e}")
        return "Sorry, I encountered an error while processing your request."

def main():
    while True:  # Outer loop to keep listening for wake word
        # Detect the wake word first
        detect_wake_word()
        print("Wake word detected. \n *********Listening for query*********")

        time.sleep(1.5)

        speak("Hi Mr.Sana, How may I help you today")

        recognizer = sr.Recognizer()
        with sr.Microphone(device_index=AUDIO_DEVICE_INDEX) as source:
            try:
                audio = recognizer.listen(source)
                query = recognizer.recognize_google(audio)
                print(f"Query: {query}")

                # Check if query requires an image
                if is_visual_query(query):
                    print("Visual input required. Capturing image...")
                    image_path = capture_image()
                    print("Processing query with image...")
                    response = process_query(query, image_path)
                else:
                    print("Processing query with text only...")
                    response = process_query(query)

                # Speak the response
                print(f"Response: {response}")
                speak(response)

            except sr.UnknownValueError:
                speak("Sorry, I didn't catch that.")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()