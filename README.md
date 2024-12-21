# JARVIS: Intelligent Voice Assistant

JARVIS is a Python-based voice assistant designed to interact with users through voice commands and queries. It leverages multiple libraries and APIs for wake word detection, speech recognition, text-to-speech, and image-based query handling.

---

## Features

- **Wake Word Detection**: Activates on the command `Hey Jarvis` using the OpenWakeWord model.
- **Speech Recognition**: Recognizes voice queries with the `speech_recognition` library.
- **Text-to-Speech**: Responds audibly using the `pyttsx3` text-to-speech engine.
- **Visual Query Processing**: Captures images using a camera for queries requiring visual context.
- **Integration with Unify API**: Processes and answers queries using Unify's GPT-based models.

---

## Prerequisites

Ensure you have the following installed and configured:

1. **Python 3.7 or higher**
2. **Required Python Libraries**:
   Install these using `pip`:
   ```bash
   pip install pyttsx3 opencv-python numpy openwakeword pyaudio python-dotenv speechrecognition unify
   ```
3. **Unify API Key**:
   Add your Unify API key to a `.env` file in the project directory:
   ```env
   API_KEY=your_api_key_here
   ```
4. **ONNX Wake Word Model**:
   Ensure the file `hey_jarvis_v0.1.onnx` is present in the project directory.

---

## Project Structure

JARVIS follows a modular directory structure:

- `jarvis_voice_control.py`: Main script for running the assistant.
- `hey_jarvis_v0.1.onnx`: Model file for wake word detection.
- `requirements.txt`: Dependencies file listing all required Python libraries.
- `run_jarvis.sh`: Optional shell script for activation.
- `test.py`: Optional script for component testing.
- `captured_image.jpg`: Temporary file for visual query images.
- `.env`: Environment file storing the API key securely.

---

## How It Works

1. **Startup**:
   - The system initializes the wake word detection model, text-to-speech engine, and microphone input.

2. **Wake Word Detection**:
   - Continuously listens for the wake word (`Hey Jarvis`).

3. **Query Handling**:
   - Prompts the user for a query upon detecting the wake word.
   - Determines whether the query requires visual input.
   - Processes queries using the Unify API (text-only or text-with-image).

4. **Response**:
   - Speaks the response back to the user via text-to-speech.

---

## Usage

To use JARVIS:

1. **Activate JARVIS**:
   Run the main script using:
   ```bash
   python jarvis_voice_control.py
   ```

2. **Interact**:
   - Trigger the assistant by saying "Hey Jarvis."
   - Follow with your query when prompted.
   - Example queries:
     - "What is the weather today?"
     - "What is this object?" (requires image capture).

---

## Configuration Options

Adjust the following settings in `jarvis_voice_control.py`:

- **Wake Word**: Modify the `WAKE_WORD` variable to change the activation phrase.
- **Audio Input Device**: Set `AUDIO_DEVICE_INDEX` for a specific microphone.
- **Text-to-Speech Voice**: Tweak properties like rate, volume, or voice in the `tts_engine`.

---

## Troubleshooting

Common issues and solutions:

1. **Wake Word Detection Fails**:
   - Verify the presence of `hey_jarvis_v0.1.onnx` in the correct directory.
   - Check that the microphone permissions are enabled.

2. **No API Response**:
   - Confirm the API key in `.env` is valid.
   - Ensure internet connectivity is stable.

3. **Speech Recognition Errors**:
   - Update the `AUDIO_DEVICE_INDEX` for the intended microphone.

4. **Pyaudio Installation Issues**:
   - For Linux systems:
     ```bash
     sudo apt-get install portaudio19-dev python3-pyaudio
     ```

---

## Future Enhancements

Planned upgrades include:

- Advanced image query handling with computer vision models.
- Home automation system integration.

---

## License

This project is licensed under the MIT License. Feel free to use, modify, and share it.
