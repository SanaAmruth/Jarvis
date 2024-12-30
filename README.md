# Jarvis Assistant Project

## Overview
This project implements a voice-activated assistant named **Jarvis**. It supports various functionalities, including:

- **Wake Word Detection**: Listens for the wake word "Hey Jarvis."
- **Speech Recognition**: Processes voice queries.
- **Spotify Integration**: Pauses, plays, and searches for songs using the Spotify API.
- **Image Capture**: Captures images via the system's camera.
- **Query Processing**: Handles queries that may or may not require visual input using the Unify API.

---

## Features

- **Wake Word Detection**:
  - Uses OpenWakeWord to listen for "Hey Jarvis."
  - Automatically pauses Spotify playback when the wake word is detected.

- **Speech Recognition**:
  - Leverages Google Speech Recognition via the `speech_recognition` library to convert speech to text.

- **Spotify Integration**:
  - Search for songs and start playback on active devices.
  - Pause playback when the wake word is detected.

- **Image Capture**:
  - Captures an image using OpenCV and sends it for query analysis if required.

- **Query Processing**:
  - Determines if a query needs visual input using the Unify API.
  - Processes textual or visual queries and provides answers.

---

## Prerequisites

- **Python**: Python 3.10
- **Spotify Account**: A Spotify Premium account
- **Unify API Key**: Obtain a Unify API key(50$ free credits will be given while creating account) to process queries

---

## Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and define the following variables:

```env
UNIFY_API_KEY=<your_unify_api_key>
CLIENT_ID=<your_spotify_client_id>
CLIENT_SECRET=<your_spotify_client_secret>
REDIRECT_URI=<your_redirect_uri>
```

### 4. Download Wake Word Models
```python
import openwakeword as oww
oww.utils.download_models(['Hey Jarvis'])
```

---

## Running the Project

To start the assistant, execute the following command:
```bash
python jarvis.py
```

---

## Functionality Breakdown

### 1. Wake Word Detection
- Listens for "Hey Jarvis" using OpenWakeWord.
- When detected, pauses Spotify playback.

### 2. Speech Recognition
- After detecting the wake word, listens for the user’s query.
- Converts voice input to text using Google Speech Recognition.

### 3. Spotify Integration
- **Pause Playback**: Automatically pauses playback when the wake word is detected.
- **Search and Play**: Extracts the song name from queries like "Play [song name]." Searches and plays the song on the active Spotify device.

### 4. Image Capture
- Captures an image using OpenCV if a query requires visual input.

### 5. Query Processing
- Uses the Unify API to determine if a query requires visual input.
- Processes the query and responds using text-to-speech.

---

## Code Structure

- `jarvis.py`: Entry point of the application.
- `requirements.txt`: Contains all dependencies.

---

## Key Libraries

- **OpenWakeWord**: For wake word detection.
- **SpeechRecognition**: For voice-to-text conversion.
- **Spotipy**: For Spotify API interaction.
- **Unify API**: For query processing and visual query handling.
- **OpenCV**: For capturing images.

---

## Troubleshooting

1. **Spotify Playback Issues**:
   - Ensure Spotify is running and connected to an active device.
   - Check that the Spotify client ID, secret, and redirect URI are correct.

2. **Wake Word Not Detected**:
   - Ensure the wake word model is correctly downloaded and set up.

3. **Environment Variables Not Loaded**:
   - Confirm that the `.env` file is present and correctly configured.

---

## Contributing
Contributions are welcome! Fork the repository, make your changes, and create a pull request.

---

## License
This project is licensed under the MIT License.

---
