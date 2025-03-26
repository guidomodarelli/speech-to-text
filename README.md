# Speech-to-Text Transcription Tool

A simple Python utility that converts speech from audio/video files to text
using OpenAI's transcription API.

## Features

- Transcribes audio/video files to text
- Saves transcriptions with timestamps
- Uses OpenAI's powerful GPT-4o transcription model

## Requirements

- Python 3.6+
- OpenAI API key
- Required Python packages:
  - openai
  - python-dotenv

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install openai python-dotenv
   ```
3. Create a `.env` file in the project directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Place your audio/video file in the project directory (default filename is
   "record.mp4")
2. Run the script:
   ```
   python speech-to-text.py
   ```
3. The transcription will be saved as a text file with a timestamp in the
   filename

## Customization

To use a different audio file, modify the `RECORDING_FILENAME` variable in the
script.

## How It Works

The script loads your OpenAI API key, opens the audio file in binary mode, and
sends it to the OpenAI transcription API. The resulting transcription is saved
to a text file and printed to the console.
