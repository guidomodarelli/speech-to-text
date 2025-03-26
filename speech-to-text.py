#!/usr/bin/env python3

from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import os

# Load environment variables from .env file
load_dotenv()

SCRIPT_DIR = Path(__file__).parent.resolve()
# Get recording filename from env var or use default
RECORDING_FILENAME = os.environ.get("RECORDING_FILENAME", "record.mp4")
# Get output format from env var or use default
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "transcription_{timestamp}.txt")
FILE_PATH = SCRIPT_DIR / RECORDING_FILENAME

client = OpenAI()
# In Python, "rb" is a mode parameter used when opening files. It means "read
# binary".
#
# - "r" indicates that the file is opened for reading (not writing)
# - "b" indicates binary mode, which is important when dealing with non-text
#   files like audio
#
# When you open the audio file with `open(FILE_PATH, "rb")`, you're opening it
# in binary read mode, which is necessary because audio files contain binary
# data rather than text. This allows Python to properly read the file's contents
# without attempting to decode it as text.
with open(FILE_PATH, "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file
    )

# Save transcription to text file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = OUTPUT_FORMAT.format(timestamp=timestamp)
output_path = SCRIPT_DIR / output_filename
with open(output_path, "w") as text_file:
    text_file.write(transcription.text)

# Also print the transcription
print(f"Transcription saved to {output_path}")
print(transcription.text)