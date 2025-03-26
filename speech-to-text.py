#!/usr/bin/env python3

from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import os
import sys
import subprocess

# Import pytube for YouTube downloads
try:
    from pytube import YouTube
except ImportError:
    print("pytube package is not installed. Install it using 'pip install pytube'")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

YOUTUBE_URL = os.environ.get("YOUTUBE_URL")

ROOT_DIR = Path(__file__).parent.resolve()
# Get recording filename from env var or use default
RECORDING_FILENAME = os.environ.get("RECORDING_FILENAME", "record.mp4")
# Get output format from env var or use default
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "transcription_{timestamp}.txt")
FILE_PATH = ROOT_DIR / RECORDING_FILENAME

# Function to check if file exists and ask for confirmation
def should_overwrite_file(file_path):
    if not file_path.exists():
        return True

    print(f"File already exists at {file_path}")
    response = input("Do you want to download and overwrite? (y/n): ").strip().lower()
    return response == 'y' or response == 'yes'

# Function to download YouTube video if URL is provided
def download_youtube_video(url, output_path):
    # First try with pytube
    if try_pytube_download(url, output_path):
        return True

    print("Pytube download failed, trying with yt-dlp...")
    return try_ytdlp_download(url, output_path)

def try_pytube_download(url, output_path):
    try:
        print(f"Downloading YouTube video from: {url}")
        yt = YouTube(url)

        # Get the audio stream with highest quality
        print("Downloading audio-only stream")
        audio_stream = yt.streams.filter(only_audio=True).first()

        # Choose file extension based on stream type
        is_audio_only = False

        if not audio_stream:
            # If no audio-only stream is available, get the lowest resolution video
            print("No audio-only stream found. Downloading lowest resolution video")
            audio_stream = yt.streams.filter(progressive=True).order_by('resolution').first()
        else:
            is_audio_only = True

        if not audio_stream:
            print("No suitable streams found")
            return False

        # Download the file
        print(f"Downloading: {audio_stream}")

        # If audio-only, modify the output path to use .mp3 extension
        if is_audio_only:
            # Get the directory and modify the filename to use .mp3
            parent_dir = output_path.parent
            new_filename = output_path.stem + ".mp3"
            output_path = parent_dir / new_filename
            global FILE_PATH
            FILE_PATH = output_path

        file_path = audio_stream.download(output_path=parent_dir, filename=output_path.name)
        print(f"Download completed: {file_path}")
        return True
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return False

def try_ytdlp_download(url, output_path):
    """Try downloading with yt-dlp as a fallback"""
    try:
        # Check if yt-dlp is installed
        try:
            subprocess.run(['yt-dlp', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("yt-dlp not found. Please install it with: pip install yt-dlp")
            return False

        # Modify output path to use .mp3 for audio downloads
        parent_dir = output_path.parent
        new_filename = output_path.stem + ".mp3"
        output_path = parent_dir / new_filename

        # Update the global FILE_PATH
        global FILE_PATH
        FILE_PATH = output_path

        # Set up the command to extract audio only and save to the specified path
        cmd = [
            'yt-dlp',
            '-f', 'bestaudio',  # Get best audio quality
            '-x',               # Extract audio
            '--audio-format', 'mp3',  # Convert to mp3
            '-o', str(output_path),
            url
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)

        if result.returncode == 0:
            print("Successfully downloaded audio using yt-dlp")
            return True
        else:
            print(f"yt-dlp exited with code {result.returncode}")
            return False
    except Exception as e:
        print(f"Error using yt-dlp: {e}")
        return False

client = OpenAI()

# Check if YouTube URL is provided and download the video
if YOUTUBE_URL:
    # Check if file exists and ask before downloading
    if should_overwrite_file(FILE_PATH):
        if not download_youtube_video(YOUTUBE_URL, FILE_PATH):
            print("Failed to download YouTube video. Exiting.")
            sys.exit(1)
        print(f"Successfully downloaded YouTube video to {FILE_PATH}")
    else:
        print(f"Using existing file at {FILE_PATH}")

# Ensure the file exists before proceeding
if not FILE_PATH.exists():
    print(f"Error: File not found at {FILE_PATH}")
    sys.exit(1)

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
output_path = ROOT_DIR / output_filename
with open(output_path, "w") as text_file:
    text_file.write(transcription.text)

# Also print the transcription
print(f"Transcription saved to {output_path}")
print(transcription.text)