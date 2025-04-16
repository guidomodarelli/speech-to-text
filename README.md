# Speech-to-Text Transcription Tool

A Python utility that converts speech from audio/video files or YouTube URLs to text using OpenAI's transcription API.

## Features

- Transcribes audio from local audio/video files or YouTube URLs
- Converts various input formats to MP3
- Automatically splits large audio files into manageable chunks for transcription
- Intelligently combines transcriptions from chunks, handling overlaps
- Saves final transcriptions to the `transcriptions/` directory
- Saves processed MP3 audio files to the `outputs/` directory
- Uses OpenAI's `gpt-4o-mini-transcribe` model for transcription and `gpt-4o-mini` for processing
- Sanitizes filenames derived from YouTube titles

## Requirements

- Python 3.6+
- OpenAI API key
- `ffmpeg` installed and available in your system PATH (required for audio splitting and conversion)
- Required Python packages:
  - openai
  - python-dotenv
  - yt-dlp

## Setup

1. Clone this repository
2. Install `ffmpeg`. Instructions vary by operating system (e.g., `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS). Verify installation by running `ffmpeg -version`.
3. Install the required Python packages:
   ```bash
   pip install openai python-dotenv yt-dlp
   ```
4. Create a `.env` file in the project directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
5. (Optional) Add `MAX_PROCESSING_DURATION` to your `.env` file to limit the processing time (in seconds) for very long files:
   ```
   # Process only the first 10 minutes (600 seconds)
   MAX_PROCESSING_DURATION=600
   ```

## Usage

Run the script from the command line, providing either a YouTube URL or a local file path.

**Transcribe a YouTube video:**

```bash
python run.py -u "https://www.youtube.com/watch?v=your_video_id"
```

**Transcribe a local audio/video file:**

```bash
python run.py -f "/path/to/your/audio_or_video.wav"
```

**Specify output directory and filename base:**

```bash
python run.py -f "my_recording.mp4" -d "/custom/output/folder" -o "meeting_notes"
```

- The processed MP3 audio file will be saved in the specified output directory (default: `outputs/`).
- The final transcription `.txt` file will be saved in the `transcriptions/` directory.
- Intermediate chunk transcriptions are saved in `transcriptions/transcriptions_chunks/`.

## Customization

- **Input Source:** Use `-u` for YouTube URLs or `-f` for local files (required).
- **Output Directory:** Use `-d` to specify where the processed MP3 file should be saved (default: `outputs/`).
- **Output Filename Base:** Use `-o` to set a base name for the output MP3 file. The script might append the sanitized YouTube title or use the input filename stem.
- **Max Processing Duration:** Set the `MAX_PROCESSING_DURATION` environment variable in `.env` to limit processing time (in seconds).
- **Chunk Duration:** Modify the `MAX_CHUNK_DURATION` constant within the `run.py` script to change the length (in minutes) of audio chunks (default: 7 minutes).

## How It Works

1.  The script parses command-line arguments to get the input source (YouTube URL or local file path) and output preferences.
2.  If a YouTube URL is provided, `yt-dlp` downloads the audio track and saves it as an MP3 file in the specified output directory.
3.  If a local file path is provided, `ffmpeg` converts it to MP3 format (if necessary) and saves it to the output directory.
4.  The script checks the audio duration. If it exceeds the chunk duration limit (or if `MAX_PROCESSING_DURATION` is set), `ffmpeg` splits the audio into overlapping chunks.
5.  Each audio chunk is sent individually to the OpenAI transcription API (`gpt-4o-mini-transcribe`).
6.  If chunks were used, the script uses another OpenAI model (`gpt-4o-mini`) to intelligently combine the transcriptions from adjacent chunks, resolving overlaps at the boundaries.
7.  The final, combined transcription is saved as a `.txt` file (with a timestamp and sanitized title/filename) in the `transcriptions/` directory and printed to the console.
