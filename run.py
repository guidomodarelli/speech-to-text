#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import os
import sys
import subprocess
import argparse
import re  # Add re import for sanitization
import shutil
import urllib.parse as urlparse # Add urllib.parse import
from openai import OpenAI
from dotenv import load_dotenv
from lib.ai_tools import AITools
from lib.logger import log_error, log_success, log_info, log_warning
from lib.youtube import YouTube, YtdlpYouTube

# Load environment variables from .env file
load_dotenv(override=True)

ROOT_DIR = Path(__file__).parent.resolve()
# Get output format from env var or use default
OUTPUT_FORMAT = "transcription_{timestamp}.txt"

# Define directories for transcriptions
TRANSCRIPTIONS_DIR = ROOT_DIR / "transcriptions"
TRANSCRIPTIONS_CHUNKS_DIR = TRANSCRIPTIONS_DIR / "transcriptions_chunks"

# Number of words to use when processing chunk boundaries
BOUNDARY_WORD_COUNT = 50
MAX_CHUNK_DURATION = 7  # Maximum duration of each audio chunk in minutes
# Maximum time to process from the audio file (in seconds, None for entire file)
MAX_PROCESSING_DURATION: float | None = None
try:
    MAX_PROCESSING_DURATION = float(os.environ.get("MAX_PROCESSING_DURATION") or "")
except ValueError:
    log_error(f"Invalid MAX_PROCESSING_DURATION: {MAX_PROCESSING_DURATION}. Using entire file.")
    MAX_PROCESSING_DURATION = None

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds using ffmpeg"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        log_error(f"Error getting audio duration: {e}")
        return 0

def split_audio_file(file_path: Path, chunk_duration_minutes=MAX_CHUNK_DURATION, max_duration=MAX_PROCESSING_DURATION):
    """
    Split audio file into chunks of specified duration with overlap to avoid cutting words.

    Args:
        file_path: Path to the audio file
        chunk_duration_minutes: Duration of each chunk in minutes
        max_duration: Maximum duration to process in seconds (None for entire file)

    Returns:
        List of paths to the chunk files
    """
    try:
        # Check if ffmpeg is installed
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            log_error("ffmpeg not found. Please install ffmpeg.")
            return []

        # Create directory for chunks
        parent_dir = file_path.parent
        chunks_dir = parent_dir / f"{file_path.stem}_chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Convert minutes to seconds
        chunk_duration_seconds = int(chunk_duration_minutes * 60)

        # Get the total duration
        total_duration = get_audio_duration(file_path)
        if total_duration <= 0:
            log_error("Could not determine file duration")
            return []

        # Limit processing duration if specified
        if max_duration is not None and max_duration > 0:
            processing_duration = min(total_duration, max_duration)
            log_info(f"Processing only the first {processing_duration} seconds of audio (out of {total_duration} total)")
        else:
            processing_duration = total_duration
            log_info(f"Processing entire audio file ({total_duration} seconds)")

        chunk_files = []
        overlap_seconds = 3 # Overlap duration in seconds

        # Generate chunks with overlap
        for i, start_time in enumerate(range(0, int(processing_duration), chunk_duration_seconds)):
            # Adjust start time to include overlap from previous chunk (except for first chunk)
            adjusted_start = max(0, start_time - overlap_seconds) if i > 0 else 0

            # Adjust duration to include overlap with next chunk
            # For the last chunk, make sure we don't exceed the processing duration
            if start_time + chunk_duration_seconds >= processing_duration:
                # Last chunk - go until the end of the processing duration
                duration = processing_duration - adjusted_start
            else:
                # Add overlap to duration
                duration = (start_time + chunk_duration_seconds + overlap_seconds) - adjusted_start

            output_file = chunks_dir / f"chunk_{i:03d}{file_path.suffix}"

            # Use ffmpeg to extract the chunk
            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-ss', str(adjusted_start),
                '-t', str(duration),
                '-c', 'copy',
                '-y',  # Overwrite output files without asking
                str(output_file)
            ]

            subprocess.run(cmd, check=True)
            chunk_files.append(output_file)

        log_success(f"Split audio into {len(chunk_files)} chunks with {overlap_seconds}-second{'s' if overlap_seconds > 1 else ''} overlap in {chunks_dir}")
        return chunk_files

    except Exception as e:
        log_error(f"Error splitting audio file: {e}")
        return []

def get_boundary_text(text: str, is_start: bool, word_count: int = BOUNDARY_WORD_COUNT) -> str:
    """
    Extract the first or last N words from a text.

    Args:
        text: The text to extract from
        is_start: If True, get the first N words; if False, get the last N words
        word_count: Number of words to extract

    Returns:
        The extracted boundary text
    """
    words = text.split(" ")

    if not words:
        return ""

    if is_start:
        # Get first N words
        boundary_words = words[:min(word_count, len(words))]
    else:
        # Get last N words
        boundary_words = words[-min(word_count, len(words)):]

    return " ".join(boundary_words)

def combine_chunks_sequentially(chunk_transcriptions: list[str], ai_tools: AITools) -> str:
    """Combine chunks sequentially with overlaps (1-2, 2-3, 3-4, etc.) focusing on boundaries"""
    if not chunk_transcriptions:
        return ""

    if len(chunk_transcriptions) == 1:
        return chunk_transcriptions[0]

    # Create a list of processed chunks, starting with the first chunk as-is
    processed_chunks = [exclude_last_boundary_words(chunk_transcriptions[0])]

    # For each subsequent chunk
    for i in range(1, len(chunk_transcriptions)):
        log_info(f"Processing boundary between chunks {i}/{len(chunk_transcriptions)-1}...")

        prev_chunk = chunk_transcriptions[i-1]
        if i > 1:
            prev_chunk = processed_chunks[-1]
        curr_chunk = chunk_transcriptions[i]

        # Extract the last N words from the first chunk
        end_of_first = get_boundary_text(prev_chunk, is_start=False)

        # Extract the first N words from the second chunk
        start_of_second = get_boundary_text(curr_chunk, is_start=True)

        # Combine the boundaries
        boundary_text = ai_tools.resolve_overlap_boundaries(end_of_first, start_of_second)

        # Extract words that aren't part of our boundary overlap processing
        words_in_curr_chunk = curr_chunk.split(" ")
        processed_chunks[-1] = exclude_last_boundary_words(processed_chunks[-1])
        if len(words_in_curr_chunk) > BOUNDARY_WORD_COUNT:
            # Keep everything except the first N words that were already processed in the boundary
            remaining_text = " ".join(words_in_curr_chunk[BOUNDARY_WORD_COUNT:])
            processed_chunks.append(boundary_text + " " + remaining_text)
        else:
            # If the chunk is small, just use the boundary text
            processed_chunks.append(boundary_text)

    # Join all processed chunks with spaces
    return " ".join(processed_chunks)

def exclude_last_boundary_words(chunk: str) -> str:
    """ Exclude the last N words from the previous chunk to avoid duplication in the boundary """
    return " ".join(chunk.split(" ")[:-BOUNDARY_WORD_COUNT])

def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that are invalid in filenames."""
    # Remove characters that are not alphanumeric, underscore, hyphen, or period
    sanitized = re.sub(r'[^\w\-.]', '_', filename)
    # Replace multiple consecutive underscores with a single one
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores or periods
    sanitized = sanitized.strip('_.')
    # Limit length to avoid issues with long filenames (e.g., 100 chars)
    # Allow slightly longer for potential ID additions
    return sanitized[:150] if len(sanitized) > 150 else sanitized

def generate_output_filename(args, video_title: str | None, file_path: Path, timestamp: str, video_id: str | None = None) -> str:
    """Generates the output filename for the transcription."""
    transcription_base_name = f"transcription_{timestamp}"
    if args.youtube_url and video_title and video_id:
        sanitized_title = sanitize_filename(video_title)
        # Include video ID in the filename
        return f"{transcription_base_name}-{video_id}-{sanitized_title}.txt"
    else:
        audio_base = file_path.stem
        # Handle cases where stem might already contain 'transcription' or timestamp patterns if re-run
        # A simpler approach is just using the sanitized base name
        sanitized_base = sanitize_filename(audio_base)
        return f"{sanitized_base}_transcription_{timestamp}.txt"

def save_transcription(transcription_text: str, output_path: Path):
    """Saves the transcription text to the specified file path."""
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as text_file:
            text_file.write(transcription_text)
        log_success(f"Transcription saved to {output_path}")
    except Exception as e:
        log_error(f"Error saving transcription to {output_path}: {e}")

def clean_chunks_directory(chunks_dir: Path):
    """Remove chunks directory and its contents if it exists"""
    # chunks_dir = file_path.parent / f"{file_path.stem}_chunks" # Removed calculation from file_path
    if chunks_dir.exists() and chunks_dir.is_dir(): # Check if it's actually a directory
        log_info(f"Removing existing chunks directory: {chunks_dir}")
        try:
            shutil.rmtree(chunks_dir) # Use shutil.rmtree for simplicity
            log_success(f"Removed chunks directory: {chunks_dir}")
        except Exception as e:
            log_error(f"Error removing directory {chunks_dir}: {e}")
    elif chunks_dir.exists():
        log_warning(f"Path exists but is not a directory, cannot clean: {chunks_dir}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Transcribe audio from a YouTube video or a local file.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-u", "--youtube-url", type=str, help="URL of the YouTube video to transcribe.")
    input_group.add_argument("-f", "--file-path", type=str, help="Path to the local audio/video file to transcribe.")

    parser.add_argument("-d", "--output-dir", type=str, default=str(ROOT_DIR / "outputs"), help="Directory to save the processed audio file.")
    parser.add_argument("-o", "--output-filename", type=str, help="Base filename for the processed audio file (e.g., output). Extension and title (if YouTube) will be added.")

    return parser.parse_args()

def get_youtube_video_id(url):
    """Extracts the YouTube video ID from a URL."""
    if url is None:
        return None
    try:
        parsed_url = urlparse.urlparse(url)
        if parsed_url.hostname in ('youtu.be',):
            return parsed_url.path[1:]
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                query = urlparse.parse_qs(parsed_url.query)
                return query.get('v', [None])[0]
            if parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/')[2]
            if parsed_url.path.startswith('/v/'):
                return parsed_url.path.split('/')[2]
    except Exception as e:
        log_error(f"Error parsing YouTube URL to get video ID: {e}")
    return None

def main():
    args = parse_args()

    client = OpenAI()
    ai_tools = AITools(client)

    # Base directories (can be overridden for YouTube)
    output_dir = Path(args.output_dir)
    transcriptions_dir = TRANSCRIPTIONS_DIR
    transcriptions_chunks_dir = TRANSCRIPTIONS_CHUNKS_DIR

    video_id = None # Initialize video_id

    # Determine output path and base filename
    output_dir.mkdir(parents=True, exist_ok=True) # Create base output dir

    # Construct base filename
    if args.output_filename:
        base_filename = args.output_filename
    elif args.youtube_url:
        base_filename = "youtube_audio" # Keep simple, ID will be added later
    elif args.file_path:
        base_filename = Path(args.file_path).stem
    else:
        base_filename = "output"

    if args.youtube_url:
        log_info(f"Processing YouTube URL: {args.youtube_url}")
        video_id = get_youtube_video_id(args.youtube_url)
        if not video_id:
            log_error("Could not extract YouTube video ID. Exiting.")
            sys.exit(1)
        log_info(f"Extracted YouTube Video ID: {video_id}")

        # Define ID-specific directories
        output_dir = output_dir / video_id # Now points to outputs/VIDEO_ID
        transcriptions_dir = transcriptions_dir / video_id # Now points to transcriptions/VIDEO_ID
        transcriptions_chunks_dir = transcriptions_dir / "transcriptions_chunks" # Now points to transcriptions/VIDEO_ID/transcriptions_chunks

        # Ensure ID-specific directories exist
        output_dir.mkdir(parents=True, exist_ok=True)
        transcriptions_dir.mkdir(parents=True, exist_ok=True)
        transcriptions_chunks_dir.mkdir(parents=True, exist_ok=True)

        youtube: YouTube = YtdlpYouTube()
        # Include video ID in the final mp3 filename
        final_filename = f"{base_filename}-{video_id}.mp3"
        file_path = output_dir / final_filename # Path is now outputs/VIDEO_ID/youtube_audio-VIDEO_ID.mp3

        if not youtube.download_audio(args.youtube_url, file_path):
            log_error("Failed to download YouTube video. Exiting.")
            sys.exit(1)

        log_success(f"Successfully downloaded and saved YouTube video as {file_path}")
        # Update base_filename to include ID for transcription filename generation
        base_filename = f"{base_filename}-{video_id}"

    elif args.file_path:
        log_info(f"Processing local file: {args.file_path}")
        input_file = Path(args.file_path)
        if not input_file.exists():
            log_error(f"Input file not found: {input_file}")
            sys.exit(1)

        # Ensure base directories exist if not YouTube
        transcriptions_dir.mkdir(parents=True, exist_ok=True)
        transcriptions_chunks_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / f"{base_filename}.mp3"

        if file_path.exists():
            log_info(f"Removing existing file: {file_path}")
            file_path.unlink()
            log_info(f"Removed existing file: {file_path}")

        if input_file.suffix.lower() != ".mp3":
            log_info(f"Converting {input_file} to MP3 format at {file_path}...")
            try:
                # Check if ffmpeg is installed
                try:
                    subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    log_error("ffmpeg not found. Please install ffmpeg to convert audio files.")
                    sys.exit(1)

                # Construct ffmpeg command for conversion
                cmd = [
                    'ffmpeg',
                    '-i', str(input_file),
                    '-vn',  # No video
                    '-acodec', 'libmp3lame', # Use MP3 codec
                    '-ab', '192k', # Audio bitrate
                    '-ar', '44100', # Audio sample rate
                    '-y',  # Overwrite output file without asking
                    str(file_path)
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                log_success(f"Successfully converted {input_file} to {file_path}")
            except subprocess.CalledProcessError as e:
                log_error(f"Error converting file with ffmpeg: {e}")
                log_error(f"ffmpeg stderr: {e.stderr.decode()}")
                sys.exit(1)
            except Exception as e:
                log_error(f"An unexpected error occurred during conversion: {e}")
                sys.exit(1)
        elif input_file != file_path:
            log_info(f"Copying {input_file} to {file_path}...")
            try:
                shutil.copyfile(input_file, file_path)
                log_success(f"Successfully copied {input_file} to {file_path}")
            except Exception as e:
                log_error(f"Error copying file: {e}")
                sys.exit(1)
        else:
            log_info(f"Input file {input_file} is already the target MP3 file.")
    else:
        # This case should not happen due to mutually exclusive group in argparse
        log_error("No input source specified (YouTube URL or file path).")
        sys.exit(1)

    # Define the expected chunks directory path based on the final file_path
    expected_chunks_dir = file_path.parent / f"{file_path.stem}_chunks"
    clean_chunks_directory(expected_chunks_dir) # Pass the specific dir path

    # Ensure the file exists before proceeding
    if not file_path.exists():
        log_error(f"Error: Processed file not found at {file_path}")
        sys.exit(1)

    # Split the audio into chunks
    log_info(f"Splitting audio into {MAX_CHUNK_DURATION}-minute chunks...")
    # split_audio_file creates chunks relative to file_path.parent, which is correct (either base output or ID-specific output)
    audio_chunks = split_audio_file(file_path)

    if not audio_chunks:
        log_warning("No audio chunks were created. Trying to transcribe the entire file...")
        # Fall back to transcribing the whole file
        transcription_text = ai_tools.process_audio_transcription(file_path)
    else:
        log_info(f"Transcribing {len(audio_chunks)} audio chunks...")

        # First transcribe all chunks individually
        chunk_transcriptions: list[str] = []
        for i, chunk_file in enumerate(audio_chunks):
            log_info(f"Transcribing chunk {i+1}/{len(audio_chunks)}...")
            chunk_transcription = ai_tools.process_audio_transcription(chunk_file)
            chunk_transcriptions.append(chunk_transcription)

            # Save individual chunk transcription to the correct (potentially ID-specific) chunks dir
            chunk_transcription_filename = f"chunk_{i:03d}_transcription.txt"
            # Use the potentially ID-specific transcriptions_chunks_dir
            chunk_transcription_path = transcriptions_chunks_dir / chunk_transcription_filename
            try:
                with open(chunk_transcription_path, "w", encoding='utf-8') as chunk_text_file:
                    chunk_text_file.write(chunk_transcription)
                log_info(f"Saved chunk transcription to {chunk_transcription_path}")
            except Exception as e:
                log_error(f"Error saving chunk transcription {chunk_transcription_path}: {e}")

        # Combine chunks sequentially (1-2, 2-3, 3-4, etc.)
        log_info("Combining chunks sequentially...")
        transcription_text = combine_chunks_sequentially(chunk_transcriptions, ai_tools)

    # Generate filename and save transcription
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Pass video_id (which is None if not YouTube) to generate_output_filename
    # Use the potentially ID-specific base_filename for YouTube titles
    output_filename = generate_output_filename(args, base_filename, file_path, timestamp, video_id)
    # Use the potentially ID-specific transcriptions_dir
    output_path = transcriptions_dir / output_filename

    save_transcription(transcription_text, output_path)

    # Log the final transcription
    log_info("--- Final Transcription ---")
    log_info(transcription_text)
    log_info("--- End Transcription ---")


if __name__ == "__main__":
    main()
