#!/usr/bin/env python3

"""
Processes audio from a YouTube video or a local file to generate a text transcription.

This script performs the following steps:
1.  Parses command-line arguments to get the input source (YouTube URL or local file path)
    and output directory/filename options.
2.  If a YouTube URL is provided:
    - Extracts the video ID.
    - Creates specific output and transcription directories based on the video ID.
    - Downloads the audio track using yt-dlp.
    - Saves the audio as an MP3 file in the designated output directory.
3.  If a local file path is provided:
    - Checks if the file exists.
    - Converts the file to MP3 format using ffmpeg if it's not already MP3, placing it
      in the designated output directory.
    - Copies the file if it's already MP3 but not in the target location.
4.  Splits the resulting MP3 audio file into smaller chunks using ffmpeg, adding a slight
    overlap between chunks to avoid cutting words during transcription.
5.  Transcribes each audio chunk individually using the OpenAI API (Whisper model).
    Individual chunk transcriptions are saved.
6.  Combines the individual chunk transcriptions sequentially:
    - It identifies the overlapping text between adjacent chunks (last N words of the previous
      chunk and first N words of the current chunk).
    - Uses an AI model (via AITools) to intelligently merge these overlapping boundary sections.
    - Concatenates the processed chunks to form the final transcription.
7.  Generates a sanitized output filename including a timestamp and, if applicable,
    the YouTube video ID and title.
8.  Saves the final combined transcription to a text file in the designated transcriptions
    directory (potentially within a video ID subdirectory).
9.  Cleans up the temporary audio chunk files and directories.

Requires:
- Python 3.x
- `openai` library
- `python-dotenv` library
- `yt-dlp` library (for YouTube downloads)
- `ffmpeg` installed and available in the system PATH (for audio splitting and conversion)
- An OpenAI API key set in a .env file or environment variables.
"""

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
VTT_CHUNK_DURATION_SECONDS = 6
DEFAULT_CHUNK_OVERLAP_SECONDS = 3
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

def split_audio_file(file_path: Path, chunk_duration_seconds: int, overlap_seconds: int = DEFAULT_CHUNK_OVERLAP_SECONDS, max_duration: float | None = MAX_PROCESSING_DURATION):
    """
    Split audio file into chunks of specified duration with optional overlap.

    Args:
        file_path: Path to the audio file
        chunk_duration_seconds: Duration of each chunk in seconds
        overlap_seconds: Overlap duration in seconds (0 for no overlap, e.g., VTT)
        max_duration: Maximum duration to process in seconds (None for entire file)

    Returns:
        List of paths to the chunk files
    """
    try:
        # Check if ffmpeg is installed
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Create directory for chunks
        parent_dir = file_path.parent
        chunks_dir = parent_dir / f"{file_path.stem}_chunks"
        chunks_dir.mkdir(exist_ok=True)

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

        # Generate chunks with overlap
        for i, start_time in enumerate(range(0, int(processing_duration), chunk_duration_seconds)):
            # Adjust start time to include overlap from previous chunk (except for first chunk)
            adjusted_start = max(0, start_time - overlap_seconds) if i > 0 and overlap_seconds > 0 else start_time
            intended_end_time = start_time + chunk_duration_seconds
            actual_end_time = min(processing_duration, intended_end_time + overlap_seconds)
            duration = actual_end_time - adjusted_start

            if duration <= 0:
                continue

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

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunk_files.append(output_file)

        overlap_desc = f"with {overlap_seconds}-second{'s' if overlap_seconds != 1 else ''} overlap" if overlap_seconds > 0 else "with no overlap"
        log_success(f"Split audio into {len(chunk_files)} chunks ({chunk_duration_seconds}s each, {overlap_desc}) in {chunks_dir}")
        return chunk_files

    except subprocess.CalledProcessError as e:
        log_error(f"Error splitting audio file with ffmpeg: {e}")
        log_error(f"ffmpeg stderr: {e.stderr.decode()}")
        return []
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

def format_vtt_timestamp(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds_int = int(seconds)
    minutes = seconds_int // 60
    hours = minutes // 60
    sec = seconds_int % 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}:{sec:02d}.{milliseconds:03d}"

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

def generate_output_filename(args, video_title: str | None, file_path: Path, timestamp: str, video_id: str | None = None, is_vtt: bool = False) -> str:
    """Generates the output filename for the transcription."""
    output_extension = ".vtt" if is_vtt else ".txt"
    if args.youtube_url and video_title and video_id:
        return f"{video_title}-{timestamp}{output_extension}"
    else:
        audio_base = file_path.stem
        # Handle cases where stem might already contain 'transcription' or timestamp patterns if re-run
        # A simpler approach is just using the sanitized base name
        sanitized_base = sanitize_filename(audio_base)
        return f"{sanitized_base}_transcription_{timestamp}{output_extension}"

def save_transcription(transcription_text: str, output_path: Path):
    """Saves the transcription text to the specified file path."""
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding='utf-8') as text_file:
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
    parser.add_argument("--vtt", action="store_true", help="Output transcription in VTT format with 6-second chunks.")

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
    """
    Main function to process audio from a YouTube URL or a local file for transcription.

    Parses command-line arguments to determine the input source (YouTube URL or file path)
    and output options.

    Workflow:
    1.  Parses command-line arguments using `parse_args`.
    2.  Initializes the OpenAI client and AITools helper.
    3.  Sets up base output and transcription directories.
    4.  Determines the base filename for output files based on arguments or input source.
    5.  If a YouTube URL is provided:
        - Extracts the video ID.
        - Creates video ID-specific subdirectories within the base output and transcription folders.
        - Downloads the audio from the YouTube URL as an MP3 file into the ID-specific output directory.
        - Updates the base filename to include the video ID.
    6.  If a local file path is provided:
        - Ensures the base transcription directories exist.
        - If the input file is not an MP3, converts it to MP3 using ffmpeg and saves it to the base output directory.
        - If the input file is already an MP3, copies it to the base output directory (if necessary).
    7.  Handles potential errors during download or conversion, exiting if issues occur.
    8.  Cleans any pre-existing audio chunk files from the expected chunks directory.
    9.  Splits the processed MP3 audio file into smaller chunks based on `MAX_CHUNK_DURATION`.
    10. If splitting is successful:
        - Transcribes each audio chunk individually using `AITools`.
        - Saves each chunk's transcription to a text file in the (potentially ID-specific) transcription chunks directory.
        - Combines the individual chunk transcriptions sequentially, potentially using an AI model for better coherence via `combine_chunks_sequentially`.
    11. If splitting fails or produces no chunks, attempts to transcribe the entire audio file at once.
    12. Generates a final timestamped filename for the complete transcription.
    13. Saves the final transcription text to the appropriate (potentially ID-specific) transcription directory.
    14. Logs the final transcription content.

    Exits the script with an error code if critical steps like input validation,
    download, conversion, or transcription fail.
    """
    args = parse_args()

    client = OpenAI()
    ai_tools = AITools(client)

    # Base directories (can be overridden for YouTube)
    output_dir = Path(args.output_dir)
    transcriptions_dir = TRANSCRIPTIONS_DIR
    transcriptions_chunks_dir = TRANSCRIPTIONS_CHUNKS_DIR

    video_id = None
    video_title_for_filename = None

    # Determine output path and base filename
    output_dir.mkdir(parents=True, exist_ok=True) # Create base output dir

    # Construct base filename
    if args.output_filename:
        base_filename_stem = args.output_filename
    elif args.youtube_url:
        base_filename_stem = "youtube_audio" # Keep simple, ID will be added later
    elif args.file_path:
        base_filename_stem = Path(args.file_path).stem
    else:
        base_filename_stem = "output"

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
        if not args.vtt:
            transcriptions_chunks_dir = transcriptions_dir / "transcriptions_chunks" # Now points to transcriptions/VIDEO_ID/transcriptions_chunks

        # Ensure ID-specific directories exist
        output_dir.mkdir(parents=True, exist_ok=True)
        transcriptions_dir.mkdir(parents=True, exist_ok=True)
        if not args.vtt:
            transcriptions_chunks_dir.mkdir(parents=True, exist_ok=True)

        youtube: YouTube = YtdlpYouTube()
        final_filename = f"{base_filename_stem}-{video_id}.mp3"
        file_path = output_dir / final_filename

        try:
            video_info = youtube.get_video_info(args.youtube_url)
            video_title_for_filename = video_info.get('title', f"{base_filename_stem}-{video_id}") if video_info else f"{base_filename_stem}-{video_id}"
        except Exception as e:
            log_warning(f"Could not fetch YouTube video title: {e}. Using base filename.")
            video_title_for_filename = f"{base_filename_stem}-{video_id}"

        if not youtube.download_audio(args.youtube_url, file_path):
            log_error("Failed to download YouTube video. Exiting.")
            sys.exit(1)

        log_success(f"Successfully downloaded and saved YouTube video as {file_path}")

    elif args.file_path:
        log_info(f"Processing local file: {args.file_path}")
        input_file = Path(args.file_path)
        if not input_file.exists():
            log_error(f"Input file not found: {input_file}")
            sys.exit(1)

        # Ensure base directories exist if not YouTube
        transcriptions_dir.mkdir(parents=True, exist_ok=True)
        if not args.vtt:
            transcriptions_chunks_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / f"{base_filename_stem}.mp3"
        video_title_for_filename = base_filename_stem

        if file_path.exists():
            log_info(f"Removing existing file: {file_path}")
            try:
                file_path.unlink()
                log_info(f"Removed existing file: {file_path}")
            except OSError as e:
                log_error(f"Error removing existing file {file_path}: {e}")
                sys.exit(1)

        if input_file.suffix.lower() != ".mp3":
            log_info(f"Converting {input_file} to MP3 format at {file_path}...")
            try:
                subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                cmd = [
                    'ffmpeg',
                    '-i', str(input_file),
                    '-vn',
                    '-acodec', 'libmp3lame',
                    '-ab', '192k',
                    '-ar', '44100',
                    '-y',
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
        elif input_file.resolve() != file_path.resolve():
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

    if args.vtt:
        chunk_duration_for_split = VTT_CHUNK_DURATION_SECONDS
        overlap_for_split = DEFAULT_CHUNK_OVERLAP_SECONDS
        log_info(f"Splitting audio for VTT: {VTT_CHUNK_DURATION_SECONDS}s chunks, no overlap...")
    else:
        chunk_duration_for_split = int(MAX_CHUNK_DURATION * 60)
        overlap_for_split = DEFAULT_CHUNK_OVERLAP_SECONDS
        log_info(f"Splitting audio for text: {MAX_CHUNK_DURATION}-minute chunks, {overlap_for_split}s overlap...")

    audio_chunks = split_audio_file(
        file_path,
        chunk_duration_seconds=chunk_duration_for_split,
        overlap_seconds=overlap_for_split
    )

    final_transcription = ""

    if not audio_chunks:
        if args.vtt:
            log_warning("No audio chunks were created. Cannot generate VTT.")
            final_transcription = "WEBVTT\n\nERROR: Could not split audio into chunks."
        else:
            log_warning("No audio chunks were created. Trying to transcribe the entire file...")
            final_transcription = ai_tools.process_audio_transcription(file_path)
    else:
        if args.vtt:
            # NOTE: Applying sequential text combination logic instead of VTT formatting
            # as requested. The output file will have a .vtt extension but contain
            # combined plain text, not standard VTT cues.
            log_warning("Processing with --vtt flag, but applying sequential text combination logic instead of VTT formatting.")
            log_info(f"Transcribing {len(audio_chunks)} audio chunks for sequential combination...")
            chunk_transcriptions: list[str] = []

            # Define path for intermediate chunk transcriptions if needed for debugging/inspection
            # This path needs to be consistent with how transcriptions_chunks_dir is defined earlier
            # It might be inside a video_id subdir if applicable.
            intermediate_chunks_dir = expected_chunks_dir.parent / f"{expected_chunks_dir.stem}_transcriptions"
            intermediate_chunks_dir.mkdir(parents=True, exist_ok=True)
            log_info(f"Saving intermediate chunk transcriptions to: {intermediate_chunks_dir}")


            for i, chunk_file in enumerate(audio_chunks):
                log_info(f"Transcribing chunk {i+1}/{len(audio_chunks)}...")
                chunk_transcription = ai_tools.process_audio_transcription(chunk_file)
                chunk_transcriptions.append(chunk_transcription)

                # Save intermediate chunk transcriptions
                chunk_transcription_filename = f"chunk_{i:03d}_transcription.txt"
                chunk_transcription_path = intermediate_chunks_dir / chunk_transcription_filename
                try:
                    with open(chunk_transcription_path, "w", encoding='utf-8') as chunk_text_file:
                        chunk_text_file.write(chunk_transcription)
                        # Log less verbosely inside the loop
                        # log_info(f"Saved chunk transcription to {chunk_transcription_path}")
                except Exception as e:
                    log_error(f"Error saving chunk transcription {chunk_transcription_path}: {e}")
            log_info(f"Finished transcribing {len(audio_chunks)} chunks.")

            log_info("Combining chunks sequentially (as requested for --vtt)...")
            # Ensure combine_chunks_sequentially uses the correct BOUNDARY_WORD_COUNT
            # which depends on the overlap used during splitting (overlap_for_split)
            # The current combine_chunks_sequentially uses a fixed BOUNDARY_WORD_COUNT=50
            # This might need adjustment if overlap_for_split changes significantly.
            final_transcription = combine_chunks_sequentially(chunk_transcriptions, ai_tools)
            log_success("Finished combining text chunks.")
            # Clean up intermediate transcription chunks dir
            try:
                shutil.rmtree(intermediate_chunks_dir)
                log_info(f"Removed intermediate transcription chunks directory: {intermediate_chunks_dir}")
            except Exception as e:
                log_error(f"Error removing intermediate transcription chunks directory {intermediate_chunks_dir}: {e}")
        else:
            log_info(f"Transcribing {len(audio_chunks)} audio chunks for plain text...")
            chunk_transcriptions: list[str] = []
            for i, chunk_file in enumerate(audio_chunks):
                log_info(f"Transcribing chunk {i+1}/{len(audio_chunks)}...")
                chunk_transcription = ai_tools.process_audio_transcription(chunk_file)
                chunk_transcriptions.append(chunk_transcription)

                chunk_transcription_filename = f"chunk_{i:03d}_transcription.txt"
                chunk_transcription_path = transcriptions_chunks_dir / chunk_transcription_filename
                try:
                    with open(chunk_transcription_path, "w", encoding='utf-8') as chunk_text_file:
                        chunk_text_file.write(chunk_transcription)
                    log_info(f"Saved chunk transcription to {chunk_transcription_path}")
                except Exception as e:
                    log_error(f"Error saving chunk transcription {chunk_transcription_path}: {e}")

            log_info("Combining chunks sequentially for plain text...")
            final_transcription = combine_chunks_sequentially(chunk_transcriptions, ai_tools)
            log_success("Finished combining text chunks.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = generate_output_filename(args, video_title_for_filename, file_path, timestamp, video_id, is_vtt=args.vtt)
    output_path = transcriptions_dir / output_filename

    save_transcription(final_transcription, output_path)

    log_info(f"--- Final Transcription ({'.vtt' if args.vtt else '.txt'}) ---")
    if args.vtt:
        log_info(final_transcription[:200] + ("..." if len(final_transcription) > 200 else ""))
    else:
        log_info(final_transcription)
    log_info("--- End Transcription ---")

    clean_chunks_directory(expected_chunks_dir)


if __name__ == "__main__":
    main()
