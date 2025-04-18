"""
Provides classes for interacting with YouTube, specifically for downloading audio content.

This module defines an abstract base class `YouTube` outlining the interface
for YouTube operations and a concrete implementation `YtdlpYouTube` that uses
the `yt-dlp` command-line tool to download audio from YouTube URLs.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import json
from lib.logger import log_error, log_cyan, log_bold, log_success, log_info

class YouTube(ABC):
    """
    Interface for YouTube operations.
    """

    @abstractmethod
    def download_audio(self, url: str, output_path: Path) -> bool:
        """
        Downloads content from a YouTube URL to the specified output path.

        Args:
            url: YouTube URL string
            output_path: Path object representing the destination location
        """
        pass

    @abstractmethod
    def get_video_info(self, url: str) -> dict:
        """
        Retrieves video information from a YouTube URL.

        Args:
            url: YouTube URL string

        Returns:
            A dictionary containing video information
        """
        pass

class YtdlpYouTube(YouTube):
    """
    Implementation of YouTube operations using yt-dlp.
    """

    def download_audio(self, url: str, output_path: Path) -> bool:
        try:
            # Check if yt-dlp is installed
            try:
                subprocess.run(['yt-dlp', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                log_error(f"{log_bold(log_cyan('yt-dlp'))} not found. Please install it with: {log_cyan('pip install yt-dlp')}")
                return False

            # Set up the command to extract audio only and save to the specified path
            cmd = [
                'yt-dlp',
                '-f', 'bestaudio',  # Get best audio quality
                '-x',               # Extract audio
                '--audio-format', 'mp3',  # Convert to mp3
                '-o', str(output_path),
                url
            ]

            log_info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)

            if result.returncode != 0:
                log_error(f"yt-dlp exited with code {result.returncode}")
                return False

            log_success("Successfully downloaded audio using yt-dlp")
            return True

        except Exception as e:
            log_error(f"Error using yt-dlp: {e}")
            return False

    def get_video_info(self, url: str) -> dict:
        """
        Retrieves video information from a YouTube URL using yt-dlp.

        Args:
            url: YouTube URL string

        Returns:
            A dictionary containing video information
        """
        try:
            # Check if yt-dlp is installed
            try:
                subprocess.run(['yt-dlp', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                log_error(f"{log_bold(log_cyan('yt-dlp'))} not found. Please install it with: {log_cyan('pip install yt-dlp')}")
                return {}

            # Set up the command to get video info
            cmd = [
                'yt-dlp',
                '--dump-json',  # Output video info in JSON format
                url
            ]

            log_info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode != 0:
                log_error(f"yt-dlp exited with code {result.returncode}")
                return {}

            # Parse the JSON output
            video_info = result.stdout.decode('utf-8')
            return json.loads(video_info)

        except Exception as e:
            log_error(f"Error using yt-dlp: {e}")
            return {}
