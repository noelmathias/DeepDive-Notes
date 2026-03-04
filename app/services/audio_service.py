import os
from yt_dlp import YoutubeDL


def download_audio(youtube_url: str, output_dir: str = "data/raw_audio") -> str:
    """
    Downloads audio from a YouTube video and converts it to WAV format.

    Args:
        youtube_url (str): The full YouTube video URL.
        output_dir (str): Directory where audio will be saved.

    Returns:
        str: Path to the saved WAV file.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info["id"]

    final_path = os.path.join(output_dir, f"{video_id}.wav")

    return final_path