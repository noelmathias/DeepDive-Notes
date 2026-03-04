from faster_whisper import WhisperModel


def transcribe_audio(audio_path: str, model_size: str = "base") -> list:
    """
    Transcribes audio using faster-whisper.

    Args:
        audio_path (str): Path to the WAV file.
        model_size (str): Whisper model size (tiny, base, small, medium, large)

    Returns:
        list: List of transcript segments with text and timestamps.
    """

    model = WhisperModel(model_size, compute_type="int8")

    segments, _ = model.transcribe(audio_path)

    transcript = []

    for segment in segments:
        transcript.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    return transcript