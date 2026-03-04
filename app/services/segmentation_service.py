def segment_transcript(transcript: list, segment_size: int = 8) -> list:
    """
    Groups transcript into simple fixed-size segments.

    Args:
        transcript (list): List of transcript segments.
        segment_size (int): Number of transcript entries per segment.

    Returns:
        list: List of grouped segments.
    """

    segments = []

    for i in range(0, len(transcript), segment_size):
        chunk = transcript[i:i + segment_size]

        segment = {
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": " ".join([s["text"] for s in chunk])
        }

        segments.append(segment)

    return segments