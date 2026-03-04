from app.services.audio_service import download_audio
from app.services.transcription_service import transcribe_audio
from app.services.segmentation_service import segment_transcript
from app.services.embedding_service import create_embedding
from app.services.vector_store_service import add_embeddings


def process_video(url, session_id):
    """
    Full pipeline:
    1. Download audio
    2. Transcribe
    3. Segment
    4. Embed
    5. Store in FAISS (session-isolated)
    """

    # Step 1: Download audio
    audio_path = download_audio(url)

    # Step 2: Transcribe
    transcript = transcribe_audio(audio_path)

    # Step 3: Segment transcript
    segments = segment_transcript(transcript)

    if not segments:
        return False

    embeddings = []
    metadata_entries = []

    video_id = url.split("v=")[-1].split("&")[0]

    for seg in segments:
        embedding = create_embedding(seg["text"])
        embeddings.append(embedding)

        metadata_entries.append({
            "video_id": video_id,
            "title": f"Video {video_id}",
            "summary": seg["text"][:300],
            "key_points":[],
            "start": seg["start"],
            "end":seg["end"]
        })

    # Store inside session-isolated FAISS
    add_embeddings(
        embeddings=embeddings,
        metadata_entries=metadata_entries,
        session_id=session_id
    )

    return True