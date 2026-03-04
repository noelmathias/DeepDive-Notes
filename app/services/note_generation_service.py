from app.services.llms_services import generate_structured_note


import time

def generate_notes(segments):
    notes = []
    failed_segments = 0

    for i, segment in enumerate(segments):
        print(f"Processing segment {i+1}/{len(segments)}...")

        try:
            start_time = time.time()
            segment_text = segment["text"][:1200]
            structured = generate_structured_note(segment_text)

            structured["start"] = segment["start"]
            structured["end"] = segment["end"]

            notes.append(structured)

            print(f"✓ Done in {round(time.time() - start_time, 2)} sec")

        except Exception as e:
            failed_segments += 1
            print(f"⚠ Skipping segment {i+1} due to error:", e)
            continue

    print(f"\nFinished processing.")
    print(f"Successful segments: {len(notes)}")
    print(f"Failed segments: {failed_segments}")

    if len(notes) == 0:
        raise RuntimeError("All segments failed. Check LLM stability.")

    return notes