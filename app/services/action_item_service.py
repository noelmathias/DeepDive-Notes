def extract_action_items(segments: list) -> list:
    """
    Extracts simple action items based on keyword detection.
    Phase 1: rule-based approach.
    """

    action_keywords = [
        "should",
        "need to",
        "must",
        "we will",
        "action",
        "task",
        "assign",
        "implement",
    ]

    action_items = []

    for segment in segments:
        text = segment["text"].lower()

        for keyword in action_keywords:
            if keyword in text:
                action_items.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "context": segment["text"]
                })
                break

    return action_items