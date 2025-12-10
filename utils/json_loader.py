import json

def load_transcript_json(path):
    """Load annotated JSON transcript file."""
    with open(path, "r") as f:
        data = json.load(f)

    return data


def to_full_transcript(data):
    return "\n".join(
        turn["text"].strip()
        for turn in data.get("dialogue_turns", [])
        if turn.get("speaker", "").lower() == "subject"
    )
