import json

def load_transcript_json(path):
    """Load subject utterances from annotated JSON transcript."""
    with open(path, "r") as f:
        data = json.load(f)

    # 只取 Subject 的话
    subject_turns = [
        turn["text"].strip()
        for turn in data.get("dialogue_turns", [])
        if turn.get("speaker", "").lower() == "subject"
    ]
    transcript = "\n".join(subject_turns)

    return transcript, data



def chunk_by_length(text, n=200):
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(0, len(words), n)]
