import json
from collections import Counter

def load_transcript_json(path):
    """
    Load transcript for the Subject by:
    - auto-detecting which speaker is the Subject (if needed)
    - extracting text from `sentences[i]["text"]` (correct field)
    """
    with open(path, "r") as f:
        data = json.load(f)

    turns = data.get("dialogue_turns", [])

    # ---- Step 1: detect who is the subject ----
    # speaker names present (excluding interviewer)
    speaker_counts = Counter(
        t.get("speaker", "").strip()
        for t in turns
        if t.get("speaker") and "interviewer" not in t["speaker"].lower()
    )

    # Most common non-interviewer = subject
    if speaker_counts:
        subject_name = speaker_counts.most_common(1)[0][0]
    else:
        subject_name = "Subject"   # fallback

    # ---- Step 2: collect all sentences belonging to subject ----
    subject_texts = []

    for turn in turns:
        if turn.get("speaker", "").strip() != subject_name:
            continue

        # NEW: use "sentences" array instead of turn["text"]
        if "sentences" in turn:
            for seg in turn["sentences"]:
                text = seg.get("text", "").strip()
                if text:
                    subject_texts.append(text)
        else:
            # fallback if text exists
            text = turn.get("text", "").strip()
            if text:
                subject_texts.append(text)

    transcript = "\n".join(subject_texts)

    return transcript, data
