import os
import sys
import json
from openai import OpenAI
from utils.io_helper import save_output

# Create client (needs OPENAI_API_KEY env variable)
client = OpenAI()


def load_transcript_json(path):
    """Load annotated JSON transcript file."""
    with open(path, "r") as f:
        data = json.load(f)

    # Combine subject speech as narrative input
    subject_texts = [
        turn["text"].strip()
        for turn in data.get("dialogue_turns", [])
        if turn.get("speaker", "").lower() == "subject"
    ]

    transcript = "\n".join(subject_texts)
    return transcript


def generate_baseline(transcript):
    """Zero-shot memoir generation using full transcript as context."""
    prompt = f"""
    You are a supportive memoir-writing assistant.
    Write a first-person life memoir based ONLY on the content below.
    Use:
    - Past tense
    - Chronological order
    - Reflective, warm storytelling tone
    - No hallucinations or invented details

    === Transcript ===
    {transcript}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_baseline.py <json_file>")
        sys.exit(1)

    json_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    transcript = load_transcript_json(json_path)
    memoir_output = generate_baseline(transcript)

    save_output("baseline", base_name, memoir_output)


if __name__ == "__main__":
    main()
