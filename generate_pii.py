import os
import sys
import json
from openai import OpenAI
from utils.io_helper import save_output

client = OpenAI()


def load_memoir_text(subject_name):
    """Load previously generated Few-shot memoir file."""
    path = os.path.join("outputs", subject_name, f"{subject_name}_fewshot.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Few-shot memoir not found → {path}\n"
            f"Please run generate_fewshot.py first."
        )
    with open(path, "r") as f:
        lines = f.readlines()
    return "".join(lines[2:])  # Skip first 2 lines (header)


def load_annotations(json_path):
    """Load annotated JSON to get entity info for PII rewriting."""
    with open(json_path, "r") as f:
        return json.load(f)


def extract_sensitive_entities(annotation_json):
    """
    Collect entity strings from sentence_annotations.
    Example: ["Italy", "America", "health benefits"]
    """
    entities = set()

    for turn in annotation_json.get("dialogue_turns", []):
        anns = turn.get("sentence_annotations", [])
        for ann in anns:
            for ent in ann.get("entities", []):
                entities.add(ent)

    return list(entities)


def rewrite_pii(memoir_text, entities):
    """
    Use GPT to rewrite memoir text, generalizing named entities.
    """
    prompt = f"""
    You are a privacy-protection rewriting assistant.

    The following text is a personal memoir. It contains real names,
    places, organizations, and possibly specific dates.

    Your task:
    - Replace each real-world entity with a generalized form
    - Do NOT remove emotional meaning
    - Keep narrative coherence and chronological order
    - Maintain first-person reflective style
    - If the original text is vague, remain vague
    - NEVER add invented facts

    Typical transformations:
      "Guangzhou" → "a city where I once lived"
      "John" → "a close friend of mine"
      "1982" → "many years ago"

    === Sensitive Entities Detected ===
    {entities}

    === Text to Rewrite ===
    {memoir_text}
    """

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_pii.py <json_file>")
        sys.exit(1)

    json_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # Load source texts and annotations
    memoir_text = load_memoir_text(base_name)
    annotations = load_annotations(json_path)
    entities = extract_sensitive_entities(annotations)

    # Rewrite with PII safety enforced
    pii_safe_text = rewrite_pii(memoir_text, entities)

    # Save result
    save_output("pii", base_name, pii_safe_text)


if __name__ == "__main__":
    main()
