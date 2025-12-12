import os
import json
import csv
from openai import OpenAI

client = OpenAI()

# ===== PATHS =====
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
OUTPUT_CSV = "human_like_scores.csv"

VERSIONS = ["baseline", "rag", "fewshot", "pii"]


# ------------------------------------------------------
#  Clean model outputs to make them valid JSON
# ------------------------------------------------------
def clean_json(raw_text):
    """
    Removes ```json and ``` markers so the string can be parsed by json.loads().
    """
    cleaned = (
        raw_text.replace("```json", "")
                .replace("```", "")
                .strip()
    )
    return cleaned


# ------------------------------------------------------
# Load transcript as the reference text
# ------------------------------------------------------
def load_reference_transcript(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    subject_lines = []
    for turn in data.get("dialogue_turns", []):
        if turn.get("speaker", "").lower() == "subject":

            # Some JSON uses "sentences" list instead of single "text"
            if "text" in turn:
                subject_lines.append(turn["text"].strip())
            elif "sentences" in turn:
                for s in turn["sentences"]:
                    subject_lines.append(s["text"].strip())

    return "\n".join(subject_lines)


# ------------------------------------------------------
#  Call GPT to get human-like evaluation scores
# ------------------------------------------------------
def get_ratings(reference, generated_text):
    prompt = f"""
You are an expert human evaluator of personal memoir narratives.

Rate the generated memoir using these 4 criteria (1–5 scale):
1. COVERAGE — How completely the memoir reflects important events in the transcript.
2. FAITHFULNESS — No hallucinations; accuracy w.r.t the transcript.
3. CHRONOLOGY — Are events in correct temporal order?
4. COHERENCE — Narrative flow, smooth transitions, readability.

Return *ONLY* valid JSON:
{{
  "coverage": <1-5>,
  "faithfulness": <1-5>,
  "chronology": <1-5>,
  "coherence": <1-5>
}}

=== ORIGINAL TRANSCRIPT ===
{reference}

=== GENERATED MEMOIR ===
{generated_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()
    cleaned = clean_json(raw)

    # try parsing
    try:
        return json.loads(cleaned)
    except Exception:
        print("⚠️ JSON parse failed. Raw output:")
        print("```")
        print(raw)
        print("```")
        print("\nCleaned output:")
        print(cleaned)
        raise


# ------------------------------------------------------
#  Evaluate all sessions
# ------------------------------------------------------
def evaluate_all():
    print("Starting human-like evaluation...\n")

    rows = []
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

    for jf in json_files:
        session_id = jf.replace(".json", "")
        print(f"\n=== Evaluating session: {session_id} ===")

        reference = load_reference_transcript(os.path.join(DATA_DIR, jf))

        out_folder = os.path.join(OUTPUT_DIR, session_id)
        if not os.path.exists(out_folder):
            print(f"  ! No outputs found for {session_id}, skipping.")
            continue

        for version in VERSIONS:
            gen_path = os.path.join(out_folder, f"{version}.txt")

            if not os.path.exists(gen_path):
                print(f"  ! Missing {version}.txt, skipping")
                continue

            print(f" -> Rating {version}...")

            with open(gen_path, "r") as f:
                generated = f.read().strip()

            scores = get_ratings(reference, generated)

            rows.append([
                session_id,
                version,
                scores["coverage"],
                scores["faithfulness"],
                scores["chronology"],
                scores["coherence"],
            ])

    # Write to CSV
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["session_id", "version", "coverage", "faithfulness", "chronology", "coherence"])
        writer.writerows(rows)

    print(f"\nAll done! Scores saved to: {OUTPUT_CSV}")


# ------------------------------------------------------
if __name__ == "__main__":
    evaluate_all()
