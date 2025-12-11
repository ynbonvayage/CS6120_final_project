import os
import csv
from bert_score import score
from utils.segmentation import load_transcript_json

# --------------------------
# Paths (modify if needed)
# --------------------------
REFERENCE_DIR = "data"
OUTPUT_DIR = "outputs"
OUTPUT_CSV = "bertscore_results.csv"


# --------------------------
# Load reference transcript using segmentation logic
# --------------------------
def load_reference_transcript(path):
    """
    Uses the same segmentation logic as the generation pipeline.
    Returns subject-only transcript string.
    """
    transcript, _ = load_transcript_json(path)
    return transcript


# --------------------------
# Compute BERTScore (returns P, R, F1)
# --------------------------
def compute_bertscore(reference, candidate):
    """
    Compute BERTScore using RoBERTa-large (default).
    Returns precision, recall, F1 float numbers.
    """
    P, R, F1 = score(
        [candidate],     # candidate list
        [reference],     # reference list
        lang="en",       # English
        verbose=False
    )
    return float(P[0]), float(R[0]), float(F1[0])


# --------------------------
# Main evaluation function
# --------------------------
def evaluate_all():

    print("\n=== Running BERTScore Evaluation ===")

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["session_id", "version", "precision", "recall", "f1"])

        # iterate through output folders
        for session in sorted(os.listdir(OUTPUT_DIR)):
            session_path = os.path.join(OUTPUT_DIR, session)

            if not os.path.isdir(session_path) or session.startswith("."):
                continue

            print(f"\nEvaluating: {session}")

            # reference JSON
            ref_json = os.path.join(REFERENCE_DIR, session + ".json")
            if not os.path.exists(ref_json):
                print(f"  ! No reference JSON found: {session}")
                continue

            reference = load_reference_transcript(ref_json)
            if not reference.strip():
                print(f"  ! Empty reference transcript: {session}")
                continue

            # evaluate all four versions
            for version in ["baseline", "rag", "fewshot", "pii"]:
                gen_path = os.path.join(session_path, f"{version}.txt")

                if not os.path.exists(gen_path):
                    print(f"  - {version}: missing file, skipped")
                    continue

                with open(gen_path, "r") as f:
                    candidate = f.read().strip()

                if not candidate:
                    print(f"  - {version}: empty candidate, skipped")
                    continue

                # compute BERTScore
                P, R, F1 = compute_bertscore(reference, candidate)
                writer.writerow([session, version, P, R, F1])

                print(f"  - {version}: F1 = {F1:.4f}")

    print(f"\n=== Done! Results saved to {OUTPUT_CSV} ===\n")


if __name__ == "__main__":
    evaluate_all()
