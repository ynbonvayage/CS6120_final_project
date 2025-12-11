import os
import json
from pathlib import Path

# ====== IMPORT FUNCTIONS FROM ROOT DIRECTORY ======
from generate_baseline import generate_baseline
from generate_rag import run_rag
from generate_fewshot import run_fewshot
from generate_pii import rewrite_pii

from utils.segmentation import load_transcript_json


DATA_DIR = "data"
OUTPUT_DIR = "outputs"


def ensure_dir(path):
    """Create directory if not exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def process_file(json_path):
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    out_dir = f"outputs/{base_name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Processing {base_name} ===")

    # segmentation returns BOTH transcript and json_data
    transcript, raw_data = load_transcript_json(json_path)

    # baseline
    print(" -> Running baseline...")
    baseline_text = generate_baseline(transcript)
    with open(f"{out_dir}/baseline.txt", "w") as f:
        f.write(baseline_text)

    # RAG
    print(" -> Running RAG...")
    rag_text = run_rag(raw_data)   
    with open(f"{out_dir}/rag.txt", "w") as f:
        f.write(rag_text)

    # Few-shot
    print(" -> Running few-shot...")
    fewshot_text = run_fewshot(raw_data)  
    with open(f"{out_dir}/fewshot.txt", "w") as f:
        f.write(fewshot_text)

    # PII-safe
    print(" -> Running PII-safe...")
    pii_text = rewrite_pii(fewshot_text, raw_data)
    with open(f"{out_dir}/pii.txt", "w") as f:
        f.write(pii_text)

    print(f" -> Completed {base_name}\n")


def main():
    ensure_dir(OUTPUT_DIR)

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    print(f"Found {len(files)} JSON files.")

    for fname in files:
        process_file(f"{DATA_DIR}/{fname}")

    print("=== All files processed successfully ===")


if __name__ == "__main__":
    main()
