import os
import json
import pandas as pd
from bert_score import score

# ----------------------------------------------------
# Utility: Load transcript for reference
# ----------------------------------------------------
def load_reference_transcript(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    subject_texts = [
        turn["text"].strip()
        for turn in data.get("dialogue_turns", [])
        if turn.get("speaker", "").lower() == "subject"
    ]
    return "\n".join(subject_texts)


# ----------------------------------------------------
# Utility: Load generated memoir from outputs
# ----------------------------------------------------
def load_generated_text(subject, model_name):
    path = os.path.join("outputs", subject, f"{subject}_{model_name}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        lines = f.readlines()
    return "".join(lines[2:])  # Skip header in first 2 lines


# ----------------------------------------------------
# Main Evaluation Logic
# ----------------------------------------------------
def evaluate_all(data_folder="data"):
    results = []

    model_types = ["baseline", "rag", "fewshot", "pii"]

    # Loop through each subject JSON file
    for filename in os.listdir(data_folder):
        if not filename.endswith(".json"):
            continue

        subject_name = os.path.splitext(filename)[0]
        json_file = os.path.join(data_folder, filename)

        print(f"\n📌 Evaluating: {subject_name}")

        reference = load_reference_transcript(json_file)

        for m in model_types:
            generated = load_generated_text(subject_name, m)
            if generated is None:
                print(f"⚠ Missing: {subject_name}_{m}.txt")
                continue

            # BERTScore
            P, R, F1 = score([generated], [reference], lang="en", verbose=False)
            results.append({
                "subject": subject_name,
                "model": m,
                "precision": float(P[0]),
                "recall": float(R[0]),
                "f1": float(F1[0]),
            })

            print(f"  {m:<8}: BERTScore-F1 = {float(F1[0]):.4f}")

    # Convert to CSV
    df = pd.DataFrame(results)
    df.to_csv("bertscore_results.csv", index=False)
    print("\n💾 Results saved → bertscore_results.csv")

    # Trend summary
    print("\n📈 Average Performance:")
    print(df.groupby("model")["f1"].mean().sort_values(ascending=False))


if __name__ == "__main__":
    evaluate_all()
