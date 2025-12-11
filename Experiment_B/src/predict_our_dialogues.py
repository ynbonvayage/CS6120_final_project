import os
import json
from pathlib import Path
from joblib import load

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "outputs" / "iemocap_emotion_experiment"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "svm_model.joblib"

DATA_DIR = BASE_DIR.parent / "data_json"

OUTPUT_DIR = BASE_DIR / "outputs" / "json_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading vectorizer & classifier...")
vectorizer = load(VECTORIZER_PATH)
clf = load(MODEL_PATH)

def predict_emotion(text: str) -> str:
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]


def process_one_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dialogue_turns = data.get("dialogue_turns", [])

    for turn in dialogue_turns:
        # 只处理 Subject 的 sentences
        if "sentences" in turn:
            for sent in turn["sentences"]:
                text = sent["text"].strip()
                if not text:
                    continue

                pred = predict_emotion(text)
                sent["predicted_emotions"] = [pred]

    # 保存预测后的 JSON
    out_path = OUTPUT_DIR / path.name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved → {out_path}")


def main():
    json_files = sorted(DATA_DIR.glob("*.json"))
    print(f"Found {len(json_files)} dialogues in {DATA_DIR}")

    for file in json_files:
        print(f"\nProcessing {file.name} ...")
        process_one_file(file)

    print("\nAll predictions saved.")


if __name__ == "__main__":
    main()
