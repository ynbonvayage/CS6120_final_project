import os
import csv
from pathlib import Path

IEMOCAP_ROOT = "/Users/Jessie/Downloads/CS6120/Project/CS6120_final_project/Experiment_B/IEMOCAP"

OUTPUT_CSV = "Experiment_B/data/iemocap_utterances_text.csv"

EMO_MAP = {
    "ang": "anger",
    "hap": "happiness",
    "sad": "sadness",
    "neu": "neutral",
    "fru": "frustration",
    "exc": "happiness",     
    "fear": "fear",
    "disg": "disgust",
    "xxx": None,            
    "oth": None             
}

def read_transcript(trans_path: str) -> dict:
    utter_dict = {}

    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("["):
                continue

            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue

            utt_id = parts[0]

            rest = parts[1].strip()
            if "]:" in rest:
                text = rest.split("]:", 1)[1].strip()
            else:
                text = rest

            utter_dict[utt_id] = text

    return utter_dict


def build_iemocap_text_csv():

    rows = []

    for sess_id in range(1, 6):
        session_name = f"Session{sess_id}"

        emo_dir = os.path.join(IEMOCAP_ROOT, session_name, "dialog", "EmoEvaluation")
        trans_dir = os.path.join(IEMOCAP_ROOT, session_name, "dialog", "transcriptions")

        print(f"Processing {session_name} ...")

        for filename in os.listdir(emo_dir):
            if not filename.endswith(".txt"):
                continue

            emo_file = os.path.join(emo_dir, filename)
            dialog_id = filename.replace(".txt", "")

            trans_file = os.path.join(trans_dir, filename)
            if not os.path.exists(trans_file):
                print(f"Warning: transcript not found for {emo_file}")
                continue

            trans_dict = read_transcript(trans_file)

            with open(emo_file, "r", encoding="latin-1") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("["):
                        continue

                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue

                    utt_id = parts[1].strip()
                    emo_raw = parts[2].strip()

                    emo = EMO_MAP.get(emo_raw)
                    if emo is None:
                        continue

                    text = trans_dict.get(utt_id, "")
                    if text.strip() == "":
                        continue

                    rows.append({
                        "session": session_name,
                        "dialog_id": dialog_id,
                        "utterance_id": utt_id,
                        "text": text,
                        "emotion": emo
                    })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["session", "dialog_id", "utterance_id", "text", "emotion"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Saved to: {OUTPUT_CSV}")
    print(f"Total utterances: {len(rows)}")

if __name__ == "__main__":
    build_iemocap_text_csv()
