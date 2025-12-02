import os
import json
from pathlib import Path

import pandas as pd
from joblib import load


# ==========================
# 1. 路径配置
# ==========================

# 已训练好的模型与向量器（来自 Experiment B）
MODEL_DIR = Path("Experiment_B/outputs/iemocap_emotion_experiment")
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "svm_model.joblib"   # 这里选用 SVM

# 你们收集的 50 份对话 JSON 存放位置
OUR_dialogue_turnsS_DIR = Path("/Users/Jessie/Downloads/CS6120/Project/CS6120_final_project/data_json")

# 输出预测结
OUTPUT_CSV = Path("Experiment_B/outputs/our_dialogue_turnss_emotion_predictions.csv")


# ==========================
# 2. 读取我们自己的对话数据
# ==========================

def load_our_dialogue_turnss() -> pd.DataFrame:
    """
    从 OUR_dialogue_turnsS_DIR 读取所有 *.json 文件。
    假定每个文件结构类似：
    {
        "dialogue_turns": [
            {
                "speaker": "Participant",
                "text": "...",
                "emotions": ["Nostalgia", "Joy"]
            },
            ...
        ]
    }

    返回一个 DataFrame，每行是一条 utterance：
    dialogue_turns_id, turn_index, speaker, text, human_emotions
    """
    rows = []

    json_files = sorted(OUR_dialogue_turnsS_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {OUR_dialogue_turnsS_DIR}")

    for path in json_files:
        dialogue_turns_id = path.stem  # 比如 01_Elena_Warm
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        turns = data.get("dialogue_turns", [])
        for idx, turn in enumerate(turns):
            text = (turn.get("text") or "").strip()
            if not text:
                continue

            speaker = turn.get("speaker", "").strip()
            emotions = turn.get("emotions") or []
            # emotions 是一个列表，比如 ["Nostalgia", "Joy"]
            human_emotions = "|".join(emotions)

            rows.append(
                {
                    "dialogue_turns_id": dialogue_turns_id,
                    "turn_index": idx,
                    "speaker": speaker,
                    "text": text,
                    "human_emotions": human_emotions,
                }
            )

    df = pd.DataFrame(rows)
    return df


# ==========================
# 3. 主流程：加载模型 → 预测 → 保存
# ==========================

def main():
    print("Loading our collected dialogue_turnss...")
    df = load_our_dialogue_turnss()
    print(f"Loaded {len(df)} utterances from {OUR_dialogue_turnsS_DIR}")
    print(df.head(5))

    print("\nLoading vectorizer and emotion classifier from Experiment B...")
    vectorizer = load(VECTORIZER_PATH)
    clf = load(MODEL_PATH)

    print("\nVectorizing texts and running prediction...")
    X = vectorizer.transform(df["text"].tolist())
    y_pred = clf.predict(X)

    df["predicted_emotion"] = y_pred

    # 保存结果
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved prediction results to: {OUTPUT_CSV}")

    # 简单看一下预测分布
    print("\nPredicted emotion distribution:")
    print(df["predicted_emotion"].value_counts())


if __name__ == "__main__":
    main()
