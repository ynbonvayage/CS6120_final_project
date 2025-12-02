import os
import csv
from pathlib import Path

# 修改成你的 IEMOCAP 数据根目录
IEMOCAP_ROOT = "/Users/Jessie/Downloads/CS6120/Project/CS6120_final_project/Experiment_B/IEMOCAP"

# 输出 CSV 名称
OUTPUT_CSV = "Experiment_B/data/iemocap_utterances_text.csv"

# 我们保留的情绪标签映射（可以根据需要扩展）
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
    """
    输入为 transcription 文件路径，如:
    IEMOCAP_full_release/Session1/dialog/transcriptions/Ses01F_impro01.txt

    输出:
    {
      "Ses01F_impro01_F000": "文本内容",
      "Ses01F_impro01_M001": "文本内容",
      ...
    }
    """
    utter_dict = {}

    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # transcript 格式示例:
            # Ses01F_impro01_F000 [1.234 - 4.234]: I can't believe this happened ...
            if not line or line.startswith("["):
                continue

            # 取 utterance_id（空格前）
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue

            utt_id = parts[0]

            # 剩余部分去掉时间戳，保留文本
            rest = parts[1].strip()
            # rest 如: [1.234 - 4.234]: text...
            if "]:" in rest:
                text = rest.split("]:", 1)[1].strip()
            else:
                text = rest

            utter_dict[utt_id] = text

    return utter_dict


def build_iemocap_text_csv():
    """
    生成一个统一的 CSV, 包含:
    session, dialog_id, utterance_id, text, emotion
    """

    rows = []

    # 遍历五个 Session
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

            # 对应的 transcript 文件
            trans_file = os.path.join(trans_dir, filename)
            if not os.path.exists(trans_file):
                print(f"Warning: transcript not found for {emo_file}")
                continue

            # 读取 transcript
            trans_dict = read_transcript(trans_file)

            # 读取情绪标签文件
            with open(emo_file, "r", encoding="latin-1") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("["):
                        continue

                    # emo 文件行格式:
                    # Ses01F_impro01_F000 angry [1.234 - 4.234]
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue

                    utt_id = parts[1].strip()
                    emo_raw = parts[2].strip()

                    # 映射表过滤
                    emo = EMO_MAP.get(emo_raw)
                    if emo is None:
                        continue

                    # 取文本
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

    # 导出 CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["session", "dialog_id", "utterance_id", "text", "emotion"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Saved to: {OUTPUT_CSV}")
    print(f"Total utterances: {len(rows)}")


# 直接运行脚本
if __name__ == "__main__":
    build_iemocap_text_csv()
