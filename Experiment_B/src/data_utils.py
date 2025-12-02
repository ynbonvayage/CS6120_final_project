# Experiment_B/src/data_utils.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 工程根目录（Experiment_B）
BASE_DIR = Path(__file__).resolve().parent.parent

# IEMOCAP 预处理后的 CSV 路径
IEMOCAP_CSV = BASE_DIR / "data" / "iemocap_utterances_text.csv"

# 输出目录
OUTPUT_DIR = BASE_DIR / "outputs" / "iemocap_emotion_experiment"

# IEMOCAP 原始标签到统一标签的映射
EMO_MAP = {
    "ang": "anger",
    "hap": "happiness",
    "exc": "happiness",   # 也可以保留为 "excited"
    "sad": "sadness",
    "fru": "frustration",
    # "fear": "fear",
    # "disg": "disgust",
    "neu": "neutral",
    "oth": "neutral",     # 视情况你也可以丢弃这些
    "xxx": "neutral",     # 无效标注可统一成 neutral 或丢弃
}

# 关心的情绪类别列表（用于画每类 F1 图）
TARGET_EMOTIONS = [
    "anger", "happiness", "sadness",
    "frustration", 
    # "fear", "disgust",
    "neutral"
]


def load_iemocap(csv_path: Path | str | None = None) -> pd.DataFrame:
    """
    从预处理好的 IEMOCAP CSV 加载数据，并映射情绪标签。
    期望至少有 'text', 'emotion' 两列。
    """
    if csv_path is None:
        csv_path = IEMOCAP_CSV

    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "emotion" not in df.columns:
        raise ValueError("CSV must contain at least 'text' and 'emotion' columns.")

    # 如果你的 CSV 里的 emotion 已经是英文标签（anger 等），可以注释掉这一行
    df["emotion"] = df["emotion"].map(EMO_MAP).fillna(df["emotion"])

    # 丢掉文本为空的行
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    return df


def split_data(df: pd.DataFrame):
    """
    将 IEMOCAP 数据随机划分为 train / val / test。
    也可以根据 session / dialog 做划分，这里先用 stratified split。
    """
    X = df["text"].tolist()
    y = df["emotion"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_tfidf(X_train, ngram_range=(1, 2), max_features=30000):
    """
    使用 TF-IDF 将文本转换为向量。
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    return vectorizer, X_train_vec
