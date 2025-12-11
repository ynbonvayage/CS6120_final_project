import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent

IEMOCAP_CSV = BASE_DIR / "data" / "iemocap_utterances_text.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "iemocap_emotion_experiment"

EMO_MAP = {
    "ang": "anger",
    "hap": "happiness",
    "exc": "happiness",  
    "sad": "sadness",
    "fru": "frustration",
    # "fear": "fear",
    # "disg": "disgust",
    "neu": "neutral",
    "oth": "neutral",    
    "xxx": "neutral",     #
}

TARGET_EMOTIONS = [
    "anger", "happiness", "sadness",
    "frustration", 
    # "fear", "disgust",
    "neutral"
]


def load_iemocap(csv_path: Path | str | None = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = IEMOCAP_CSV

    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "emotion" not in df.columns:
        raise ValueError("CSV must contain at least 'text' and 'emotion' columns.")

    df["emotion"] = df["emotion"].map(EMO_MAP).fillna(df["emotion"])

    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    return df


def split_data(df: pd.DataFrame):
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
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    return vectorizer, X_train_vec
