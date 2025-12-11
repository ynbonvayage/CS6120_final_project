import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

from data_utils import (
    load_iemocap,
    split_data,
    build_tfidf,
    TARGET_EMOTIONS,
    OUTPUT_DIR,
)
from models import train_logreg, train_svm, train_mlp, evaluate_model


def plot_macro_f1_bar(results_dict, save_path: Path):
    models = list(results_dict.keys())
    scores = [results_dict[m] for m in models]

    plt.figure(figsize=(6, 4))
    plt.bar(models, scores)
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1 Comparison on IEMOCAP (Emotion)")
    plt.ylim(0.0, 1.0)
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_per_class_f1(
    reports_dict,
    target_emotions,
    save_path: Path
):
    models = list(reports_dict.keys())
    num_classes = len(target_emotions)
    x = np.arange(num_classes)
    width = 0.8 / len(models)  

    plt.figure(figsize=(10, 5))

    for idx, model_name in enumerate(models):
        report = reports_dict[model_name]
        f1_scores = []
        for emo in target_emotions:
            if emo in report:
                f1_scores.append(report[emo]["f1-score"])
            else:
                f1_scores.append(0.0)
        positions = x + (idx - len(models) / 2) * width + width / 2
        plt.bar(positions, f1_scores, width=width, label=model_name)

    plt.xticks(x, target_emotions, rotation=30)
    plt.ylabel("F1-score")
    plt.title("Per-class F1 Comparison on IEMOCAP")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_iemocap()
    print("Loaded IEMOCAP, shape:", df.shape)
    print(df["emotion"].value_counts())

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    vectorizer, X_train_vec = build_tfidf(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    logreg = train_logreg(X_train_vec, y_train)
    svm = train_svm(X_train_vec, y_train)
    mlp = train_mlp(X_train_vec, y_train)

    macro_f1_results = {}
    per_class_reports = {}

    for name, model in [
        ("LogReg", logreg),
        ("LinearSVC", svm),
        ("MLP", mlp),
    ]:
        macro_f1, report_dict, y_pred_test = evaluate_model(
            name + " (Test)",
            model,
            X_test_vec,
            y_test,
            target_labels=TARGET_EMOTIONS
        )
        macro_f1_results[name] = macro_f1
        per_class_reports[name] = report_dict

        pred_df = pd.DataFrame({
            "text": X_test,
            "gold_emotion": y_test,
            "pred_emotion": y_pred_test
        })
        pred_df.to_csv(OUTPUT_DIR / f"{name.lower()}_iemocap_test_predictions.csv", index=False)

    joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(logreg, OUTPUT_DIR / "logreg_model.joblib")
    joblib.dump(svm, OUTPUT_DIR / "svm_model.joblib")
    joblib.dump(mlp, OUTPUT_DIR / "mlp_model.joblib")
    print(f"Saved models and vectorizer to {OUTPUT_DIR}")

    plot_macro_f1_bar(
        macro_f1_results,
        OUTPUT_DIR / "macro_f1_comparison_iemocap.png"
    )

    plot_per_class_f1(
        per_class_reports,
        TARGET_EMOTIONS,
        OUTPUT_DIR / "per_class_f1_comparison_iemocap.png"
    )

    print("All done.")


if __name__ == "__main__":
    main()
