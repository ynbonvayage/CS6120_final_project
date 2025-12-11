from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report


def train_logreg(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        multi_class="auto"
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(512,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=20,
        verbose=True,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(name, model, X, y_true, target_labels=None):
    y_pred = model.predict(X)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n===== {name} =====")
    print("Macro-F1:", macro_f1)
    print(classification_report(y_true, y_pred, labels=target_labels))

    report_dict = classification_report(
        y_true, y_pred, labels=target_labels, output_dict=True
    )

    return macro_f1, report_dict, y_pred
