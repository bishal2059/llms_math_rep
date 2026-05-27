from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
        "report": classification_report(y_test, pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }