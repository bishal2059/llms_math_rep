from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score



def evaluate_classifier(model, X_test, y_test):
    predictions = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro"
        ),
        "report": classification_report(
            y_test,
            predictions
        )
    }

    return results