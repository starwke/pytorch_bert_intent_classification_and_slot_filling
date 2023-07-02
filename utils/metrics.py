def get_metrices(trues, preds, mode):
    if mode == "cls":
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        acc = accuracy_score(trues, preds)
        precision = precision_score(trues, preds, average="micro")
        recall = recall_score(trues, preds, average="micro")
        f1 = f1_score(trues, preds, average="micro")
    elif mode == "ner":
        from seqeval.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        acc = accuracy_score(trues, preds)
        precision = precision_score(trues, preds)
        recall = recall_score(trues, preds)
        f1 = f1_score(trues, preds)
    return acc, precision, recall, f1

def get_report(trues, preds, mode):
    if mode == "cls":
        from sklearn.metrics import classification_report

        report = classification_report(trues, preds)
    elif mode == "ner":
        from seqeval.metrics import classification_report

        report = classification_report(trues, preds)
    return report