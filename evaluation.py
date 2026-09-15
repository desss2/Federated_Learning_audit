from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss


# Evaluation helpers
def evaluate(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1