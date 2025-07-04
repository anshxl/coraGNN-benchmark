import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from data_loader import load_cora

def main():
    # Load data
    data = load_cora("data")
    X = data.x.numpy()
    y = data.y.numpy()

    # Split according to masks
    train_idx = data.train_mask.numpy()
    val_idx = data.val_mask.numpy()
    test_idx = data.test_mask.numpy()

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train logistic regression model
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    # Evaluate
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_pred = clf.predict(X_test)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    val_f1 = f1_score(y_val, val_pred, average='macro')
    test_f1 = f1_score(y_test, test_pred, average='macro')

    print(f"[Baseline] Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    print(f"[Baseline] Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()