import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("results.csv")

threshold = 0.5
df["Predicted"] = (df["Similarity"] >= threshold).astype(int)
if "Actual" not in df.columns:
    print("❗ Add an 'Actual' column (1=match, 0=no match) to results.csv for evaluation.")
else:
    acc = accuracy_score(df["Actual"], df["Predicted"])
    prec = precision_score(df["Actual"], df["Predicted"])
    rec = recall_score(df["Actual"], df["Predicted"])
    f1 = f1_score(df["Actual"], df["Predicted"])
    auc = roc_auc_score(df["Actual"], df["Similarity"])
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1: {f1:.2f}")
    print(f"ROC-AUC: {auc:.2f}")
