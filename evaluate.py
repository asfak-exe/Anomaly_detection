import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


GROUND_TRUTH = r"C:\Users\6000_J1I\Desktop\asfak_project\PADS_AE\synthetic_tests\ground_truth.csv"

PREDICTIONS = r"C:\Users\6000_J1I\Desktop\asfak_project\PADS_AE\FINAL\output\run_8\predictions.csv"

gt = pd.read_csv(GROUND_TRUTH)

pred = pd.read_csv(PREDICTIONS)


gt = gt[
    [
        "Timestamp",
        "is_anomaly",
    ]
]

pred = pred[
    [
        "Timestamp",
        "is_anomaly",
    ]
]



merged = gt.merge(
    pred,
    on="Timestamp",
    suffixes=(
        "_true",
        "_pred",
    ),
)
y_true = merged["is_anomaly_true"]

y_pred = merged["is_anomaly_pred"]

print(merged.shape)

print(merged["is_anomaly_true"].value_counts())

print(merged["is_anomaly_pred"].value_counts())


tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
).ravel()


precision = precision_score(
    y_true,
    y_pred,
)

recall = recall_score(
    y_true,
    y_pred,
)

f1 = f1_score(
    y_true,
    y_pred,
)


print("Threshold:", threshold)

print(pred["loss"].describe())

print(pred[pred["is_anomaly"] == 1]["loss"].describe())
# print(gt["Timestamp"].head())
# print(pred["Timestamp"].head())
# print(gt.Timestamp.equals(pred.Timestamp))


# print(gt["is_anomaly"].sum())
# print(pred["is_anomaly"].sum())

# print(
#     gt[pred["is_anomaly_true"] == 1][
#         ["Timestamp", "loss", "is_anomaly_pred"]
#     ].head(20)
# )


# print("="*60)
# print("Evaluation")
# print("="*60)

# print(f"TP : {tp}")
# print(f"FP : {fp}")
# print(f"FN : {fn}")
# print(f"TN : {tn}")

# print()

# print(f"Precision : {precision:.4f}")
# print(f"Recall    : {recall:.4f}")
# print(f"F1 Score  : {f1:.4f}")