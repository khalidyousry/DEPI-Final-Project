import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, roc_auc_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────
df = pd.read_csv('Code/predictive_maintenance.csv')

le_type = LabelEncoder()
df['Type_enc'] = le_type.fit_transform(df['Type'])

FEATURES = ['Type_enc', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# 2. TASK 1 — Binary Classification
# ─────────────────────────────────────────
y_binary = df['Target']
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

binary_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
}

binary_rows = []
for name, model in binary_models.items():
    model.fit(X_tr1, y_tr1)
    y_pred = model.predict(X_te1)
    rep = classification_report(y_te1, y_pred, output_dict=True)
    binary_rows.append({
        "Model": name,
        "Task": "Binary",
        "Accuracy": round(accuracy_score(y_te1, y_pred), 4),
        "Precision_Failure": round(rep['1']['precision'], 4),
        "Recall_Failure":    round(rep['1']['recall'], 4),
        "F1_Failure":        round(rep['1']['f1-score'], 4),
        "Precision_NoFail":  round(rep['0']['precision'], 4),
        "Recall_NoFail":     round(rep['0']['recall'], 4),
        "F1_NoFail":         round(rep['0']['f1-score'], 4),
        "Macro_F1":          round(rep['macro avg']['f1-score'], 4),
        "Weighted_F1":       round(rep['weighted avg']['f1-score'], 4),
    })

# ─────────────────────────────────────────
# 3. TASK 2 — Multi-Class Classification
# ─────────────────────────────────────────
df['Failure Type'] = df['Failure Type'].replace('Random Failures', 'No Failure')
le_target = LabelEncoder()
y_multi = le_target.fit_transform(df['Failure Type'])
classes = list(le_target.classes_)

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_scaled, y_multi, test_size=0.2, random_state=42, stratify=y_multi)

multi_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                         objective='multi:softmax',
                                         num_class=len(classes),
                                         eval_metric='mlogloss'),
}

multi_rows = []
per_class_rows = []

for name, model in multi_models.items():
    model.fit(X_tr2, y_tr2)
    y_pred = model.predict(X_te2)
    rep = classification_report(y_te2, y_pred, target_names=classes, output_dict=True)

    multi_rows.append({
        "Model": name,
        "Task": "MultiClass",
        "Accuracy":    round(accuracy_score(y_te2, y_pred), 4),
        "Macro_F1":    round(rep['macro avg']['f1-score'], 4),
        "Weighted_F1": round(rep['weighted avg']['f1-score'], 4),
    })

    for cls in classes:
        per_class_rows.append({
            "Model":     name,
            "Class":     cls,
            "Precision": round(rep[cls]['precision'], 4),
            "Recall":    round(rep[cls]['recall'], 4),
            "F1":        round(rep[cls]['f1-score'], 4),
            "Support":   int(rep[cls]['support']),
        })

# ─────────────────────────────────────────
# 4. Save CSVs
# ─────────────────────────────────────────
df_binary  = pd.DataFrame(binary_rows)
df_multi   = pd.DataFrame(multi_rows)
df_perclass = pd.DataFrame(per_class_rows)
df_summary = pd.concat([
    df_binary[['Model','Task','Accuracy','Macro_F1','Weighted_F1']],
    df_multi[['Model','Task','Accuracy','Macro_F1','Weighted_F1']]
], ignore_index=True)

df_binary.to_csv('Code/results_binary.csv', index=False)
df_multi.to_csv('Code/results_multiclass.csv', index=False)
df_perclass.to_csv('Code/results_per_class.csv', index=False)
df_summary.to_csv('Code/results_summary.csv', index=False)

print("✅ CSVs saved:")
print("   Code/results_binary.csv")
print("   Code/results_multiclass.csv")
print("   Code/results_per_class.csv")
print("   Code/results_summary.csv")
print()
print(df_summary.to_string(index=False))
