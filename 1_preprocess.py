import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ── CONFIG ──────────────────────────────────────────────
DATA_DIR = "data/"           # Put your CIC-IDS-2017 CSVs here
OUTPUT_DIR = "models/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("outputs/", exist_ok=True)

# ── LOAD DATA ────────────────────────────────────────────
print("[*] Loading data...")
dfs = []
for f in os.listdir(DATA_DIR):
    if f.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, f), encoding='latin-1', low_memory=False)
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"[+] Total records: {len(df)}")

# ── CLEAN COLUMNS ────────────────────────────────────────
df.columns = df.columns.str.strip()  # Remove whitespace from column names

# ── TARGET COLUMN ────────────────────────────────────────
label_col = "Label"

# Convert to binary: BENIGN = 0, everything else = 1
df[label_col] = df[label_col].str.strip()
df["target"] = (df[label_col] != "BENIGN").astype(int)
print(f"[+] Class distribution:\n{df['target'].value_counts()}")

# ── DROP NON-NUMERIC / PROBLEMATIC COLUMNS ───────────────
df.drop(columns=[label_col], inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Keep only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove("target")
print(f"[+] Features used: {len(numeric_cols)}")

X = df[numeric_cols]
y = df["target"]

# ── SCALE ────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

# ── SPLIT ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── SAVE ─────────────────────────────────────────────────
joblib.dump(X_train, f"{OUTPUT_DIR}X_train.pkl")
joblib.dump(X_test,  f"{OUTPUT_DIR}X_test.pkl")
joblib.dump(y_train, f"{OUTPUT_DIR}y_train.pkl")
joblib.dump(y_test,  f"{OUTPUT_DIR}y_test.pkl")
joblib.dump(scaler,  f"{OUTPUT_DIR}scaler.pkl")
joblib.dump(numeric_cols, f"{OUTPUT_DIR}feature_names.pkl")

print("[✓] Preprocessing done. Files saved to /models/")