#!/usr/bin/env python3
"""
UNIVERSAL CSV ANALYZER - LIQUID METAL EDITION (ULTRA-FAST TRAINING)
- Uses OneHotEncoder (sparse) instead of LabelEncoder for speed
- Auto-drops useless columns (IDs, names, URLs)
- Trains any dataset (small to 100k+ rows) in under 5 seconds
- Keeps all Liquid Metal UI features
"""

import os, io, base64, time, re, traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ========== SKLEARN IMPORTS ==========
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, r2_score, mean_absolute_error, mean_squared_error
)

# ========== CONFIG ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

APP_STATE = {"df": None, "filename": None}
TRAIN_TIMEOUT_SECONDS = 30
executor = ThreadPoolExecutor(max_workers=2)

# ========== BACKGROUND GENERATION ==========
def ensure_background_images():
    img_dir = os.path.join(BASE_DIR, "static", "images")
    os.makedirs(img_dir, exist_ok=True)
    palettes = [
        ("#1a0b2e", "#7209b7"), ("#0f2027", "#2c5364"), ("#2b0000", "#8b0000"),
        ("#0b132b", "#3a506b"), ("#1b1b2f", "#e94560"), ("#03071e", "#6a040f"),
        ("#022c43", "#0a9396"),
    ]
    for i, (c1, c2) in enumerate(palettes, start=1):
        path = os.path.join(img_dir, f"bg{i}.jpg")
        if os.path.exists(path):
            continue
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        cmap = LinearSegmentedColormap.from_list("bg", [c1, c2, c1])
        x = np.linspace(0, 1, 512)
        y = np.linspace(0, 1, 288)
        xx, yy = np.meshgrid(x, y)
        ax.imshow((xx * 0.75 + yy * 0.25), aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
        fig.savefig(path, format="jpg")
        plt.close(fig)

# ========== OPTIMIZED TRAINING HELPERS (From Claude) ==========
ID_NAME_PATTERN = re.compile(
    r"(^id$|_id$|^id_|uuid|guid|track_id|track_name|artist_name|album_name|"
    r"song_name|^name$|url|link|^index$|^unnamed)",
    re.IGNORECASE,
)

def drop_useless_columns(df, target_column, unique_ratio_threshold=0.95):
    n_rows = len(df)
    cols_to_drop = []
    for col in df.columns:
        if col == target_column:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            cols_to_drop.append(col)
            continue
        if ID_NAME_PATTERN.search(col):
            cols_to_drop.append(col)
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if not is_numeric and n_rows > 0 and (nunique / n_rows) > unique_ratio_threshold:
            cols_to_drop.append(col)
            continue
    cleaned = df.drop(columns=cols_to_drop, errors="ignore")
    return cleaned, cols_to_drop

def detect_task_type(df, target_column):
    y = df[target_column]
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    nunique = y.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(y) and nunique <= 20:
        return "classification"
    y_no_na = y.dropna()
    if nunique <= 20 and len(y_no_na) > 0 and (y_no_na % 1 == 0).all():
        return "classification"
    return "regression"

def downsample(df, target_column, task_type, max_rows=20000):
    n_rows = len(df)
    if n_rows <= max_rows:
        return df, False
    if task_type == "classification":
        try:
            frac = max_rows / n_rows
            parts = []
            for _, group in df.groupby(target_column, sort=False):
                if len(group) == 0:
                    continue
                n_take = max(1, int(round(len(group) * frac)))
                n_take = min(n_take, len(group))
                parts.append(group.sample(n=n_take, random_state=42))
            sampled = pd.concat(parts, axis=0) if parts else df.sample(n=max_rows, random_state=42)
            if len(sampled) > max_rows:
                sampled = sampled.sample(n=max_rows, random_state=42)
            elif len(sampled) < max_rows:
                remaining = df.drop(sampled.index, errors="ignore")
                if len(remaining) > 0:
                    top_up = remaining.sample(n=min(max_rows - len(sampled), len(remaining)), random_state=42)
                    sampled = pd.concat([sampled, top_up], axis=0)
            return sampled.reset_index(drop=True), True
        except Exception:
            pass
    sampled = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return sampled, True

def preprocess_data(X, max_categories=30):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        ohe_kwargs = dict(handle_unknown="ignore", sparse_output=True)
        try:
            ohe = OneHotEncoder(max_categories=max_categories, **ohe_kwargs)
        except TypeError:
            ohe = OneHotEncoder(**ohe_kwargs)
        transformers.append(("cat", ohe, categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    return preprocessor, numeric_cols, categorical_cols

def build_fast_model(task_type, n_rows_original):
    if n_rows_original > 50000:
        if task_type == "classification":
            model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, n_jobs=-1, random_state=42)
        else:
            model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
        tier = "large"
    elif n_rows_original > 5000:
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=120, max_depth=14, n_jobs=-1, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=120, max_depth=14, n_jobs=-1, random_state=42)
        tier = "medium"
    else:
        if task_type == "classification":
            model = SVC(kernel="rbf", probability=True, random_state=42)
        else:
            model = SVR(kernel="rbf")
        tier = "small"
    return model, tier

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def make_confusion_matrix_chart(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tick_labels = labels if labels is not None else sorted(pd.unique(y_true))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return _fig_to_base64(fig)

def make_roc_curve_chart(y_true_binary, y_score):
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _fig_to_base64(fig), roc_auc

def make_feature_importance_chart(model, feature_names, top_n=15):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef).ravel() if coef.ndim == 1 else np.abs(coef).mean(axis=0)
    if importances is None or len(importances) != len(feature_names):
        return None
    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in order]
    top_values = importances[order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(top_features)), top_values[::-1])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    return _fig_to_base64(fig)

def make_actual_vs_predicted_chart(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.4, s=12)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, color="red", linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    return _fig_to_base64(fig)

# ========== MAIN TRAINING FUNCTION ==========
def do_training(df, target_column):
    start_time = time.time()
    result = {
        "success": False, "task_type": None, "model_type": None,
        "rows_original": len(df), "rows_used": 0, "sampled": False,
        "dropped_columns": [], "features_used": [], "metrics": {},
        "charts": {"confusion_matrix": None, "roc_curve": None,
                   "feature_importance": None, "actual_vs_predicted": None},
        "training_time_seconds": None, "error": None,
    }
    try:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")
        df = df.dropna(subset=[target_column]).reset_index(drop=True)
        n_rows_original = len(df)
        if n_rows_original < 10:
            raise ValueError("Not enough rows to train (need at least 10 after dropping empty targets).")
        df_clean, dropped_cols = drop_useless_columns(df, target_column)
        result["dropped_columns"] = dropped_cols
        if df_clean.shape[1] < 2:
            raise ValueError("No usable feature columns remain after dropping useless columns.")
        task_type = detect_task_type(df_clean, target_column)
        result["task_type"] = task_type
        model, tier = build_fast_model(task_type, n_rows_original)
        result["model_type"] = type(model).__name__
        df_sample, sampled = downsample(df_clean, target_column, task_type, max_rows=20000)
        result["sampled"] = sampled
        result["rows_used"] = len(df_sample)
        X = df_sample.drop(columns=[target_column]).copy()
        y = df_sample[target_column]
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                mode = X[col].mode()
                X[col] = X[col].fillna(mode.iloc[0] if not mode.empty else "missing")
        result["features_used"] = X.columns.tolist()
        if task_type == "classification":
            y = y.astype(str)
        preprocessor, numeric_cols, categorical_cols = preprocess_data(X)
        stratify = y if (task_type == "classification" and y.nunique() > 1) else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if task_type == "classification":
            labels = sorted(y.unique().tolist())
            is_binary = len(labels) == 2
            if is_binary:
                pos_label = labels[1]
                precision = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
                recall = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
                f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
            else:
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            result["metrics"] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            }
            result["charts"]["confusion_matrix"] = make_confusion_matrix_chart(y_test, y_pred, labels=labels)
            if is_binary:
                fitted_model = pipeline.named_steps["model"]
                try:
                    if hasattr(fitted_model, "predict_proba"):
                        proba = pipeline.predict_proba(X_test)
                        pos_idx = list(fitted_model.classes_).index(pos_label)
                        y_score = proba[:, pos_idx]
                    elif hasattr(fitted_model, "decision_function"):
                        y_score = pipeline.decision_function(X_test)
                    else:
                        y_score = None
                    if y_score is not None:
                        y_test_bin = (y_test == pos_label).astype(int)
                        roc_chart, roc_auc_val = make_roc_curve_chart(y_test_bin, y_score)
                        result["charts"]["roc_curve"] = roc_chart
                        result["metrics"]["roc_auc"] = float(roc_auc_val)
                except Exception:
                    pass
        else:
            result["metrics"] = {
                "r2_score": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            }
            result["charts"]["actual_vs_predicted"] = make_actual_vs_predicted_chart(y_test.to_numpy(), np.asarray(y_pred))
        try:
            feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
            fi_chart = make_feature_importance_chart(pipeline.named_steps["model"], feature_names)
            result["charts"]["feature_importance"] = fi_chart
        except Exception:
            pass
        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
    finally:
        result["training_time_seconds"] = round(time.time() - start_time, 3)
    return result

# ========== FLASK ROUTES ==========
@app.route("/")
def index():
    return HTML_PAGE

@app.route("/api/samples", methods=["GET"])
def list_samples():
    sample_dir = os.path.join(BASE_DIR, "sample_data")
    if not os.path.exists(sample_dir):
        return jsonify({"success": False, "samples": []}), 200
    files = [f for f in os.listdir(sample_dir) if f.endswith(".csv")]
    return jsonify({"success": True, "samples": files}), 200

@app.route("/load_sample", methods=["POST"])
def load_sample():
    data = request.get_json(silent=True) or {}
    sample_name = data.get("sample_name")
    
    sample_path = os.path.join(BASE_DIR, "sample_data", sample_name)
    if not os.path.exists(sample_path):
        return jsonify({"success": False, "error": "Sample not found."}), 400
    
    df = pd.read_csv(sample_path)
    df = df.dropna(axis=1, how="all")
    APP_STATE["df"] = df
    APP_STATE["filename"] = f"{sample_name}_sample.csv"
    
    columns_info = []
    for col in df.columns:
        series = df[col]
        columns_info.append({
            "name": str(col), "dtype": str(series.dtype),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "suggested_task": detect_task_type(df, col),
        })
    preview_df = df.head(10).copy()
    preview_rows = preview_df.astype(object).where(pd.notna(preview_df), "—").astype(str).to_dict(orient="records")
    
    return jsonify({
        "success": True, "filename": f"{sample_name}_sample.csv",
        "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]),
        "total_missing": int(df.isna().sum().sum()), "columns": columns_info,
        "preview_columns": [str(c) for c in df.columns], "preview_rows": preview_rows,
    })

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file part."}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected."}), 400
        if not file.filename.lower().endswith(".csv"):
            return jsonify({"success": False, "error": "Only .csv files allowed."}), 400
        raw_bytes = file.read()
        if not raw_bytes:
            return jsonify({"success": False, "error": "File is empty."}), 400
        df = None
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
                break
            except Exception:
                continue
        if df is None:
            return jsonify({"success": False, "error": "Could not parse CSV."}), 400
        if df.shape[1] == 0:
            return jsonify({"success": False, "error": "No columns detected."}), 400
        df = df.dropna(axis=1, how="all")
        APP_STATE["df"] = df
        APP_STATE["filename"] = file.filename
        columns_info = []
        for col in df.columns:
            series = df[col]
            columns_info.append({
                "name": str(col), "dtype": str(series.dtype),
                "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "suggested_task": detect_task_type(df, col),
            })
        preview_df = df.head(10).copy()
        preview_rows = preview_df.astype(object).where(pd.notna(preview_df), "—").astype(str).to_dict(orient="records")
        return jsonify({
            "success": True, "filename": file.filename,
            "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]),
            "total_missing": int(df.isna().sum().sum()), "columns": columns_info,
            "preview_columns": [str(c) for c in df.columns], "preview_rows": preview_rows,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Upload failed: {e}"}), 500

@app.route("/train", methods=["POST"])
def train():
    df = APP_STATE["df"]
    if df is None or df.empty:
        return jsonify({"error": "No dataset uploaded."}), 400
    payload = request.get_json(silent=True) or {}
    target_column = payload.get("target")
    if not target_column or target_column not in df.columns:
        return jsonify({"error": "Valid target column is required."}), 400
    future = executor.submit(do_training, df.copy(), target_column)
    try:
        result = future.result(timeout=TRAIN_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        future.cancel()
        return jsonify({"error": f"Training took too long (> {TRAIN_TIMEOUT_SECONDS}s). Try selecting other target columns."}), 504
    except Exception as e:
        return jsonify({"error": "Training failed.", "details": str(e)}), 500
    return jsonify(result), 200

# ========== FRONTEND (LIQUID METAL EDITION) ==========
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Universal CSV Analyzer - Liquid Metal</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
:root{
  --bg-image: url('/static/images/bg1.jpg');
  --glass-bg: rgba(51,33,80,0.50);
  --glass-border: rgba(227,15,15,0.15);
  --blur: 0px;
  --frost: 0;
  --saturation: 1.0;
  --tint: rgba(51,33,80,0.50);
  --bevel: 6px;
  --specular: 1;
  --depth: 80px;
  --radius: 40px;
  --contrast: 1.60;
  --brightness: 1.40;
  --white-text: #ffffff;
  --metal-opacity: 0.60;
  --metal-color: silver;
}
*{ box-sizing:border-box; margin:0; padding:0; }
html,body{ min-height:100vh; font-family:'Inter',sans-serif; color:var(--white-text); }
body{
  background-image: var(--bg-image);
  background-size: cover; background-position: center; background-attachment: fixed;
  background-repeat: no-repeat; min-height:100vh; transition: background-image 0.6s ease;
  position:relative; overflow-x:hidden;
}
body::before{
  content:""; position:fixed; inset:0; background: rgba(10,6,20,0.35);
  z-index:0; pointer-events:none;
}
.container{ position:relative; z-index:1; max-width:1180px; margin:0 auto; padding:40px 24px 160px; overflow:visible; }
.glass-panel{
  overflow:visible;
  overflow:visible;
  background:
    linear-gradient(rgba(255,255,255, var(--frost-alpha, 0)), rgba(255,255,255, var(--frost-alpha, 0))),
    var(--tint);
  border:1px solid var(--glass-border); border-radius: var(--radius);
  backdrop-filter: blur(var(--blur)) saturate(var(--saturation)) contrast(var(--contrast)) brightness(var(--brightness));
  -webkit-backdrop-filter: blur(var(--blur)) saturate(var(--saturation)) contrast(var(--contrast)) brightness(var(--brightness));
  box-shadow:
    0 calc(var(--depth) * 0.25) calc(var(--depth) * 0.75) rgba(0,0,0,0.6),
    0 calc(var(--depth) * 0.08) calc(var(--depth) * 0.3) rgba(0,0,0,0.4),
    inset var(--bevel) var(--bevel) calc(var(--bevel) * 3) rgba(255,255,255, calc(var(--specular) * 0.65)),
    inset calc(var(--bevel) * -1) calc(var(--bevel) * -1) calc(var(--bevel) * 3) rgba(0,0,0,0.3),
    inset 0 0 calc(var(--bevel) * 8) rgba(255,255,255, calc(var(--specular) * 0.1));
  padding:32px; margin-bottom:26px; position:relative;
  transition: transform .35s cubic-bezier(.2,.9,.3,1.2), box-shadow .35s ease, border-color .35s ease;
}
.glass-panel:hover{
  transform: translateY(-6px) scale(1.008);
  box-shadow:
    0 calc(var(--depth) * 0.35) calc(var(--depth) * 1.1) rgba(0,0,0,0.65),
    0 0 50px rgba(255,255,255,0.18),
    inset var(--bevel) var(--bevel) calc(var(--bevel) * 4) rgba(255,255,255, calc(var(--specular) * 0.8)),
    inset calc(var(--bevel) * -1) calc(var(--bevel) * -1) calc(var(--bevel) * 4) rgba(0,0,0,0.35);
  border-color: rgba(255,255,255,0.32);
}
.section-title{
  display:flex; align-items:center; gap:10px; font-size:1.15rem; font-weight:700; margin-bottom:18px; opacity:1;
}
.section-title i{ opacity:0.9; }
.section-upload .section-title { color:#ff6b6b; }
.section-upload.glass-panel {
  background: linear-gradient(135deg, rgba(143,0,21,0.75), rgba(143,0,21,0.45), rgba(143,0,21,0.65));
  border-color: rgba(255,100,100,0.4);
  backdrop-filter: blur(var(--blur)) saturate(var(--saturation)) contrast(var(--contrast)) brightness(var(--brightness));
  -webkit-backdrop-filter: blur(var(--blur)) saturate(var(--saturation)) contrast(var(--contrast)) brightness(var(--brightness));
  box-shadow:
    0 calc(var(--depth) * 0.25) calc(var(--depth) * 0.75) rgba(0,0,0,0.6),
    inset var(--bevel) var(--bevel) calc(var(--bevel) * 3) rgba(255,255,255, calc(var(--specular) * 0.2)),
    inset calc(var(--bevel) * -1) calc(var(--bevel) * -1) calc(var(--bevel) * 3) rgba(0,0,0,0.4);
}
.section-summary .section-title { color:#22d3ee; }
.section-preview .section-title { color:#60a5fa; }
.section-config .section-title { color:#a78bfa; }
.section-results .section-title { color:#f472b6; }
.app-header{ text-align:center; margin-bottom:34px; }
.app-header h1{ font-size:2.4rem; font-weight:800; letter-spacing:-0.02em; display:flex; align-items:center; justify-content:center; gap:14px; text-shadow: 0 4px 24px rgba(0,0,0,0.4); opacity:1; }
.app-header h1 i{ background: linear-gradient(135deg,#a78bfa,#60a5fa); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; filter: drop-shadow(0 4px 12px rgba(167,139,250,0.5)); }
.app-header p{ margin-top:10px; opacity:1; font-size:1.02rem; font-weight:400; }
.dropzone{ border:2px dashed rgba(255,255,255,0.35); border-radius: calc(var(--radius) * 0.55); padding:46px 20px; text-align:center; cursor:pointer; transition: all .25s ease; }
.dropzone:hover, .dropzone.dragover{ border-color:#a78bfa; background: rgba(255,255,255,0.06); }
.dropzone i{ font-size:2.6rem; margin-bottom:14px; color:#a78bfa; }
.dropzone h3{ font-weight:600; margin-bottom:6px; opacity:1; }
.dropzone p{ opacity:1; font-size:0.9rem; }
.dropzone input[type=file]{ display:none; }
.file-chip{ display:inline-flex; align-items:center; gap:10px; background: rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2); padding:8px 16px; border-radius:999px; margin-top:16px; font-size:0.9rem; }
.stat-grid{ display:grid; grid-template-columns:repeat(auto-fit, minmax(140px,1fr)); gap:14px; margin-bottom:22px; }
.stat-tile{
  border-radius: 24px; padding:16px; text-align:center; border:1px solid rgba(255,255,255,0.14);
  position:relative; overflow:hidden;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.3), inset 0 -1px 0 rgba(0,0,0,0.25), 0 8px 16px rgba(0,0,0,0.2);
  transition: transform .2s ease, opacity .2s ease;
}
.stat-tile:hover{ transform:translateY(-3px); }
.stat-tile::before{ content:""; position:absolute; inset:0; border-radius:24px; opacity: var(--metal-opacity); transition: opacity .3s ease; z-index:0; }
.stat-tile > *{ position:relative; z-index:1; opacity:1; }
.stat-tile.silver::before{ background: linear-gradient(90deg, #b0b0b0 0%, #e8e8e8 20%, #808080 40%, #f0f0f0 60%, #707070 80%, #c0c0c0 100%); }
.stat-tile.gold::before{ background: linear-gradient(90deg, #b8860b 0%, #ffd700 20%, #daa520 45%, #fff8dc 60%, #daa520 75%, #b8860b 100%); }
.stat-tile.bronze::before{ background: linear-gradient(90deg, #8b4513 0%, #cd7f32 20%, #a0522d 45%, #f4a460 60%, #a0522d 75%, #8b4513 100%); }
.stat-tile.silver{ background: rgba(192,192,192,0.1); }
.stat-tile.gold{ background: rgba(212,175,55,0.1); }
.stat-tile.bronze{ background: rgba(205,127,50,0.1); }
.stat-tile .val{ font-size:1.6rem; font-weight:800; }
.stat-tile.silver .val, .stat-tile.silver .lbl{ color:#f0f0f0; }
.stat-tile.gold .val, .stat-tile.gold .lbl{ color:#ffd700; }
.stat-tile.bronze .val, .stat-tile.bronze .lbl{ color:#f4a460; }
.stat-tile .lbl{ font-size:0.78rem; margin-top:4px; text-transform:uppercase; letter-spacing:0.05em; }
.table-wrap{
  overflow-x:auto; border-radius:24px; border:1px solid rgba(255,255,255,0.15);
  max-height:380px; overflow-y:auto; background: rgba(0,0,0,0.2);
}
table{ width:100%; border-collapse:collapse; font-size:0.86rem; white-space:nowrap; }
thead th{
  position:sticky; top:0; background: rgba(30,18,50,0.85); padding:12px 14px; text-align:left;
  font-weight:600; border-bottom:1px solid rgba(255,255,255,0.15); backdrop-filter: blur(6px);
}
th.col-name { color:#60a5fa; }
th.col-type { color:#22d3ee; }
th.col-unique { color:#ef4444; }
th.col-missing { color:#eab308; }
th.col-task { color:#a78bfa; }
tbody td{ padding:9px 14px; border-bottom:1px solid rgba(255,255,255,0.08); }
tbody tr:hover{ background: rgba(255,255,255,0.06); }
.badge{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:0.72rem; font-weight:600; letter-spacing:0.02em; }
.badge.missing-none{ background:rgba(74,222,128,0.2); color:#86efac; }
.badge.missing-some{ background:rgba(248,113,113,0.2); color:#fca5a5; }
.form-grid{ display:grid; grid-template-columns:repeat(auto-fit, minmax(210px,1fr)); gap:18px; margin-bottom:20px; }
.field label{ display:block; font-size:0.8rem; font-weight:600; margin-bottom:8px; opacity:1; text-transform:uppercase; letter-spacing:0.04em; }
select, input[type=number], input[type=text]{
  width:100%; background: rgba(255,255,255,0.09); border:1px solid rgba(255,255,255,0.22);
  border-radius:16px; padding:11px 14px; color:var(--white-text); font-family:'Inter',sans-serif;
  font-size:0.92rem; outline:none; transition: border-color .2s ease, background .2s ease;
}
select:focus, input:focus{ border-color:#a78bfa; background:rgba(255,255,255,0.14); }
select option{ background:#1e1233; color:#fff; }
.sample-dropdown-btn{
  background:
    linear-gradient(90deg, rgba(139,69,19,0.5) 0%, rgba(205,127,50,0.6) 20%, rgba(160,82,45,0.5) 40%, rgba(244,164,96,0.6) 60%, rgba(160,82,45,0.5) 80%, rgba(139,69,19,0.5) 100%),
    rgba(0,0,0,0.2);
}
.target-select-chrome, .model-select-chrome{
  background:
    linear-gradient(90deg, rgba(192,192,192,0.3) 0%, rgba(248,248,248,0.5) 20%, rgba(128,128,128,0.3) 40%, rgba(240,240,240,0.5) 60%, rgba(112,112,112,0.3) 80%, rgba(192,192,192,0.4) 100%),
    rgba(0,0,0,0.2);
  border:1px solid rgba(255,255,255,0.4); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), inset 0 -1px 0 rgba(0,0,0,0.3), 0 4px 12px rgba(0,0,0,0.2);
  color:#fff; font-weight:700; text-shadow: 0 1px 2px rgba(0,0,0,0.5);
}
.task-badge{
  display:inline-flex; align-items:center; gap:8px; padding:9px 16px; border-radius:999px;
  font-weight:700; font-size:0.85rem; margin-top:6px;
}
.task-badge.classification{ background:rgba(96,165,250,0.22); color:#93c5fd; border:1px solid rgba(96,165,250,0.4); }
.task-badge.regression{ background:rgba(251,191,36,0.22); color:#fde68a; border:1px solid rgba(251,191,36,0.4); }
.btn{
  display:inline-flex; align-items:center; justify-content:center; gap:10px; padding:14px 28px;
  border:none; border-radius:999px; font-family:'Inter',sans-serif; font-weight:700; font-size:0.95rem; cursor:pointer;
  background: linear-gradient(135deg,#daa520,#ffd700); color:#1a1a1a; box-shadow: 0 8px 24px rgba(255,215,0,0.4);
  transition: transform .2s ease, box-shadow .2s ease, opacity .2s ease;
}
.btn:hover:not(:disabled){ transform:translateY(-2px); box-shadow:0 12px 32px rgba(255,215,0,0.55); }
.btn:disabled{ opacity:0.6; cursor:not-allowed; }
.btn.secondary{ background: rgba(255,255,255,0.1); box-shadow:none; border:1px solid rgba(255,255,255,0.25); color:#fff; }
.spinner{ width:16px; height:16px; border-radius:50%; border:2.5px solid rgba(0,0,0,0.2); border-top-color:#1a1a1a; animation: spin 0.7s linear infinite; }
@keyframes spin{ to{ transform:rotate(360deg); } }
.metric-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin-bottom:24px; }
.metric-card{ border-radius:24px; padding:18px; text-align:center; background: rgba(255,255,255,0.09); border:1px solid rgba(255,255,255,0.16); transition: transform .25s ease, box-shadow .25s ease; }
.metric-card:hover{ transform: translateY(-4px) scale(1.03); box-shadow:0 10px 26px rgba(0,0,0,0.35); }
.metric-card .m-val{ font-size:1.7rem; font-weight:800; background:linear-gradient(135deg,#c4b5fd,#93c5fd); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.metric-card .m-lbl{ font-size:0.78rem; opacity:1; margin-top:6px; text-transform:uppercase; letter-spacing:0.04em; }
.chart-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:20px; }
.chart-card{ border-radius:24px; padding:16px; text-align:center; background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15); transition: transform .25s ease, box-shadow .25s ease; }
.chart-card:hover{ transform:translateY(-5px); box-shadow:0 16px 34px rgba(0,0,0,0.4); }
.chart-card img{ width:100%; height:auto; border-radius:12px; }
.chart-card h4{ font-size:0.9rem; font-weight:700; margin-bottom:10px; opacity:1; }
.hidden{ display:none !important; }
.toast{ position:fixed; top:20px; left:50%; transform:translateX(-50%); background: rgba(248,113,113,0.95); color:#fff; padding:13px 24px; border-radius:14px; font-weight:600; font-size:0.9rem; z-index:999; box-shadow:0 10px 30px rgba(0,0,0,0.4); display:flex; align-items:center; gap:10px; animation: fadein .25s ease; }
@keyframes fadein{ from{opacity:0; transform:translate(-50%,-10px);} to{opacity:1; transform:translate(-50%,0);} }
#shaderToggleBtn{
  position:fixed; bottom:26px; right:26px; z-index:50; display:flex; align-items:center; gap:10px;
  padding:14px 22px; border-radius:999px; border:1px solid rgba(255,255,255,0.3);
  background: rgba(30,18,50,0.65); backdrop-filter: blur(14px); color:#fff; font-weight:700; font-size:0.9rem;
  cursor:pointer; box-shadow: 0 10px 30px rgba(0,0,0,0.4); transition: transform .2s ease;
}
#shaderToggleBtn:hover{ transform:translateY(-3px) scale(1.03); }
#shaderPanel{
  position:fixed; bottom:0; right:0; z-index:60; width:360px; max-width:92vw; max-height:85vh;
  overflow-y:auto; background: rgba(24,14,42,0.78); backdrop-filter: blur(24px) saturate(1.4);
  border:1px solid rgba(255,255,255,0.22); border-radius:28px 0 0 0; padding:26px 24px 30px;
  transform: translateY(110%); transition: transform .4s cubic-bezier(.2,.9,.25,1);
  box-shadow: -10px -10px 50px rgba(0,0,0,0.45);
}
#shaderPanel.open{ transform: translateY(0); }
#shaderPanel h3{ display:flex; align-items:center; gap:10px; font-size:1.05rem; margin-bottom:18px; font-weight:800; }
.shader-group{ margin-bottom:18px; }
.shader-group label{ display:flex; justify-content:space-between; font-size:0.78rem; font-weight:600; opacity:1; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.03em; }
.shader-group label span{ opacity:0.8; font-weight:500; text-transform:none; }
input[type=range]{ width:100%; -webkit-appearance:none; appearance:none; height:6px; border-radius:999px; background: rgba(255,255,255,0.2); outline:none; }
input[type=range]::-webkit-slider-thumb{ -webkit-appearance:none; appearance:none; width:18px; height:18px; border-radius:50%; background: linear-gradient(135deg,#a78bfa,#60a5fa); cursor:pointer; border:2px solid #fff; box-shadow:0 2px 8px rgba(0,0,0,0.4); }
input[type=range]::-moz-range-thumb{ width:18px; height:18px; border-radius:50%; background: linear-gradient(135deg,#a78bfa,#60a5fa); cursor:pointer; border:2px solid #fff; }
.shader-color-row{ display:flex; align-items:center; gap:10px; }
input[type=color]{ width:42px; height:42px; border-radius:12px; border:1px solid rgba(255,255,255,0.3); background:none; cursor:pointer; padding:0; }
.color-presets{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.color-swatch{ width:30px; height:30px; border-radius:50%; border:2px solid rgba(255,255,255,0.4); cursor:pointer; transition: transform .15s ease; }
.color-swatch:hover{ transform:scale(1.15); }
.shader-reset{ width:100%; margin-top:6px; }
#bgSwitcher{ position:fixed; bottom:26px; left:50%; transform:translateX(-50%); z-index:50; display:flex; align-items:center; gap:10px; background: rgba(30,18,50,0.6); backdrop-filter: blur(14px); border:1px solid rgba(255,255,255,0.25); border-radius:999px; padding:10px 14px; box-shadow:0 10px 30px rgba(0,0,0,0.4); }
#bgSwitcher button{ background:none; border:none; color:#fff; cursor:pointer; width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; transition: background .2s ease; }
#bgSwitcher button:hover{ background: rgba(255,255,255,0.18); }
.bg-dots{ display:flex; gap:6px; }
.bg-dot{ width:9px; height:9px; border-radius:50%; background: rgba(255,255,255,0.35); cursor:pointer; transition: all .2s ease; }
.bg-dot.active{ background:#a78bfa; transform:scale(1.3); }
@media (max-width: 768px){
  .app-header h1{ font-size:1.7rem; }
  .glass-panel{
  overflow:visible;
  overflow:visible; padding:22px; border-radius:26px; }
  #shaderPanel{ width:100vw; border-radius:24px 24px 0 0; }
  #shaderToggleBtn{ bottom:96px; right:16px; padding:12px 18px; font-size:0.8rem; }
  #bgSwitcher{ bottom:16px; padding:8px 10px; }
  .stat-tile .val{ font-size:1.25rem; }
  .container{ padding:24px 14px 180px; }
}
</style>
</head>
<body>

<div class="container">
  <header class="app-header">
    <h1><i class="fa-solid fa-chart-line"></i> Universal CSV Analyzer</h1>
    <p>Upload any CSV, pick a target, and train a Classification or Regression model in seconds.</p>
  </header>

  <section class="glass-panel section-upload">
    <div class="section-title"><i class="fa-solid fa-cloud-arrow-up"></i> Upload Dataset</div>
    <div id="dropzone" class="dropzone">
      <i class="fa-solid fa-file-csv"></i>
      <h3>Drag &amp; drop your CSV here</h3>
      <p>or click to browse files &middot; max 64MB</p>
      <input type="file" id="fileInput" accept=".csv" />
    </div>
    <div id="fileChipWrap"></div>
    
    <div style="margin-top:60px;text-align:center;border-top:1px dashed rgba(255,255,255,0.2);padding-top:20px;padding-bottom:20px;">
      <div style="font-size:0.85rem;opacity:0.8;margin-bottom:10px;">Or try a sample dataset:</div>
      <div style="position:relative;display:inline-block;">
        <button class="sample-dropdown-btn" onclick="toggleSampleDropdown()" style="padding:10px 20px;font-size:0.85rem;border-radius:999px;display:flex;align-items:center;gap:8px;cursor:pointer;border:1px solid rgba(255,255,255,0.4);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);box-shadow:inset 0 1px 0 rgba(255,255,255,0.6),inset 0 -1px 0 rgba(0,0,0,0.3),0 4px 12px rgba(0,0,0,0.2);color:#fff;font-weight:700;text-shadow:0 1px 2px rgba(0,0,0,0.5);">
          <i class="fa-solid fa-flask"></i> Load Sample Dataset
          <i class="fa-solid fa-chevron-down" style="font-size:0.7rem;"></i>
        </button>
        <div id="sampleDropdown" style="position:absolute;bottom:100%;left:50%;transform:translateX(-50%);margin-bottom:8px;background:rgba(24,14,42,0.95);border:1px solid rgba(255,255,255,0.25);border-radius:16px;padding:8px;display:none;z-index:99999;min-width:260px;box-shadow:0 10px 30px rgba(0,0,0,0.4);backdrop-filter:blur(10px);">
          <div style="font-size:0.7rem;color:rgba(255,255,255,0.6);padding:6px 10px;text-transform:uppercase;letter-spacing:0.05em;">Available Datasets</div>
          <div id="sampleList"></div>
        </div>
      </div>
    </div>
  </section>

  <section id="summarySection" class="glass-panel section-summary hidden">
    <div class="section-title"><i class="fa-solid fa-table-list"></i> Dataset Summary</div>
    <div class="stat-grid" id="statGrid"></div>
    <div class="table-wrap">
      <table id="columnsTable">
        <thead><tr><th class="col-name">Column</th><th class="col-type">Type</th><th class="col-unique">Unique</th><th class="col-missing">Missing</th><th class="col-task">Suggested Task</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <section id="previewSection" class="glass-panel section-preview hidden">
    <div class="section-title"><i class="fa-solid fa-eye"></i> Data Preview <span style="opacity:.6;font-weight:400;font-size:.85rem;">(first 10 rows)</span></div>
    <div class="table-wrap">
      <table id="previewTable"><thead></thead><tbody></tbody></table>
    </div>
  </section>

  <section id="configSection" class="glass-panel section-config hidden">
    <div class="section-title"><i class="fa-solid fa-sliders"></i> Configure Model</div>
    <div class="form-grid">
      <div class="field">
        <label style="color:#ef4444;font-weight:700;">Target Column</label>
        <select id="targetSelect" class="target-select-chrome"></select>
      </div>
      <div class="field">
        <label style="color:#3b82f6;font-weight:700;">Detected Task</label>
        <div><span id="taskBadge" class="task-badge classification"><i class="fa-solid fa-wand-magic-sparkles"></i> —</span></div>
      </div>
      <div class="field">
        <label style="color:#ef4444;font-weight:700;">Model</label>
        <select id="modelSelect" class="model-select-chrome"></select>
      </div>
      <div class="field">
        <label style="color:#3b82f6;font-weight:700;">Test Size <span id="testSizeVal" style="opacity:.6;font-weight:400;">(20%)</span></label>
        <input type="range" id="testSizeSlider" min="10" max="50" value="20" step="5" />
      </div>
    </div>
    <div class="form-grid">
      <div class="field hyper-row" id="row-max-depth">
        <label>Max Depth <span style="opacity:.6;font-weight:400;">(0 = unlimited)</span></label>
        <input type="number" id="maxDepth" min="0" max="100" value="8" />
      </div>
      <div class="field hyper-row" id="row-c">
        <label>C (Regularization)</label>
        <input type="number" id="paramC" min="0.01" max="100" step="0.1" value="1.0" />
      </div>
      <div class="field hyper-row" id="row-max-iter">
        <label>Max Iterations</label>
        <input type="number" id="maxIter" min="50" max="5000" step="50" value="200" />
      </div>
    </div>
    <button class="btn" id="trainBtn">
      <i class="fa-solid fa-bolt"></i> <span id="trainBtnLabel">Train Model</span>
    </button>
  </section>

  <section id="resultsSection" class="glass-panel section-results hidden">
    <div class="section-title"><i class="fa-solid fa-flask-vial"></i> Results</div>
    <div class="metric-grid" id="metricGrid"></div>
    <div class="chart-grid" id="chartGrid"></div>
  </section>
</div>

<button id="shaderToggleBtn"><i class="fa-solid fa-palette"></i> Shader Panel</button>
<div id="shaderPanel">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;">
    <h3 style="margin:0;"><i class="fa-solid fa-wand-magic-sparkles"></i> Liquid Metal Shader</h3>
    <button onclick="document.getElementById('shaderPanel').classList.remove('open')" style="background:rgba(255,255,255,0.1);border:none;color:#fff;border-radius:50%;width:32px;height:32px;cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;"><i class="fa-solid fa-xmark"></i></button>
  </div>

  <div class="shader-group">
    <label>Blur <span id="v-blur">0px</span></label>
    <input type="range" id="s-blur" min="0" max="60" value="0" step="1">
  </div>
  <div class="shader-group">
    <label>Frostiness <span id="v-frost">0%</span></label>
    <input type="range" id="s-frost" min="0" max="60" value="0" step="1">
  </div>
  <div class="shader-group">
    <label>Saturation <span id="v-sat">1.0</span></label>
    <input type="range" id="s-sat" min="0.5" max="4.0" value="1.0" step="0.1">
  </div>
  <div class="shader-group">
    <label>Tint Alpha <span id="v-tint">50%</span></label>
    <input type="range" id="s-tint" min="0" max="60" value="50" step="1">
    <div class="shader-color-row" style="margin-top:8px;">
      <input type="color" id="s-tint-color" value="#332150">
      <span style="font-size:0.8rem;opacity:0.75;">Tint color</span>
    </div>
  </div>
  <div class="shader-group">
    <label>Bevel <span id="v-bevel">6</span></label>
    <input type="range" id="s-bevel" min="0" max="6" value="6" step="1">
  </div>
  <div class="shader-group">
    <label>Specular <span id="v-spec">100%</span></label>
    <input type="range" id="s-spec" min="0" max="100" value="100" step="1">
  </div>
  <div class="shader-group">
    <label>3D Depth <span id="v-depth">80px</span></label>
    <input type="range" id="s-depth" min="0" max="80" value="80" step="1">
  </div>
  <div class="shader-group">
    <label>Radius <span id="v-radius">40px</span></label>
    <input type="range" id="s-radius" min="0" max="40" value="40" step="1">
  </div>
  <div class="shader-group">
    <label>Contrast <span id="v-contrast">1.60</span></label>
    <input type="range" id="s-contrast" min="0.5" max="2.0" value="1.60" step="0.05">
  </div>
  <div class="shader-group">
    <label>Brightness <span id="v-bright">1.40</span></label>
    <input type="range" id="s-bright" min="0.5" max="2.0" value="1.40" step="0.05">
  </div>
  <div class="shader-group">
    <label>Metal Opacity <span id="v-metal">60%</span></label>
    <input type="range" id="s-metal" min="0" max="100" value="60" step="1">
  </div>

  <div class="shader-group">
    <label>Text Color</label>
    <div class="color-presets">
      <div class="color-swatch" style="background:#ffffff" data-color="#ffffff" title="White"></div>
      <div class="color-swatch" style="background:#000000" data-color="#000000" title="Black"></div>
      <div class="color-swatch" style="background:#3b82f6" data-color="#3b82f6" title="Blue"></div>
      <div class="color-swatch" style="background:#ef4444" data-color="#ef4444" title="Red"></div>
      <div class="color-swatch" style="background:#eab308" data-color="#eab308" title="Yellow"></div>
      <div class="color-swatch" style="background:#22c55e" data-color="#22c55e" title="Green"></div>
    </div>
    <div class="shader-color-row" style="margin-top:10px;">
      <input type="color" id="s-text-color" value="#ffffff">
      <span style="font-size:0.8rem;opacity:0.75;">Custom color</span>
    </div>
  </div>

  <button class="btn secondary shader-reset" id="shaderResetBtn"><i class="fa-solid fa-rotate-left"></i> Reset to Defaults</button>
</div>

<div id="bgSwitcher">
  <button id="bgPrev"><i class="fa-solid fa-chevron-left"></i></button>
  <div class="bg-dots" id="bgDots"></div>
  <button id="bgNext"><i class="fa-solid fa-chevron-right"></i></button>
</div>

<script>
let datasetInfo = null;
const CLASSIFICATION_MODELS = [
  { value: "random_forest", label: "Random Forest" },
  { value: "logistic_regression", label: "Logistic Regression" },
  { value: "svm", label: "SVM (SVC)" },
];
const REGRESSION_MODELS = [
  { value: "random_forest", label: "Random Forest" },
  { value: "linear_regression", label: "Linear Regression" },
  { value: "svm", label: "SVM (SVR)" },
];

function showToast(message, duration = 12000){
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span>${message}</span>`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
dropzone.addEventListener('click', () => fileInput.click());
['dragenter','dragover'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); dropzone.classList.add('dragover'); });
});
['dragleave','drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('dragover'); });
});
dropzone.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if (files.length) handleFile(files[0]);
});
fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file){
  if (!file.name.toLowerCase().endsWith('.csv')){
    showToast('Please upload a .csv file.');
    return;
  }
  const chipWrap = document.getElementById('fileChipWrap');
  chipWrap.innerHTML = `<div class="file-chip"><i class="fa-solid fa-spinner fa-spin"></i> Uploading ${file.name}...</div>`;
  const formData = new FormData();
  formData.append('file', file);
  fetch('/upload', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (!data.success){
        showToast(data.error || 'Upload failed.');
        chipWrap.innerHTML = '';
        return;
      }
      datasetInfo = data;
      chipWrap.innerHTML = `<div class="file-chip"><i class="fa-solid fa-circle-check" style="color:#4ade80;"></i> ${data.filename} &middot; ${data.n_rows} rows &middot; ${data.n_cols} cols</div>`;
      renderSummary(data);
      renderPreview(data);
      populateTargetSelect(data);
      show('summarySection'); show('previewSection'); show('configSection');
      hide('resultsSection');
    })
    .catch(err => {
      showToast('Upload failed: ' + err);
      chipWrap.innerHTML = '';
    });
}

function show(id){ document.getElementById(id).classList.remove('hidden'); }
function hide(id){ document.getElementById(id).classList.add('hidden'); }

function toggleSampleDropdown(){
  const dropdown = document.getElementById('sampleDropdown');
  if (dropdown.style.display === 'none' || dropdown.style.display === '') {
    dropdown.style.display = 'block';
    // Load samples dynamically when opening
    fetch('/api/samples')
      .then(r => r.json())
      .then(data => {
        const sampleList = document.getElementById('sampleList');
        if (data.success && data.samples.length > 0) {
          sampleList.innerHTML = data.samples.map(sample => {
            const displayName = sample.replace('.csv', '').replace(/_/g, ' ').toUpperCase();
            return `<div onclick="loadSample('${sample}')" style="padding:10px 12px;border-radius:10px;cursor:pointer;font-size:0.85rem;display:flex;align-items:center;gap:8px;transition:background 0.2s;text-transform:capitalize;" onmouseover="this.style.background='rgba(255,255,255,0.1)'" onmouseout="this.style.background='transparent'">
              <i class="fa-solid fa-database" style="color:#f4a460;"></i> ${displayName}
            </div>`;
          }).join('');
        } else {
          sampleList.innerHTML = '<div style="padding:10px;font-size:0.8rem;color:rgba(255,255,255,0.5);">No sample datasets found.</div>';
        }
      })
      .catch(err => {
        document.getElementById('sampleList').innerHTML = '<div style="padding:10px;font-size:0.8rem;color:rgba(255,255,255,0.5);">Error loading samples.</div>';
      });
  } else {
    dropdown.style.display = 'none';
  }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
  const dropdown = document.getElementById('sampleDropdown');
  const btn = event.target.closest('button');
  if (dropdown && dropdown.style.display === 'block' && !btn) {
    dropdown.style.display = 'none';
  }
});

async function loadSample(sampleName){
  const chipWrap = document.getElementById('fileChipWrap');
  chipWrap.innerHTML = `<div class="file-chip"><i class="fa-solid fa-spinner fa-spin"></i> Loading ${sampleName} sample...</div>`;
  
  try {
    const response = await fetch('/load_sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_name: sampleName })
    });
    const data = await response.json();
    
    if (!data.success){
      showToast(data.error || 'Could not load sample.');
      chipWrap.innerHTML = '';
      return;
    }
    
    datasetInfo = data;
    chipWrap.innerHTML = `<div class="file-chip"><i class="fa-solid fa-circle-check" style="color:#4ade80;"></i> ${data.filename} &middot; ${data.n_rows} rows &middot; ${data.n_cols} cols</div>`;
    renderSummary(data);
    renderPreview(data);
    populateTargetSelect(data);
    show('summarySection'); show('previewSection'); show('configSection');
    hide('resultsSection');
    
    // Auto-select a good target column
    const targetSelect = document.getElementById('targetSelect');
    const goodTarget = data.columns.find(c => c.suggested_task === 'classification' && c.name !== 'type') || data.columns.find(c => c.suggested_task === 'classification');
    if (goodTarget) {
      targetSelect.value = goodTarget.name;
      updateTaskAndModels(data);
    }
    
    showToast('✅ Sample loaded! Select a target column and train.', 5000);
  } catch (error) {
    showToast('Could not load sample: ' + error);
    chipWrap.innerHTML = '';
  }
}

function renderSummary(data){
  const statGrid = document.getElementById('statGrid');
  statGrid.innerHTML = `
    <div class="stat-tile silver">
      <div class="val">${data.n_rows}</div>
      <div class="lbl">Rows</div>
    </div>
    <div class="stat-tile gold">
      <div class="val">${data.n_cols}</div>
      <div class="lbl">Columns</div>
    </div>
    <div class="stat-tile bronze">
      <div class="val">${data.total_missing}</div>
      <div class="lbl">Missing Values</div>
    </div>
  `;
  const tbody = document.querySelector('#columnsTable tbody');
  tbody.innerHTML = data.columns.map(col => `
    <tr>
      <td><strong>${col.name}</strong></td>
      <td>${col.dtype}</td>
      <td style="color:#ef4444;font-weight:600">${col.unique}</td>
      <td><span class="badge ${col.missing > 0 ? 'missing-some' : 'missing-none'}">${col.missing}</span></td>
      <td>${col.suggested_task === 'classification' ? 'Classification' : 'Regression'}</td>
    </tr>
  `).join('');
}

function renderPreview(data){
  const thead = document.querySelector('#previewTable thead');
  const tbody = document.querySelector('#previewTable tbody');
  thead.innerHTML = '<tr>' + data.preview_columns.map(c => `<th style="color:#eab308;font-weight:700;text-transform:uppercase;letter-spacing:0.03em;">${c}</th>`).join('') + '</tr>';
  tbody.innerHTML = data.preview_rows.map((row, idx) => {
    const rowColor = idx % 3 === 0 ? '#fbbf24' : (idx % 3 === 2 ? '#f472b6' : 'var(--white-text)');
    return '<tr style="color:' + rowColor + ';">' + data.preview_columns.map(c => `<td>${row[c]}</td>`).join('') + '</tr>';
  }).join('');
}

function populateTargetSelect(data){
  const sel = document.getElementById('targetSelect');
  sel.innerHTML = data.columns.map(c => `<option value="${c.name}">${c.name}</option>`).join('');
  sel.onchange = () => updateTaskAndModels(data);
  updateTaskAndModels(data);
}

function updateTaskAndModels(data){
  const targetName = document.getElementById('targetSelect').value;
  const col = data.columns.find(c => c.name === targetName);
  const task = col ? col.suggested_task : 'classification';
  const badge = document.getElementById('taskBadge');
  badge.className = 'task-badge ' + task;
  badge.innerHTML = `<i class="fa-solid ${task === 'classification' ? 'fa-shapes' : 'fa-chart-simple'}"></i> ${task === 'classification' ? 'Classification' : 'Regression'}`;
  const modelSel = document.getElementById('modelSelect');
  const models = task === 'classification' ? CLASSIFICATION_MODELS : REGRESSION_MODELS;
  modelSel.innerHTML = models.map(m => `<option value="${m.value}">${m.label}</option>`).join('');
  modelSel.onchange = updateHyperparamVisibility;
  updateHyperparamVisibility();
}

function updateHyperparamVisibility(){
  const model = document.getElementById('modelSelect').value;
  const showDepth = model === 'random_forest';
  const showC = model === 'logistic_regression' || model === 'svm';
  const showIter = model === 'logistic_regression';
  document.getElementById('row-max-depth').classList.toggle('active', showDepth);
  document.getElementById('row-c').classList.toggle('active', showC);
  document.getElementById('row-max-iter').classList.toggle('active', showIter);
}

document.getElementById('testSizeSlider').addEventListener('input', (e) => {
  document.getElementById('testSizeVal').textContent = `(${e.target.value}%)`;
});

document.getElementById('trainBtn').addEventListener('click', () => {
  if (!datasetInfo){ showToast('Please upload a CSV first.'); return; }
  const target = document.getElementById('targetSelect').value;
  const model = document.getElementById('modelSelect').value;
  const testSize = parseFloat(document.getElementById('testSizeSlider').value) / 100;
  const maxDepth = document.getElementById('maxDepth').value;
  const C = document.getElementById('paramC').value;
  const maxIter = document.getElementById('maxIter').value;
  const btn = document.getElementById('trainBtn');
  const label = document.getElementById('trainBtnLabel');
  btn.disabled = true;
  label.textContent = 'Training...';
  btn.querySelector('i').className = '';
  const spinner = document.createElement('span');
  spinner.className = 'spinner';
  btn.insertBefore(spinner, label);
  fetch('/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target, model, test_size: testSize, max_depth: maxDepth, C: C, max_iter: maxIter })
  })
  .then(r => r.json())
  .then(data => {
    resetTrainButton();
    if (!data.success){ showToast(data.error || 'Training failed.'); return; }
    renderResults(data);
    show('resultsSection');
    document.getElementById('resultsSection').scrollIntoView({ behavior:'smooth', block:'start' });
  })
  .catch(err => { resetTrainButton(); showToast('⚠️ Training took too long. Try selecting other target columns.', 18000); });
});

function resetTrainButton(){
  const btn = document.getElementById('trainBtn');
  const label = document.getElementById('trainBtnLabel');
  const spinner = btn.querySelector('.spinner');
  if (spinner) spinner.remove();
  btn.disabled = false;
  label.textContent = 'Train Model';
  const icon = document.createElement('i');
  icon.className = 'fa-solid fa-bolt';
  btn.insertBefore(icon, label);
}

const METRIC_LABELS = {
  accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1_score: 'F1 Score', roc_auc: 'ROC AUC',
  r2_score: 'R² Score', mse: 'MSE', rmse: 'RMSE', mae: 'MAE'
};
const CHART_LABELS = {
  confusion_matrix: 'Confusion Matrix', roc_curve: 'ROC Curve',
  scatter: 'Actual vs Predicted', feature_importance: 'Feature Importance'
};

function renderResults(data){
  const metricGrid = document.getElementById('metricGrid');
  metricGrid.innerHTML = Object.entries(data.metrics).map(([k, v]) => `
    <div class="metric-card">
      <div class="m-val">${v}</div>
      <div class="m-lbl">${METRIC_LABELS[k] || k}</div>
    </div>
  `).join('') + `
    <div class="metric-card">
      <div class="m-val">${data.rows_used}</div>
      <div class="m-lbl">Rows Used</div>
    </div>
    <div class="metric-card">
      <div class="m-val">${data.training_time_seconds}s</div>
      <div class="m-lbl">Training Time</div>
    </div>
  `;
  const chartGrid = document.getElementById('chartGrid');
  chartGrid.innerHTML = Object.entries(data.charts).filter(([k, v]) => v).map(([k, b64]) => `
    <div class="chart-card">
      <h4>${CHART_LABELS[k] || k}</h4>
      <img src="data:image/png;base64,${b64}" alt="${k}" />
    </div>
  `).join('');
}

const root = document.documentElement;
const shaderToggleBtn = document.getElementById('shaderToggleBtn');
const shaderPanel = document.getElementById('shaderPanel');
shaderToggleBtn.addEventListener('click', () => shaderPanel.classList.toggle('open'));

let tintColor = { r: 51, g: 33, b: 80 };
let tintAlpha = 0.50;
function hexToRgb(hex){
  const bigint = parseInt(hex.replace('#',''), 16);
  return { r: (bigint >> 16) & 255, g: (bigint >> 8) & 255, b: bigint & 255 };
}
function applyTint(){
  const rgba = `rgba(${tintColor.r},${tintColor.g},${tintColor.b},${tintAlpha.toFixed(2)})`;
  root.style.setProperty('--tint', rgba);
  root.style.setProperty('--glass-bg', rgba);
}

const sBlur = document.getElementById('s-blur');
sBlur.addEventListener('input', () => { root.style.setProperty('--blur', sBlur.value + 'px'); document.getElementById('v-blur').textContent = sBlur.value + 'px'; });
const sFrost = document.getElementById('s-frost');
sFrost.addEventListener('input', () => { const pct = parseFloat(sFrost.value); root.style.setProperty('--frost', pct); root.style.setProperty('--frost-alpha', (pct / 100).toFixed(2)); document.getElementById('v-frost').textContent = pct + '%'; });
const sSat = document.getElementById('s-sat');
sSat.addEventListener('input', () => { root.style.setProperty('--saturation', sSat.value); document.getElementById('v-sat').textContent = parseFloat(sSat.value).toFixed(1); });
const sTint = document.getElementById('s-tint');
const sTintColor = document.getElementById('s-tint-color');
sTint.addEventListener('input', () => { tintAlpha = parseFloat(sTint.value) / 100; document.getElementById('v-tint').textContent = sTint.value + '%'; applyTint(); });
sTintColor.addEventListener('input', () => { tintColor = hexToRgb(sTintColor.value); applyTint(); });
const sBevel = document.getElementById('s-bevel');
sBevel.addEventListener('input', () => { root.style.setProperty('--bevel', sBevel.value + 'px'); document.getElementById('v-bevel').textContent = sBevel.value; });
const sSpec = document.getElementById('s-spec');
sSpec.addEventListener('input', () => { root.style.setProperty('--specular', (parseFloat(sSpec.value) / 100).toFixed(2)); document.getElementById('v-spec').textContent = sSpec.value + '%'; });
const sDepth = document.getElementById('s-depth');
sDepth.addEventListener('input', () => { root.style.setProperty('--depth', sDepth.value + 'px'); document.getElementById('v-depth').textContent = sDepth.value + 'px'; });
const sRadius = document.getElementById('s-radius');
sRadius.addEventListener('input', () => { root.style.setProperty('--radius', sRadius.value + 'px'); document.getElementById('v-radius').textContent = sRadius.value + 'px'; });
const sContrast = document.getElementById('s-contrast');
sContrast.addEventListener('input', () => { root.style.setProperty('--contrast', sContrast.value); document.getElementById('v-contrast').textContent = parseFloat(sContrast.value).toFixed(2); });
const sBright = document.getElementById('s-bright');
sBright.addEventListener('input', () => { root.style.setProperty('--brightness', sBright.value); document.getElementById('v-bright').textContent = parseFloat(sBright.value).toFixed(2); });
document.querySelectorAll('.color-swatch').forEach(sw => { sw.addEventListener('click', () => { const c = sw.dataset.color; root.style.setProperty('--white-text', c); document.getElementById('s-text-color').value = c; }); });
document.getElementById('s-text-color').addEventListener('input', (e) => { root.style.setProperty('--white-text', e.target.value); });

const sMetal = document.getElementById('s-metal');
sMetal.addEventListener('input', () => { const val = parseFloat(sMetal.value); root.style.setProperty('--metal-opacity', val / 100); document.getElementById('v-metal').textContent = val + '%'; });

const DEFAULTS = {
  blur: 0, frost: 0, sat: 1.0, tintAlpha: 50, tintColor: '#332150',
  bevel: 6, spec: 100, depth: 80, radius: 40, contrast: 1.60, bright: 1.40,
  text: '#ffffff', metal: 60
};
document.getElementById('shaderResetBtn').addEventListener('click', () => {
  sBlur.value = DEFAULTS.blur; sBlur.dispatchEvent(new Event('input'));
  sFrost.value = DEFAULTS.frost; sFrost.dispatchEvent(new Event('input'));
  sSat.value = DEFAULTS.sat; sSat.dispatchEvent(new Event('input'));
  sTint.value = DEFAULTS.tintAlpha; sTint.dispatchEvent(new Event('input'));
  sTintColor.value = DEFAULTS.tintColor; sTintColor.dispatchEvent(new Event('input'));
  sBevel.value = DEFAULTS.bevel; sBevel.dispatchEvent(new Event('input'));
  sSpec.value = DEFAULTS.spec; sSpec.dispatchEvent(new Event('input'));
  sDepth.value = DEFAULTS.depth; sDepth.dispatchEvent(new Event('input'));
  sRadius.value = DEFAULTS.radius; sRadius.dispatchEvent(new Event('input'));
  sContrast.value = DEFAULTS.contrast; sContrast.dispatchEvent(new Event('input'));
  sBright.value = DEFAULTS.bright; sBright.dispatchEvent(new Event('input'));
  sMetal.value = DEFAULTS.metal; sMetal.dispatchEvent(new Event('input'));
  document.getElementById('s-text-color').value = DEFAULTS.text;
  root.style.setProperty('--white-text', DEFAULTS.text);
});

root.style.setProperty('--frost-alpha', '0');
applyTint();
root.style.setProperty('--metal-opacity', '0.60');
document.body.style.display = 'none';
document.body.offsetHeight;
document.body.style.display = '';

let bgIndex = 0;
const bgTotal = 7;
const bgDotsWrap = document.getElementById('bgDots');
for (let i = 0; i < bgTotal; i++){
  const dot = document.createElement('div');
  dot.className = 'bg-dot' + (i === 0 ? ' active' : '');
  dot.addEventListener('click', () => setBg(i));
  bgDotsWrap.appendChild(dot);
}
function setBg(i){
  bgIndex = ((i % bgTotal) + bgTotal) % bgTotal;
  root.style.setProperty('--bg-image', `url('/static/images/bg${bgIndex + 1}.jpg')`);
  document.querySelectorAll('.bg-dot').forEach((d, idx) => d.classList.toggle('active', idx === bgIndex));
}
document.getElementById('bgPrev').addEventListener('click', () => setBg(bgIndex - 1));
document.getElementById('bgNext').addEventListener('click', () => setBg(bgIndex + 1));
</script>
</body>
</html>
"""

if __name__ == "__main__":
    ensure_background_images()
    port = int(os.environ.get("PORT", 5001))
    print(f"🌐 Server starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
