#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 UNIVERSAL CSV ANALYZER
================================================================================
A single-file, production-ready Flask web application that lets a user:

  1. Upload any CSV file (drag & drop or file picker)
  2. Automatically profile it (columns, dtypes, row count, missing values)
  3. Preview the first 10 rows in a styled HTML table
  4. Pick a target column -> the app auto-detects Classification vs Regression
  5. Pick a model (RandomForest / LogisticRegression / SVM for classification,
     RandomForest / LinearRegression / SVR for regression) and tune a couple
     of hyperparameters
  6. Train the model server-side with scikit-learn and get back metrics +
     Matplotlib charts (base64-encoded PNGs) rendered directly in the page

The front-end is a premium "Liquid Glass" glassmorphism UI with a live
"Shader Panel" that lets the user tweak blur, frost, saturation, tint,
bevel, specular, 3D depth, corner radius, contrast, brightness and text
color in real time via CSS custom properties, plus a background image
switcher that cycles through 7 procedurally generated background images.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
    pip install flask pandas numpy scikit-learn matplotlib
    python app.py

Then open http://127.0.0.1:5000 in your browser.

Everything (backend routes + HTML + CSS + JavaScript) lives in this ONE
file, as required. Python 3.11+ compatible.
================================================================================
"""

# ------------------------------------------------------------------------
# STANDARD LIBRARY IMPORTS
# ------------------------------------------------------------------------
import os
import io
import base64
import traceback

# ------------------------------------------------------------------------
# THIRD-PARTY IMPORTS
# ------------------------------------------------------------------------
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify

# Matplotlib must use a non-interactive backend since we run headless
# on a web server (no display available).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

# ------------------------------------------------------------------------
# FLASK APP SETUP
# ------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB upload cap

# ------------------------------------------------------------------------
# GLOBAL IN-MEMORY STATE
# ------------------------------------------------------------------------
# For a small single-user demo app we keep the currently uploaded dataframe
# in a simple module-level dict rather than wiring up a database or a
# session/token system. This is intentionally simple; for a multi-user
# production deployment you'd key this by session id instead.
APP_STATE = {
    "df": None,          # the raw (uncleaned) uploaded dataframe
    "filename": None,    # original filename
}

# Threshold used to decide classification vs regression for a numeric
# target column: if the column has few unique values relative to its
# length, we treat it as categorical (classification) even though it's
# numeric (e.g. a 0/1 flag, or a 1-5 star rating).
CLASSIFICATION_UNIQUE_THRESHOLD = 15
CLASSIFICATION_UNIQUE_RATIO = 0.05


# ==========================================================================
# BACKGROUND IMAGE GENERATION
# ==========================================================================
def ensure_background_images():
    """
    Procedurally generate 7 gradient background JPGs (bg1.jpg .. bg7.jpg)
    into /static/images/ the first time the app runs, so the "Liquid
    Glass" UI has real images to blur/tint against without requiring the
    user to supply their own artwork. If the files already exist, they
    are left untouched.
    """
    img_dir = os.path.join(BASE_DIR, "static", "images")
    os.makedirs(img_dir, exist_ok=True)

    # Seven distinct two-color gradients, each tuned to look good behind
    # translucent glass panels (deep, moody, high-contrast colors).
    palettes = [
        ("#1a0b2e", "#7209b7"),  # 1: violet nebula
        ("#0f2027", "#2c5364"),  # 2: deep teal
        ("#2b0000", "#8b0000"),  # 3: crimson depths
        ("#0b132b", "#3a506b"),  # 4: midnight navy
        ("#1b1b2f", "#e94560"),  # 5: dark pink pop
        ("#03071e", "#6a040f"),  # 6: obsidian ember
        ("#022c43", "#0a9396"),  # 7: ocean current
    ]

    for i, (c1, c2) in enumerate(palettes, start=1):
        path = os.path.join(img_dir, f"bg{i}.jpg")
        if os.path.exists(path):
            continue
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        cmap = LinearSegmentedColormap.from_list("bg", [c1, c2, c1])
        # Diagonal-ish gradient: combine an x gradient and a subtle y one.
        x = np.linspace(0, 1, 512)
        y = np.linspace(0, 1, 288)
        xx, yy = np.meshgrid(x, y)
        gradient = (xx * 0.75 + yy * 0.25)
        ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])

        fig.savefig(path, format="jpg")
        plt.close(fig)


# ==========================================================================
# DATA HELPERS
# ==========================================================================
def detect_task_type(series: pd.Series) -> str:
    """
    Heuristically decide whether a target column implies a Classification
    or Regression problem:
      - Non-numeric columns are always Classification.
      - Numeric columns are Classification if they have relatively few
        unique values (few distinct classes / integer-coded labels),
        otherwise Regression.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"

    n = len(series)
    nunique = series.nunique(dropna=True)
    if n == 0:
        return "classification"

    if nunique <= CLASSIFICATION_UNIQUE_THRESHOLD or (nunique / n) < CLASSIFICATION_UNIQUE_RATIO:
        return "classification"
    return "regression"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - numeric columns -> median
      - categorical/object columns -> mode (most frequent value)
    Returns a NEW dataframe (does not mutate the original in place).
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_series = df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
    return df


def encode_categoricals(df: pd.DataFrame):
    """
    Label-encode every non-numeric column in place (on a copy).
    Returns (encoded_df, {col_name: fitted_LabelEncoder}).
    """
    df = df.copy()
    encoders = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def fig_to_base64(fig) -> str:
    """Render a Matplotlib figure to a base64-encoded PNG data string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def style_dark_axes(ax):
    """Apply a consistent 'glass UI friendly' dark/transparent chart style."""
    ax.set_facecolor("none")
    ax.tick_params(colors="#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    ax.yaxis.label.set_color("#ffffff")
    ax.title.set_color("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#ffffff88")
    ax.grid(True, alpha=0.15, color="#ffffff")


# ==========================================================================
# CHART BUILDERS
# ==========================================================================
def build_confusion_matrix_chart(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    fig.patch.set_alpha(0)
    im = ax.imshow(cm, cmap="magma")
    style_dark_axes(ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] < thresh else "black",
                fontsize=10, fontweight="bold",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#ffffff")
    plt.setp(cbar.ax.get_yticklabels(), color="#ffffff")
    fig.tight_layout()
    return fig_to_base64(fig)


def build_roc_chart(y_test, proba, classes):
    """
    Build an ROC curve chart. Handles both binary and multiclass
    (one-vs-rest, per-class curves + macro average) cases.
    Returns (base64_png, macro_auc) or (None, None) if not computable.
    """
    n_classes = len(classes)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    fig.patch.set_alpha(0)
    style_dark_axes(ax)

    try:
        if n_classes == 2:
            # Binary case: proba column 1 is the "positive" class score.
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color="#7ee787", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.3f})")
            macro_auc = roc_auc
        else:
            # Multiclass: binarize labels and plot one curve per class.
            y_bin = label_binarize(y_test, classes=classes)
            aucs = []
            colors = plt.cm.plasma(np.linspace(0.15, 0.9, n_classes))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                ax.plot(fpr, tpr, color=colors[i], lw=1.8,
                        label=f"Class {classes[i]} (AUC={roc_auc:.2f})")
            macro_auc = float(np.mean(aucs))

        ax.plot([0, 1], [0, 1], linestyle="--", color="#ffffff55", lw=1.2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        legend = ax.legend(loc="lower right", fontsize=8, facecolor="#00000055", edgecolor="none")
        for text in legend.get_texts():
            text.set_color("#ffffff")
        fig.tight_layout()
        return fig_to_base64(fig), macro_auc
    except Exception:
        plt.close(fig)
        return None, None


def build_feature_importance_chart(feature_names, importances):
    order = np.argsort(importances)[::-1][:15]  # top 15 features
    names = [feature_names[i] for i in order]
    vals = [importances[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, max(3.2, 0.35 * len(names))))
    fig.patch.set_alpha(0)
    style_dark_axes(ax)
    ax.barh(range(len(names))[::-1], vals, color="#a78bfa")
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig_to_base64(fig)


def build_regression_scatter_chart(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    fig.patch.set_alpha(0)
    style_dark_axes(ax)
    ax.scatter(y_true, y_pred, alpha=0.65, color="#60a5fa", edgecolor="#ffffff33", s=40)

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#f87171", lw=2, label="Ideal fit")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    legend = ax.legend(loc="upper left", fontsize=9, facecolor="#00000055", edgecolor="none")
    for text in legend.get_texts():
        text.set_color("#ffffff")
    fig.tight_layout()
    return fig_to_base64(fig)


# ==========================================================================
# MODEL FACTORY
# ==========================================================================
def build_model(task_type: str, model_name: str, hyperparams: dict):
    """
    Instantiate the requested scikit-learn estimator.

    hyperparams may contain: max_depth (int|None), C (float), max_iter (int)
    Unused hyperparameters for a given model are simply ignored.
    """
    max_depth = hyperparams.get("max_depth")
    if max_depth in ("", None, 0):
        max_depth = None
    else:
        max_depth = int(max_depth)

    C = float(hyperparams.get("C") or 1.0)
    max_iter = int(hyperparams.get("max_iter") or 200)

    if task_type == "classification":
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=250, max_depth=max_depth, random_state=42, n_jobs=-1
            )
        elif model_name == "logistic_regression":
            return LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        elif model_name == "svm":
            return SVC(C=C, probability=True, kernel="rbf", random_state=42)
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
    else:  # regression
        if model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=250, max_depth=max_depth, random_state=42, n_jobs=-1
            )
        elif model_name == "linear_regression":
            return LinearRegression()
        elif model_name == "svm":
            return SVR(C=C, kernel="rbf")
        else:
            raise ValueError(f"Unknown regression model: {model_name}")


# ==========================================================================
# ROUTES
# ==========================================================================
@app.route("/")
def index():
    """Serve the single-page front-end (HTML + CSS + JS all inline)."""
    return HTML_PAGE


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept a CSV file upload, profile it, and stash the raw dataframe in
    memory for the subsequent /train call. Returns column metadata and a
    10-row preview so the front-end can render everything without a
    second round trip.
    """
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file part in the request."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected."}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"success": False, "error": "Please upload a .csv file."}), 400

        raw_bytes = file.read()
        if not raw_bytes:
            return jsonify({"success": False, "error": "The uploaded file is empty."}), 400

        # Try a couple of common encodings before giving up.
        df = None
        last_err = None
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                df = None

        if df is None:
            return jsonify({"success": False, "error": f"Could not parse CSV: {last_err}"}), 400

        if df.shape[1] == 0:
            return jsonify({"success": False, "error": "No columns detected in the CSV."}), 400

        # Drop fully-empty rows/columns which are almost always parsing noise.
        df = df.dropna(axis=1, how="all")

        APP_STATE["df"] = df
        APP_STATE["filename"] = file.filename

        columns_info = []
        for col in df.columns:
            series = df[col]
            is_numeric = bool(pd.api.types.is_numeric_dtype(series))
            columns_info.append({
                "name": str(col),
                "dtype": str(series.dtype),
                "is_numeric": is_numeric,
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "suggested_task": detect_task_type(series),
            })

        preview_df = df.head(10).copy()
        # Convert to plain strings for safe, consistent JSON/table rendering.
        preview_rows = preview_df.astype(object).where(pd.notna(preview_df), "—").astype(str).to_dict(orient="records")

        return jsonify({
            "success": True,
            "filename": file.filename,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "total_missing": int(df.isna().sum().sum()),
            "columns": columns_info,
            "preview_columns": [str(c) for c in df.columns],
            "preview_rows": preview_rows,
        })

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Upload failed: {e}"}), 500


@app.route("/train", methods=["POST"])
def train():
    """
    Train a scikit-learn model on the previously uploaded dataframe and
    return metrics + base64-encoded chart images as JSON.

    Expected JSON body:
    {
        "target": "<column name>",
        "model": "random_forest" | "logistic_regression" | "svm" | "linear_regression",
        "max_depth": <int|null>,
        "C": <float>,
        "max_iter": <int>,
        "test_size": <float, optional, default 0.2>
    }
    """
    try:
        if APP_STATE["df"] is None:
            return jsonify({"success": False, "error": "No dataset uploaded yet. Please upload a CSV first."}), 400

        payload = request.get_json(silent=True) or {}
        target = payload.get("target")
        model_name = payload.get("model")
        test_size = float(payload.get("test_size") or 0.2)
        test_size = min(max(test_size, 0.1), 0.5)

        if not target or target not in APP_STATE["df"].columns:
            return jsonify({"success": False, "error": "Please choose a valid target column."}), 400
        if not model_name:
            return jsonify({"success": False, "error": "Please choose a model."}), 400

        raw_df = APP_STATE["df"]
        task_type = detect_task_type(raw_df[target])

        # 1) Clean missing values (median for numeric, mode for categorical)
        df = clean_dataframe(raw_df)

        # 2) Label-encode every categorical column (including the target,
        #    if it happens to be categorical / classification).
        df_encoded, encoders = encode_categoricals(df)

        # 3) Split features / target
        X = df_encoded.drop(columns=[target])
        y = df_encoded[target]

        if X.shape[1] == 0:
            return jsonify({"success": False, "error": "No feature columns remain after removing the target."}), 400

        # Stratify classification splits when every class has >= 2 samples.
        stratify = None
        if task_type == "classification":
            vc = y.value_counts()
            if (vc >= 2).all() and len(vc) >= 2:
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        hyperparams = {
            "max_depth": payload.get("max_depth"),
            "C": payload.get("C"),
            "max_iter": payload.get("max_iter"),
        }
        model = build_model(task_type, model_name, hyperparams)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        response = {
            "success": True,
            "task_type": task_type,
            "model": model_name,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(X.shape[1]),
            "metrics": {},
            "charts": {},
        }

        # ------------------------------------------------------------
        # CLASSIFICATION METRICS + CHARTS
        # ------------------------------------------------------------
        if task_type == "classification":
            classes = sorted(pd.unique(y))
            class_labels = classes
            # If we label-encoded the target, map back to original names for display.
            if target in encoders:
                le = encoders[target]
                class_labels = [le.inverse_transform([c])[0] for c in classes]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            response["metrics"] = {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
            }

            response["charts"]["confusion_matrix"] = build_confusion_matrix_chart(
                y_test, y_pred, class_labels
            )

            # ROC curve requires predicted probabilities.
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                roc_b64, macro_auc = build_roc_chart(y_test, proba, classes)
                if roc_b64:
                    response["charts"]["roc_curve"] = roc_b64
                    response["metrics"]["roc_auc"] = round(float(macro_auc), 4)

        # ------------------------------------------------------------
        # REGRESSION METRICS + CHARTS
        # ------------------------------------------------------------
        else:
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))

            response["metrics"] = {
                "r2_score": round(float(r2), 4),
                "mse": round(float(mse), 4),
                "rmse": round(rmse, 4),
                "mae": round(float(mae), 4),
            }

            response["charts"]["scatter"] = build_regression_scatter_chart(
                np.asarray(y_test), np.asarray(y_pred)
            )

        # ------------------------------------------------------------
        # FEATURE IMPORTANCE (Random Forest bonus chart)
        # ------------------------------------------------------------
        if hasattr(model, "feature_importances_"):
            response["charts"]["feature_importance"] = build_feature_importance_chart(
                list(X.columns), model.feature_importances_
            )

        return jsonify(response)

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Training failed: {e}"}), 500


# ==========================================================================
# FRONT-END: HTML + CSS + JAVASCRIPT (single-page app, inlined)
# ==========================================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Universal CSV Analyzer</title>

<!-- Google Fonts: Inter -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<!-- Font Awesome 6.4.0 -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
/* ======================================================================
   ROOT THEME VARIABLES ("Liquid Glass" shader controls)
   These are the single source of truth for the glass aesthetic and are
   updated live by the Shader Panel sliders via JavaScript.
   ====================================================================== */
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
}

*{ box-sizing:border-box; margin:0; padding:0; }

html,body{
  min-height:100vh;
  font-family:'Inter',sans-serif;
  color:var(--white-text);
}

body{
  background-image: var(--bg-image);
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
  background-repeat: no-repeat;
  min-height:100vh;
  transition: background-image 0.6s ease;
  position:relative;
  overflow-x:hidden;
}

/* subtle dark overlay so content stays legible over any background */
body::before{
  content:"";
  position:fixed; inset:0;
  background: rgba(10,6,20,0.35);
  z-index:0;
  pointer-events:none;
}

.container{
  position:relative;
  z-index:1;
  max-width:1180px;
  margin:0 auto;
  padding:40px 24px 160px;
}

/* ======================================================================
   GLASS PANEL — the core "Liquid Glass" building block
   ====================================================================== */
.glass-panel{
  background:
    linear-gradient(rgba(255,255,255, var(--frost-alpha, 0)), rgba(255,255,255, var(--frost-alpha, 0))),
    var(--tint);
  border:1px solid var(--glass-border);
  border-radius: var(--radius);
  backdrop-filter: blur(var(--blur)) saturate(var(--saturation)) contrast(var(--contrast)) brightness(var(--brightness));
  -webkit-backdrop-filter: blur(var(--blur)) saturate(var(--saturation)) contrast(var(--contrast)) brightness(var(--brightness));
  box-shadow:
    0 var(--bevel) var(--depth) rgba(0,0,0,0.45),
    inset 0 1px 0 rgba(255,255,255, calc(var(--specular) * 0.55)),
    inset 0 -1px 0 rgba(0,0,0,0.25);
  padding:32px;
  margin-bottom:26px;
  position:relative;
  transition: transform .35s cubic-bezier(.2,.9,.3,1.2), box-shadow .35s ease, border-color .35s ease;
}

.glass-panel:hover{
  transform: translateY(-6px) scale(1.008);
  box-shadow:
    0 calc(var(--bevel) + 6px) calc(var(--depth) + 20px) rgba(0,0,0,0.5),
    0 0 46px rgba(255,255,255,0.16),
    inset 0 1px 0 rgba(255,255,255, calc(var(--specular) * 0.7));
  border-color: rgba(255,255,255,0.28);
}

/* ======================================================================
   HEADER
   ====================================================================== */
.app-header{
  text-align:center;
  margin-bottom:34px;
}
.app-header h1{
  font-size:2.4rem;
  font-weight:800;
  letter-spacing:-0.02em;
  display:flex; align-items:center; justify-content:center; gap:14px;
  text-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.app-header h1 i{
  background: linear-gradient(135deg,#a78bfa,#60a5fa);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  filter: drop-shadow(0 4px 12px rgba(167,139,250,0.5));
}
.app-header p{
  margin-top:10px;
  opacity:0.85;
  font-size:1.02rem;
  font-weight:400;
}

/* ======================================================================
   DROPZONE
   ====================================================================== */
.dropzone{
  border:2px dashed rgba(255,255,255,0.35);
  border-radius: calc(var(--radius) * 0.55);
  padding:46px 20px;
  text-align:center;
  cursor:pointer;
  transition: all .25s ease;
}
.dropzone:hover, .dropzone.dragover{
  border-color:#a78bfa;
  background: rgba(255,255,255,0.06);
}
.dropzone i{
  font-size:2.6rem;
  margin-bottom:14px;
  color:#a78bfa;
}
.dropzone h3{ font-weight:600; margin-bottom:6px; }
.dropzone p{ opacity:0.75; font-size:0.9rem; }
.dropzone input[type=file]{ display:none; }

.file-chip{
  display:inline-flex; align-items:center; gap:10px;
  background: rgba(255,255,255,0.1);
  border:1px solid rgba(255,255,255,0.2);
  padding:8px 16px; border-radius:999px;
  margin-top:16px; font-size:0.9rem;
}

/* ======================================================================
   SECTION TITLES
   ====================================================================== */
.section-title{
  display:flex; align-items:center; gap:10px;
  font-size:1.15rem; font-weight:700;
  margin-bottom:18px;
}
.section-title i{ opacity:0.85; }

/* ======================================================================
   STAT TILES
   ====================================================================== */
.stat-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(140px,1fr));
  gap:14px;
  margin-bottom:22px;
}
.stat-tile{
  background: rgba(255,255,255,0.08);
  border:1px solid rgba(255,255,255,0.14);
  border-radius:18px;
  padding:16px;
  text-align:center;
  transition: transform .2s ease, background .2s ease;
}
.stat-tile:hover{ transform:translateY(-3px); background:rgba(255,255,255,0.14); }
.stat-tile .val{ font-size:1.6rem; font-weight:800; }
.stat-tile .lbl{ font-size:0.78rem; opacity:0.75; text-transform:uppercase; letter-spacing:0.05em; margin-top:4px; }

/* ======================================================================
   TABLES
   ====================================================================== */
.table-wrap{
  overflow-x:auto;
  border-radius:16px;
  border:1px solid rgba(255,255,255,0.15);
  max-height:380px;
  overflow-y:auto;
}
table{ width:100%; border-collapse:collapse; font-size:0.86rem; white-space:nowrap; }
thead th{
  position:sticky; top:0;
  background: rgba(30,18,50,0.85);
  padding:10px 14px; text-align:left;
  font-weight:600; border-bottom:1px solid rgba(255,255,255,0.15);
  backdrop-filter: blur(6px);
}
tbody td{
  padding:9px 14px;
  border-bottom:1px solid rgba(255,255,255,0.08);
}
tbody tr:hover{ background: rgba(255,255,255,0.06); }
.badge{
  display:inline-block; padding:2px 9px; border-radius:999px;
  font-size:0.72rem; font-weight:600; letter-spacing:0.02em;
}
.badge.missing-none{ background:rgba(74,222,128,0.2); color:#86efac; }
.badge.missing-some{ background:rgba(248,113,113,0.2); color:#fca5a5; }

/* ======================================================================
   FORM CONTROLS
   ====================================================================== */
.form-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(210px,1fr));
  gap:18px;
  margin-bottom:20px;
}
.field label{
  display:block; font-size:0.8rem; font-weight:600;
  margin-bottom:8px; opacity:0.85; text-transform:uppercase; letter-spacing:0.04em;
}
select, input[type=number], input[type=text]{
  width:100%;
  background: rgba(255,255,255,0.09);
  border:1px solid rgba(255,255,255,0.22);
  border-radius:12px;
  padding:11px 14px;
  color:var(--white-text);
  font-family:'Inter',sans-serif;
  font-size:0.92rem;
  outline:none;
  transition: border-color .2s ease, background .2s ease;
}
select:focus, input:focus{ border-color:#a78bfa; background:rgba(255,255,255,0.14); }
select option{ background:#1e1233; color:#fff; }

.task-badge{
  display:inline-flex; align-items:center; gap:8px;
  padding:9px 16px; border-radius:999px; font-weight:700; font-size:0.85rem;
  margin-top:6px;
}
.task-badge.classification{ background:rgba(96,165,250,0.22); color:#93c5fd; border:1px solid rgba(96,165,250,0.4); }
.task-badge.regression{ background:rgba(251,191,36,0.22); color:#fde68a; border:1px solid rgba(251,191,36,0.4); }

.hyper-row{ display:none; }
.hyper-row.active{ display:block; }

/* ======================================================================
   BUTTONS
   ====================================================================== */
.btn{
  display:inline-flex; align-items:center; justify-content:center; gap:10px;
  padding:14px 28px;
  border:none; border-radius:999px;
  font-family:'Inter',sans-serif; font-weight:700; font-size:0.95rem;
  cursor:pointer;
  background: linear-gradient(135deg,#7c3aed,#3b82f6);
  color:#fff;
  box-shadow: 0 8px 24px rgba(124,58,237,0.4);
  transition: transform .2s ease, box-shadow .2s ease, opacity .2s ease;
}
.btn:hover:not(:disabled){ transform:translateY(-2px); box-shadow:0 12px 32px rgba(124,58,237,0.55); }
.btn:disabled{ opacity:0.6; cursor:not-allowed; }
.btn.secondary{
  background: rgba(255,255,255,0.1);
  box-shadow:none;
  border:1px solid rgba(255,255,255,0.25);
}

.spinner{
  width:16px; height:16px; border-radius:50%;
  border:2.5px solid rgba(255,255,255,0.35);
  border-top-color:#fff;
  animation: spin 0.7s linear infinite;
}
@keyframes spin{ to{ transform:rotate(360deg); } }

/* ======================================================================
   RESULTS / CHARTS
   ====================================================================== */
.metric-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:14px; margin-bottom:24px;
}
.metric-card{
  background: rgba(255,255,255,0.09);
  border:1px solid rgba(255,255,255,0.16);
  border-radius:18px; padding:18px; text-align:center;
  transition: transform .25s ease, box-shadow .25s ease;
}
.metric-card:hover{ transform: translateY(-4px) scale(1.03); box-shadow:0 10px 26px rgba(0,0,0,0.35); }
.metric-card .m-val{ font-size:1.7rem; font-weight:800; background:linear-gradient(135deg,#c4b5fd,#93c5fd); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.metric-card .m-lbl{ font-size:0.78rem; opacity:0.8; margin-top:6px; text-transform:uppercase; letter-spacing:0.04em; }

.chart-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
  gap:20px;
}
.chart-card{
  background: rgba(255,255,255,0.06);
  border:1px solid rgba(255,255,255,0.15);
  border-radius:20px; padding:16px; text-align:center;
  transition: transform .25s ease, box-shadow .25s ease;
}
.chart-card:hover{ transform:translateY(-5px); box-shadow:0 16px 34px rgba(0,0,0,0.4); }
.chart-card img{ width:100%; height:auto; border-radius:12px; }
.chart-card h4{ font-size:0.9rem; font-weight:700; margin-bottom:10px; opacity:0.9; }

.hidden{ display:none !important; }

.toast{
  position:fixed; top:20px; left:50%; transform:translateX(-50%);
  background: rgba(248,113,113,0.95); color:#fff;
  padding:13px 24px; border-radius:14px; font-weight:600; font-size:0.9rem;
  z-index:999; box-shadow:0 10px 30px rgba(0,0,0,0.4);
  display:flex; align-items:center; gap:10px;
  animation: fadein .25s ease;
}
@keyframes fadein{ from{opacity:0; transform:translate(-50%,-10px);} to{opacity:1; transform:translate(-50%,0);} }

/* ======================================================================
   SHADER PANEL (bottom-right toggle + sliding glass drawer)
   ====================================================================== */
#shaderToggleBtn{
  position:fixed; bottom:26px; right:26px; z-index:50;
  display:flex; align-items:center; gap:10px;
  padding:14px 22px;
  border-radius:999px; border:1px solid rgba(255,255,255,0.3);
  background: rgba(30,18,50,0.65);
  backdrop-filter: blur(14px);
  color:#fff; font-weight:700; font-size:0.9rem;
  cursor:pointer;
  box-shadow: 0 10px 30px rgba(0,0,0,0.4);
  transition: transform .2s ease;
}
#shaderToggleBtn:hover{ transform:translateY(-3px) scale(1.03); }

#shaderPanel{
  position:fixed; bottom:0; right:0; z-index:60;
  width:360px; max-width:92vw; max-height:85vh;
  overflow-y:auto;
  background: rgba(24,14,42,0.78);
  backdrop-filter: blur(24px) saturate(1.4);
  border:1px solid rgba(255,255,255,0.22);
  border-radius:28px 0 0 0;
  padding:26px 24px 30px;
  transform: translateY(110%);
  transition: transform .4s cubic-bezier(.2,.9,.25,1);
  box-shadow: -10px -10px 50px rgba(0,0,0,0.45);
}
#shaderPanel.open{ transform: translateY(0); }
#shaderPanel h3{
  display:flex; align-items:center; gap:10px;
  font-size:1.05rem; margin-bottom:18px; font-weight:800;
}
.shader-group{ margin-bottom:18px; }
.shader-group label{
  display:flex; justify-content:space-between;
  font-size:0.78rem; font-weight:600; opacity:0.85; margin-bottom:6px;
  text-transform:uppercase; letter-spacing:0.03em;
}
.shader-group label span{ opacity:0.6; font-weight:500; text-transform:none; }
input[type=range]{
  width:100%; -webkit-appearance:none; appearance:none;
  height:6px; border-radius:999px;
  background: rgba(255,255,255,0.2);
  outline:none;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none; appearance:none;
  width:18px; height:18px; border-radius:50%;
  background: linear-gradient(135deg,#a78bfa,#60a5fa);
  cursor:pointer; border:2px solid #fff;
  box-shadow:0 2px 8px rgba(0,0,0,0.4);
}
input[type=range]::-moz-range-thumb{
  width:18px; height:18px; border-radius:50%;
  background: linear-gradient(135deg,#a78bfa,#60a5fa);
  cursor:pointer; border:2px solid #fff;
}
.shader-color-row{ display:flex; align-items:center; gap:10px; }
input[type=color]{
  width:42px; height:42px; border-radius:12px; border:1px solid rgba(255,255,255,0.3);
  background:none; cursor:pointer; padding:0;
}
.color-presets{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.color-swatch{
  width:30px; height:30px; border-radius:50%;
  border:2px solid rgba(255,255,255,0.4);
  cursor:pointer; transition: transform .15s ease;
}
.color-swatch:hover{ transform:scale(1.15); }
.shader-reset{ width:100%; margin-top:6px; }

/* ======================================================================
   BACKGROUND SWITCHER (bottom-center)
   ====================================================================== */
#bgSwitcher{
  position:fixed; bottom:26px; left:50%; transform:translateX(-50%);
  z-index:50;
  display:flex; align-items:center; gap:10px;
  background: rgba(30,18,50,0.6);
  backdrop-filter: blur(14px);
  border:1px solid rgba(255,255,255,0.25);
  border-radius:999px;
  padding:10px 14px;
  box-shadow:0 10px 30px rgba(0,0,0,0.4);
}
#bgSwitcher button{
  background:none; border:none; color:#fff; cursor:pointer;
  width:32px; height:32px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  transition: background .2s ease;
}
#bgSwitcher button:hover{ background: rgba(255,255,255,0.18); }
.bg-dots{ display:flex; gap:6px; }
.bg-dot{
  width:9px; height:9px; border-radius:50%;
  background: rgba(255,255,255,0.35);
  cursor:pointer; transition: all .2s ease;
}
.bg-dot.active{ background:#a78bfa; transform:scale(1.3); }

/* ======================================================================
   RESPONSIVE
   ====================================================================== */
@media (max-width: 768px){
  .app-header h1{ font-size:1.7rem; }
  .glass-panel{ padding:22px; border-radius:26px; }
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

  <!-- ==================== HEADER ==================== -->
  <header class="app-header">
    <h1><i class="fa-solid fa-chart-line"></i> Universal CSV Analyzer</h1>
    <p>Upload any CSV, pick a target, and train a Classification or Regression model in seconds.</p>
  </header>

  <!-- ==================== 1. UPLOAD CARD ==================== -->
  <section class="glass-panel">
    <div class="section-title"><i class="fa-solid fa-cloud-arrow-up"></i> Upload Dataset</div>
    <div id="dropzone" class="dropzone">
      <i class="fa-solid fa-file-csv"></i>
      <h3>Drag &amp; drop your CSV here</h3>
      <p>or click to browse files &middot; max 64MB</p>
      <input type="file" id="fileInput" accept=".csv" />
    </div>
    <div id="fileChipWrap"></div>
  </section>

  <!-- ==================== 2. DATASET SUMMARY ==================== -->
  <section id="summarySection" class="glass-panel hidden">
    <div class="section-title"><i class="fa-solid fa-table-list"></i> Dataset Summary</div>
    <div class="stat-grid" id="statGrid"></div>
    <div class="table-wrap">
      <table id="columnsTable">
        <thead><tr><th>Column</th><th>Type</th><th>Unique</th><th>Missing</th><th>Suggested Task</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <!-- ==================== 3. PREVIEW ==================== -->
  <section id="previewSection" class="glass-panel hidden">
    <div class="section-title"><i class="fa-solid fa-eye"></i> Data Preview <span style="opacity:.6;font-weight:400;font-size:.85rem;">(first 10 rows)</span></div>
    <div class="table-wrap">
      <table id="previewTable">
        <thead></thead>
        <tbody></tbody>
      </table>
    </div>
  </section>

  <!-- ==================== 4. CONFIGURE & TRAIN ==================== -->
  <section id="configSection" class="glass-panel hidden">
    <div class="section-title"><i class="fa-solid fa-sliders"></i> Configure Model</div>

    <div class="form-grid">
      <div class="field">
        <label>Target Column</label>
        <select id="targetSelect"></select>
      </div>
      <div class="field">
        <label>Detected Task</label>
        <div><span id="taskBadge" class="task-badge classification"><i class="fa-solid fa-wand-magic-sparkles"></i> —</span></div>
      </div>
      <div class="field">
        <label>Model</label>
        <select id="modelSelect"></select>
      </div>
      <div class="field">
        <label>Test Size <span id="testSizeVal" style="opacity:.6;font-weight:400;">(20%)</span></label>
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

  <!-- ==================== 5. RESULTS ==================== -->
  <section id="resultsSection" class="glass-panel hidden">
    <div class="section-title"><i class="fa-solid fa-flask-vial"></i> Results</div>
    <div class="metric-grid" id="metricGrid"></div>
    <div class="chart-grid" id="chartGrid"></div>
  </section>

</div>

<!-- ==================== SHADER PANEL TOGGLE ==================== -->
<button id="shaderToggleBtn"><i class="fa-solid fa-palette"></i> Shader Panel</button>

<div id="shaderPanel">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;">
    <h3 style="margin:0;"><i class="fa-solid fa-wand-magic-sparkles"></i> Liquid Glass Shader</h3>
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

<!-- ==================== BACKGROUND SWITCHER ==================== -->
<div id="bgSwitcher">
  <button id="bgPrev"><i class="fa-solid fa-chevron-left"></i></button>
  <div class="bg-dots" id="bgDots"></div>
  <button id="bgNext"><i class="fa-solid fa-chevron-right"></i></button>
</div>

<script>
/* ============================================================
   GLOBAL STATE
   ============================================================ */
let datasetInfo = null;          // response from /upload
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

/* ============================================================
   TOAST / ERROR MESSAGES
   ============================================================ */
function showToast(message){
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span>${message}</span>`;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

/* ============================================================
   FILE UPLOAD (drag & drop + click)
   ============================================================ */
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');

dropzone.addEventListener('click', () => fileInput.click());

['dragenter','dragover'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.add('dragover');
  });
});
['dragleave','drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.remove('dragover');
  });
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

/* ============================================================
   RENDER: DATASET SUMMARY
   ============================================================ */
function renderSummary(data){
  const statGrid = document.getElementById('statGrid');
  statGrid.innerHTML = `
    <div class="stat-tile"><div class="val">${data.n_rows}</div><div class="lbl">Rows</div></div>
    <div class="stat-tile"><div class="val">${data.n_cols}</div><div class="lbl">Columns</div></div>
    <div class="stat-tile"><div class="val">${data.total_missing}</div><div class="lbl">Missing Values</div></div>
  `;

  const tbody = document.querySelector('#columnsTable tbody');
  tbody.innerHTML = data.columns.map(col => `
    <tr>
      <td><strong>${col.name}</strong></td>
      <td>${col.dtype}</td>
      <td>${col.unique}</td>
      <td><span class="badge ${col.missing > 0 ? 'missing-some' : 'missing-none'}">${col.missing}</span></td>
      <td>${col.suggested_task === 'classification' ? 'Classification' : 'Regression'}</td>
    </tr>
  `).join('');
}

/* ============================================================
   RENDER: PREVIEW TABLE
   ============================================================ */
function renderPreview(data){
  const thead = document.querySelector('#previewTable thead');
  const tbody = document.querySelector('#previewTable tbody');
  thead.innerHTML = '<tr>' + data.preview_columns.map(c => `<th>${c}</th>`).join('') + '</tr>';
  tbody.innerHTML = data.preview_rows.map(row =>
    '<tr>' + data.preview_columns.map(c => `<td>${row[c]}</td>`).join('') + '</tr>'
  ).join('');
}

/* ============================================================
   TARGET COLUMN / TASK DETECTION / MODEL LIST
   ============================================================ */
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

/* ============================================================
   TRAIN MODEL
   ============================================================ */
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
    body: JSON.stringify({
      target, model, test_size: testSize,
      max_depth: maxDepth, C: C, max_iter: maxIter
    })
  })
  .then(r => r.json())
  .then(data => {
    resetTrainButton();
    if (!data.success){
      showToast(data.error || 'Training failed.');
      return;
    }
    renderResults(data);
    show('resultsSection');
    document.getElementById('resultsSection').scrollIntoView({ behavior:'smooth', block:'start' });
  })
  .catch(err => {
    resetTrainButton();
    showToast('Training failed: ' + err);
  });
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

/* ============================================================
   RENDER RESULTS: METRICS + CHARTS
   ============================================================ */
const METRIC_LABELS = {
  accuracy: 'Accuracy', precision: 'Precision', recall: 'Recall', f1_score: 'F1 Score', roc_auc: 'ROC AUC',
  r2_score: 'R\u00b2 Score', mse: 'MSE', rmse: 'RMSE', mae: 'MAE'
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
      <div class="m-val">${data.n_train}/${data.n_test}</div>
      <div class="m-lbl">Train / Test rows</div>
    </div>
  `;

  const chartGrid = document.getElementById('chartGrid');
  chartGrid.innerHTML = Object.entries(data.charts).map(([k, b64]) => `
    <div class="chart-card">
      <h4>${CHART_LABELS[k] || k}</h4>
      <img src="data:image/png;base64,${b64}" alt="${k}" />
    </div>
  `).join('');
}

/* ============================================================
   SHADER PANEL — real-time "Liquid Glass" CSS variable control
   ============================================================ */
const root = document.documentElement;

const shaderToggleBtn = document.getElementById('shaderToggleBtn');
const shaderPanel = document.getElementById('shaderPanel');
shaderToggleBtn.addEventListener('click', () => shaderPanel.classList.toggle('open'));

// current tint color / alpha kept separately so we can recompose rgba()
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

// --- Blur ---
const sBlur = document.getElementById('s-blur');
sBlur.addEventListener('input', () => {
  root.style.setProperty('--blur', sBlur.value + 'px');
  document.getElementById('v-blur').textContent = sBlur.value + 'px';
});

// --- Frostiness ---
const sFrost = document.getElementById('s-frost');
sFrost.addEventListener('input', () => {
  const pct = parseFloat(sFrost.value);
  root.style.setProperty('--frost', pct);
  root.style.setProperty('--frost-alpha', (pct / 100).toFixed(2));
  document.getElementById('v-frost').textContent = pct + '%';
});

// --- Saturation ---
const sSat = document.getElementById('s-sat');
sSat.addEventListener('input', () => {
  root.style.setProperty('--saturation', sSat.value);
  document.getElementById('v-sat').textContent = parseFloat(sSat.value).toFixed(1);
});

// --- Tint alpha + color ---
const sTint = document.getElementById('s-tint');
const sTintColor = document.getElementById('s-tint-color');
sTint.addEventListener('input', () => {
  tintAlpha = parseFloat(sTint.value) / 100;
  document.getElementById('v-tint').textContent = sTint.value + '%';
  applyTint();
});
sTintColor.addEventListener('input', () => {
  tintColor = hexToRgb(sTintColor.value);
  applyTint();
});

// --- Bevel ---
const sBevel = document.getElementById('s-bevel');
sBevel.addEventListener('input', () => {
  root.style.setProperty('--bevel', sBevel.value + 'px');
  document.getElementById('v-bevel').textContent = sBevel.value;
});

// --- Specular (stored as 0-1 fraction) ---
const sSpec = document.getElementById('s-spec');
sSpec.addEventListener('input', () => {
  root.style.setProperty('--specular', (parseFloat(sSpec.value) / 100).toFixed(2));
  document.getElementById('v-spec').textContent = sSpec.value + '%';
});

// --- 3D Depth ---
const sDepth = document.getElementById('s-depth');
sDepth.addEventListener('input', () => {
  root.style.setProperty('--depth', sDepth.value + 'px');
  document.getElementById('v-depth').textContent = sDepth.value + 'px';
});

// --- Radius ---
const sRadius = document.getElementById('s-radius');
sRadius.addEventListener('input', () => {
  root.style.setProperty('--radius', sRadius.value + 'px');
  document.getElementById('v-radius').textContent = sRadius.value + 'px';
});

// --- Contrast ---
const sContrast = document.getElementById('s-contrast');
sContrast.addEventListener('input', () => {
  root.style.setProperty('--contrast', sContrast.value);
  document.getElementById('v-contrast').textContent = parseFloat(sContrast.value).toFixed(2);
});

// --- Brightness ---
const sBright = document.getElementById('s-bright');
sBright.addEventListener('input', () => {
  root.style.setProperty('--brightness', sBright.value);
  document.getElementById('v-bright').textContent = parseFloat(sBright.value).toFixed(2);
});

// --- Text color presets + picker ---
document.querySelectorAll('.color-swatch').forEach(sw => {
  sw.addEventListener('click', () => {
    const c = sw.dataset.color;
    root.style.setProperty('--white-text', c);
    document.getElementById('s-text-color').value = c;
  });
});
document.getElementById('s-text-color').addEventListener('input', (e) => {
  root.style.setProperty('--white-text', e.target.value);
});

// --- Reset to defaults ---
const DEFAULTS = {
  blur: 0, frost: 0, sat: 1.0, tintAlpha: 50, tintColor: '#332150',
  bevel: 6, spec: 100, depth: 80, radius: 40, contrast: 1.60, bright: 1.40, text: '#ffffff'
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
  document.getElementById('s-text-color').value = DEFAULTS.text;
  root.style.setProperty('--white-text', DEFAULTS.text);
});

// initialize frost-alpha + tint on load
root.style.setProperty('--frost-alpha', '0');
applyTint();

/* ============================================================
   BACKGROUND SWITCHER (7 images, bottom-center)
   ============================================================ */
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

# ==========================================================================
# ENTRYPOINT
# ==========================================================================
if __name__ == "__main__":
    ensure_background_images()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))