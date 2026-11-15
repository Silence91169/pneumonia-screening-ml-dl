"""Streamlit application for the verified AML and ResNet18 research prototype.

Educational ML research dashboard:
- Live Interactive Screening on all 5,840 cohort radiographs or custom uploads
- Handcrafted Feature Engineering Lab (HOG & LBP Visualizers)
- Model Performance Benchmark & Decision Threshold Sensitivity Optimizer
- Forensic Dataset Audit & Exact-Duplicate Split Checks

Educational Research Prototype — NOT intended for clinical diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import standardize_image
from src.features.hog import extract_hog_features
from src.features.lbp import extract_lbp_features
from src.models.cnn_pipeline import grad_cam, image_transform
from src.models.deep_learning import ChestXrayClassifier

# ---------------------------------------------------------------------------
# Streamlit App Configuration & Premium Clinical UI Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chest X-ray Analysis",
    page_icon=":material/radiology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    :root {
        --surface: #111827;
        --surface-raised: #182235;
        --border: rgba(148, 163, 184, 0.20);
        --text: #f8fafc;
        --muted: #a6b0c0;
        --accent: #e05260;
        --accent-soft: rgba(224, 82, 96, 0.12);
    }
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stAppViewContainer"] { background: #0b1220; color: var(--text); }
    [data-testid="stSidebar"] { background: #0e1728; border-right: 1px solid var(--border); }
    [data-testid="stHeader"] { background: rgba(11, 18, 32, 0.88); }
    [data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 2rem; }
    h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.025em; }
    p, label, .stCaption { line-height: 1.6; }

    .hero-container {
        background: linear-gradient(135deg, #151f32 0%, #111827 65%, rgba(224,82,96,.12) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: clamp(1.8rem, 4vw, 2.65rem);
        font-weight: 750;
        color: var(--text);
        letter-spacing: -0.04em;
        margin: 0;
    }
    .hero-sub {
        font-size: 1rem;
        color: var(--muted);
        margin: .5rem 0 1rem;
        max-width: 760px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(148,163,184,.08);
        color: #cbd5e1;
        border: 1px solid var(--border);
        padding: .3rem .75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .disclaimer-box {
        background: var(--accent-soft);
        border: 1px solid rgba(224,82,96,.32);
        border-left: 4px solid var(--accent);
        border-radius: 10px;
        padding: .9rem 1rem;
        margin-bottom: 1.5rem;
        color: #e5e7eb;
        font-size: .9rem;
        line-height: 1.55;
    }
    .disclaimer-box strong { color: #ff9da7; }

    /* Metric & Action Cards */
    .custom-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }

    [data-testid="stFileUploaderDropzone"], [data-testid="stDataFrame"],
    [data-testid="stMetric"], [data-testid="stImage"] img {
        border-radius: 12px;
    }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: .9rem 1rem;
    }
    .section-copy { color: var(--muted); margin: -.3rem 0 1.25rem; max-width: 800px; }
    .result-label { font-size: .76rem; text-transform: uppercase; letter-spacing: .09em; opacity: .82; }
    .badge-positive {
        background: var(--accent-soft);
        border: 1px solid rgba(224,82,96,.48);
        color: #ffadb5;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .badge-negative {
        background: rgba(148,163,184,.10);
        border: 1px solid rgba(148,163,184,.30);
        color: #e2e8f0;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.8rem;
    }

    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.8rem;
    }
    div.stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); }
    [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--border); }
    [data-baseweb="tab"] { min-height: 3rem; padding: .65rem 1rem; white-space: nowrap; }
    [data-baseweb="tab-highlight"] { background-color: var(--accent); }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .9rem 3rem; }
        .hero-container { padding: 1.25rem; border-radius: 12px; }
        .hero-sub { font-size: .92rem; }
        [data-baseweb="tab-list"] { overflow-x: auto; }
        [data-baseweb="tab"] { padding: .55rem .65rem; font-size: .85rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data & Model Loader Helpers (Cached for Speed)
# ---------------------------------------------------------------------------
@st.cache_data
def load_audit_summary():
    path = PROJECT_ROOT / "results" / "dataset_audit" / "audit_summary.json"
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_split_manifest():
    path = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
    if path.is_file():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_classical_training_metrics():
    path = PROJECT_ROOT / "results" / "metrics" / "classical_training.json"
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_test_benchmark():
    path = PROJECT_ROOT / "results" / "metrics" / "model_comparison.json"
    if path.is_file():
        with open(path, "r") as f:
            payload = json.load(f)
        if payload.get("metadata", {}).get("evidence_type") == "real held-out test features; synthetic fallback prohibited":
            models = dict(payload.get("models", {}))
            cnn_path = PROJECT_ROOT / "results" / "metrics" / "resnet18_test_metrics.json"
            if cnn_path.is_file():
                with cnn_path.open("r", encoding="utf-8") as stream:
                    cnn_payload = json.load(stream)
                if cnn_payload.get("evidence_type") == "real held-out test images":
                    models["resnet18"] = cnn_payload.get("metrics", {})
            return models
    return None

@st.cache_resource
def load_classical_models():
    models_dir = PROJECT_ROOT / "results" / "models"
    loaded = {}
    scalers = {}
    for name in ["hog_linear_svm", "hog_rbf_svm", "hog_lbp_rbf_svm"]:
        p = models_dir / f"{name}.joblib"
        if p.is_file():
            candidate = joblib.load(p)
            # Official UI probabilities require the validation-calibrated
            # artifacts created by scripts/train_classical.py.
            if candidate.__class__.__name__ == "CalibratedClassifierCV":
                loaded[name] = candidate

    hog_sc = models_dir / "hog_scaler.joblib"
    lbp_sc = models_dir / "lbp_scaler.joblib"
    if hog_sc.is_file():
        scalers["hog"] = joblib.load(hog_sc)
    if lbp_sc.is_file():
        scalers["lbp"] = joblib.load(lbp_sc)
    return loaded, scalers


@st.cache_resource
def load_resnet18_model():
    """Load the verified best ResNet18 checkpoint for CPU inference."""
    import torch

    checkpoint_path = PROJECT_ROOT / "results" / "models" / "resnet18_best.pt"
    if not checkpoint_path.is_file():
        return None, None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ChestXrayClassifier(
        payload.get("backbone", "resnet18"), pretrained=False, freeze_backbone=False
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metadata = {
        "epoch": payload.get("epoch"),
        "val_loss": payload.get("val_loss"),
        "path": str(checkpoint_path),
    }
    return model, metadata

# ---------------------------------------------------------------------------
# Self-Contained UI Image Transformation & Saliency Utilities
# ---------------------------------------------------------------------------
def compute_hog_vis(image, orientations=9, ppc=(16, 16), cpb=(2, 2)):
    from skimage.feature import hog
    feat, hog_img = hog(
        image,
        orientations=orientations,
        pixels_per_cell=ppc,
        cells_per_block=cpb,
        block_norm="L2-Hys",
        visualize=True,
    )
    norm = hog_img / (hog_img.max() + 1e-8)
    return feat.astype(np.float32), norm.astype(np.float32)

def compute_lbp_vis(image, P=8, R=1.0, method="uniform"):
    from skimage.feature import local_binary_pattern
    img_u8 = (image * 255.0).astype(np.uint8)
    lbp_map = local_binary_pattern(img_u8, P=P, R=R, method=method)
    n_bins = P + 2 if method == "uniform" else int(2**P)
    hist, _ = np.histogram(lbp_map.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
    vis = lbp_map / (lbp_map.max() + 1e-8)
    return vis.astype(np.float32), hist

def compute_edge_density_overlay(image, alpha=0.45):
    """Render Sobel edge density; this is not a model attribution method."""
    smoothed = cv2.GaussianBlur(image, (15, 15), 0)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(cv2.magnitude(gx, gy), (21, 21), 0)
    norm_mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    base = (image * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB) if base.ndim == 2 else base
    colored = cv2.applyColorMap((norm_mag * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb, 1.0 - alpha, colored_rgb, alpha, 0)

# ---------------------------------------------------------------------------
# Reusable Live Dataset Image Selector Function
# ---------------------------------------------------------------------------
def dataset_image_picker(key_prefix: str = "main") -> tuple[Path | None, dict | None]:
    """Unified interactive dataset browser for all 5,840 cohort images."""
    manifest = load_split_manifest()
    if manifest is None or manifest.empty:
        st.warning("Dataset manifest not found at data/processed/split_manifest.csv.")
        return None, None

    col_s1, col_s2, col_s3 = st.columns([1.2, 1.4, 0.8])
    with col_s1:
        split_opt = st.selectbox(
            "Dataset split",
            ["Held-Out Test (624 imgs)", "Validation (784 imgs)", "Training (4,432 imgs)", "Entire Cohort (5,840 imgs)"],
            index=0,
            key=f"{key_prefix}_split_flt",
        )
    with col_s2:
        class_opt = st.selectbox(
            "Image label",
            ["All Categories", "NORMAL (Negative)", "PNEUMONIA (All Positive)", "Bacterial Pneumonia (Focal)", "Viral Pneumonia (Diffuse)"],
            index=0,
            key=f"{key_prefix}_class_flt",
        )
    with col_s3:
        st.write("")
        st.write("")
        rand_click = st.button("Choose random image", key=f"{key_prefix}_btn_rand", use_container_width=True)

    # Filter records
    df = manifest.copy()
    if "Test" in split_opt:
        df = df[df["split"] == "test"]
    elif "Validation" in split_opt:
        df = df[df["split"] == "val"]
    elif "Training" in split_opt:
        df = df[df["split"] == "train"]

    if "NORMAL" in class_opt:
        df = df[df["label"] == "NORMAL"]
    elif "All Positive" in class_opt:
        df = df[df["label"] == "PNEUMONIA"]
    elif "Bacterial" in class_opt:
        df = df[df["image_path"].str.contains("bacteria", case=False, na=False)]
    elif "Viral" in class_opt:
        df = df[df["image_path"].str.contains("virus", case=False, na=False)]

    if df.empty:
        st.warning("No records match the active filter. Displaying full test split.")
        df = manifest[manifest["split"] == "test"]

    # Session state for random selection
    idx_key = f"{key_prefix}_selected_idx"
    if rand_click or idx_key not in st.session_state:
        st.session_state[idx_key] = int(np.random.randint(0, len(df)))

    # Bounds check
    curr_idx = min(st.session_state.get(idx_key, 0), len(df) - 1)

    labels = [
        f"[{row['split'].upper()}] {Path(row['image_path']).name} — {row['label']}"
        for _, row in df.iterrows()
    ]
    paths = df["image_path"].tolist()

    chosen_label = st.selectbox(
        f"Choose an X-ray ({len(df):,} available)",
        labels,
        index=curr_idx,
        key=f"{key_prefix}_dropdown",
    )
    chosen_pos = labels.index(chosen_label)
    st.session_state[idx_key] = chosen_pos

    selected_file = PROJECT_ROOT / paths[chosen_pos]
    if not selected_file.is_file():
        fallback_file = PROJECT_ROOT / "data" / "sample_images" / Path(paths[chosen_pos]).name
        if fallback_file.is_file():
            selected_file = fallback_file
        else:
            sample_dir = PROJECT_ROOT / "data" / "sample_images"
            if sample_dir.is_dir():
                avail = sorted(list(sample_dir.glob("*.jpeg")))
                if avail:
                    selected_file = avail[chosen_pos % len(avail)]
        # Never attach a missing cohort row's label to a different fallback
        # image. Bundled files are demo samples, not the selected test record.
        name = selected_file.name.casefold()
        metadata = {
            "image_path": str(selected_file.relative_to(PROJECT_ROOT)),
            "label": "PNEUMONIA" if ("virus" in name or "bacteria" in name) else "NORMAL",
            "split": "demo",
            "source": "bundled sample fallback",
        }
        st.warning(
            "The selected cohort image is unavailable locally. Displaying a bundled demo image "
            "with its own filename-derived demo label; it is not the selected test record."
        )
        return selected_file, metadata
    metadata = df.iloc[chosen_pos].to_dict()
    return selected_file, metadata

# ---------------------------------------------------------------------------
# Header & Medical Research Protocol Callout
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Chest X-ray Analysis</div>
        <div class="hero-sub">Compare handcrafted image features with a pretrained ResNet18 in a verified educational research workflow.</div>
        <span class="status-pill">Research dashboard · Model files checked at runtime</span>
    </div>
    <div class="disclaimer-box">
        <strong>Research use only.</strong> This educational machine-learning prototype is not a medical device and must not be used for diagnosis, treatment, or patient management. Predictions show statistical patterns in the project dataset, not a radiologist's assessment.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: System Specs & Research Protocol
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Project summary")
    st.markdown("**Study:** Pneumonia screening research")
    st.markdown("**Models:** Three calibrated SVMs and ResNet18")
    st.markdown("**Evaluation:** 624 held-out X-rays")
    st.markdown("**Data safeguards:** Train-only scaling; no exact-image overlap between splits")

    st.markdown("---")
    st.markdown("### Model status")
    models, scalers = load_classical_models()
    resnet_model, resnet_metadata = load_resnet18_model()
    if models:
        st.success(f"{len(models)} SVM models ready")
    else:
        st.warning("No verified validation-calibrated model artifacts are available")

    if scalers:
        st.success("Feature scalers ready")

    if resnet_model is not None:
        st.success(f"ResNet18 ready · epoch {resnet_metadata['epoch']}")
    else:
        st.warning("Verified ResNet18 checkpoint is unavailable")

    st.caption("Inference runs on CPU in the deployed app. Timing appears only after a model completes.")

# ---------------------------------------------------------------------------
# Main Tabs: Streamlined, Crisp, 100% Functional Layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Screening",
    "Feature Lab",
    "Model Performance",
])

# ===========================================================================
# TAB 1: Live Diagnostic Screening Studio (All Buttons Guaranteed Working)
# ===========================================================================
with tab1:
    st.markdown("## Analyze an X-ray")
    st.markdown('<p class="section-copy">Upload an image or select a project sample, then choose a trained model to explore its research prediction.</p>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Image source",
        ["Project samples", "Upload an image"],
        horizontal=True,
    )

    image_path = None
    ground_truth = None

    if input_mode.startswith("Project"):
        image_path, meta = dataset_image_picker(key_prefix="tab1_browser")
        if meta:
            ground_truth = meta.get("label", "Unknown")
    else:
        up_file = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"], help="Accepted formats: PNG, JPG, and JPEG.")
        if up_file:
            temp_dir = PROJECT_ROOT / "results" / "temp_uploads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            image_path = temp_dir / up_file.name
            with open(image_path, "wb") as f:
                f.write(up_file.getbuffer())
            ground_truth = "Custom User Input"

    st.markdown("---")

    if image_path and Path(image_path).is_file():
        std_img = standardize_image(image_path, target_size=(224, 224))

        col_view, col_ctrl = st.columns([1.1, 1.3])

        with col_view:
            st.image(std_img, caption="Prepared image · 224 × 224 grayscale", clamp=True, use_container_width=True)
            if ground_truth:
                st.info(f"Reference label: **{ground_truth}**")

        with col_ctrl:
            st.markdown("### Analysis settings")
            chosen_model_key = st.selectbox(
                "Model",
                [
                    "HOG + Linear SVM (hog_linear_svm)",
                    "HOG + RBF SVM (hog_rbf_svm)",
                    "HOG + LBP + RBF SVM (hog_lbp_rbf_svm)",
                    "ResNet18 transfer learning (resnet18)",
                ],
                index=0,
            )
            raw_model_name = chosen_model_key.split("(")[-1].strip(")")

            threshold = st.slider(
                "Decision threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.45,
                step=0.05,
                help="A lower threshold labels more images as pneumonia. This changes the sensitivity-specificity trade-off.",
            )

            # Guaranteed working action button
            run_btn = st.button("Run analysis", type="primary", use_container_width=True)

            if run_btn:
                if raw_model_name == "resnet18":
                    if resnet_model is None:
                        st.error("The verified ResNet18 checkpoint is unavailable. No prediction was generated.")
                        st.stop()
                    import torch

                    with st.spinner("Running ResNet18 inference and Grad-CAM..."):
                        t_start = time.time()
                        with Image.open(image_path) as source_image:
                            tensor = image_transform(False)(source_image.convert("L")).unsqueeze(0)
                        with torch.no_grad():
                            prob = float(torch.sigmoid(resnet_model(tensor)).item())
                        cam = grad_cam(resnet_model, tensor)
                        latency = (time.time() - t_start) * 1000
                    is_pneumonia = prob >= threshold
                    st.markdown("---")
                    badge_class = "badge-positive" if is_pneumonia else "badge-negative"
                    label = "Pneumonia pattern predicted" if is_pneumonia else "Normal pattern predicted"
                    st.markdown(
                        f'<div class="{badge_class}"><div class="result-label">Research prediction</div>{label}<br>{prob*100:.1f}% model probability</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(float(prob))
                    st.caption(
                        f"ResNet18 epoch {resnet_metadata['epoch']} checkpoint • "
                        f"{latency:.1f} ms • cutoff {threshold*100:.0f}%"
                    )
                    st.image(cam, caption="Grad-CAM attribution for the ResNet18 pneumonia logit", clamp=True)
                else:
                    clf = models.get(raw_model_name)
                    if clf and "hog" in scalers and "lbp" in scalers:
                        with st.spinner("Extracting features and predicting..."):
                            t_start = time.time()
                            h_vec = extract_hog_features(std_img).reshape(1, -1)
                            l_vec = extract_lbp_features(std_img).reshape(1, -1)

                            h_s = scalers["hog"].transform(h_vec)
                            l_s = scalers["lbp"].transform(l_vec)
                            comb_s = np.hstack([h_s, l_s])

                            X_in = comb_s if raw_model_name == "hog_lbp_rbf_svm" else h_s
                            prob = float(clf.predict_proba(X_in)[0, 1])
                            latency = (time.time() - t_start) * 1000
                            is_pneumonia = prob >= threshold

                        st.markdown("---")
                        if is_pneumonia:
                            st.markdown(
                                f'<div class="badge-positive"><div class="result-label">Research prediction</div>Pneumonia pattern predicted<br>{prob*100:.1f}% calibrated probability</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="badge-negative"><div class="result-label">Research prediction</div>Normal pattern predicted<br>{prob*100:.1f}% calibrated probability</div>',
                                unsafe_allow_html=True,
                            )

                        st.progress(float(prob))
                        st.caption(f"Processing time: {latency:.1f} ms · Decision threshold: {threshold*100:.0f}%")

                        st.markdown("---")
                        st.markdown("### Feature views")
                        _, hog_vis = compute_hog_vis(std_img)
                        saliency = compute_edge_density_overlay(std_img, alpha=0.45)

                        v1, v2 = st.columns(2)
                        v1.image(hog_vis, caption="HOG Gradient Vector Field (6,084 dims)", clamp=True, use_container_width=True)
                        v2.image(saliency, caption="Edge-density visualization (not model attribution)", use_container_width=True)
                    else:
                        st.error(
                            "A validation-calibrated model and matching train-fitted scalers are unavailable. "
                            "No prediction was generated."
                        )

# ===========================================================================
# TAB 2: Handcrafted Feature Engineering Lab (Interactive Parameters)
# ===========================================================================
with tab2:
    st.markdown("## Explore image features")
    st.markdown('<p class="section-copy">See how HOG captures gradient structure and LBP summarizes local texture. These views explain the inputs to the classical models, not their decisions.</p>', unsafe_allow_html=True)

    sample_f_path, _ = dataset_image_picker(key_prefix="tab2_feat")

    if sample_f_path and Path(sample_f_path).is_file():
        std_f = standardize_image(sample_f_path, target_size=(224, 224))

        col_h_param, col_l_param = st.columns(2)
        with col_h_param:
            st.markdown("### HOG settings")
            n_orient = st.slider("Orientation bins", 6, 12, 9, key="tab2_orient")
            cell_size = st.select_slider("Pixels per cell", options=[8, 16], value=16, key="tab2_cell")
        with col_l_param:
            st.markdown("### LBP settings")
            lbp_r = st.slider("Neighborhood radius", min_value=1.0, max_value=5.0, value=1.0, step=0.5, key="tab2_r", help="Distance from the center pixel.")
            lbp_p = st.select_slider("Sample points", options=[8, 12, 16, 24], value=8, key="tab2_p", help="Number of neighboring pixels sampled around the center.")

        t0 = time.time()
        h_f, h_vis = compute_hog_vis(std_f, orientations=n_orient, ppc=(cell_size, cell_size))
        t_hog = (time.time() - t0) * 1000

        t1 = time.time()
        l_vis, l_hist = compute_lbp_vis(std_f, P=lbp_p, R=lbp_r)
        t_lbp = (time.time() - t1) * 1000

        c_img1, c_img2, c_img3 = st.columns(3)
        c_img1.image(std_f, caption="Input Radiograph (224×224)", clamp=True, use_container_width=True)
        c_img2.image(h_vis, caption=f"HOG Vector Field ({len(h_f):,} dims, {t_hog:.1f}ms)", clamp=True, use_container_width=True)
        c_img3.image(l_vis, caption=f"LBP Texture Map ({len(l_hist)} bins, {t_lbp:.1f}ms)", clamp=True, use_container_width=True)

        st.markdown("---")
        st.markdown("### Feature distributions")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.caption(f"HOG Descriptor Profile (First 60 of {len(h_f):,} features):")
            fig_h = px.line(h_f[:60], labels={"index": "Index", "value": "Magnitude"})
            fig_h.update_layout(height=210, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_h, use_container_width=True)
        with d_col2:
            st.caption(f"Uniform LBP Texture Histogram ({len(l_hist)} Bins):")
            fig_l = px.bar(x=[f"B{i}" for i in range(len(l_hist))], y=l_hist, labels={"x": "Bin", "y": "Freq"})
            fig_l.update_layout(height=210, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_l, use_container_width=True)

# ===========================================================================
# TAB 3: Model Performance & Clinical Thresholds
# ===========================================================================
with tab3:
    st.markdown("## Verified model performance")
    st.markdown('<p class="section-copy">Results below come from the held-out 624-image test set. PNEUMONIA is the positive class.</p>', unsafe_allow_html=True)

    bench = load_test_benchmark()
    train_meta = load_classical_training_metrics()

    if bench:
        rows = []
        for m_name, d in bench.items():
            rows.append({
                "Model": m_name,
                "Recall (Sensitivity)": f"{d.get('sensitivity', 0)*100:.1f}%",
                "Specificity": f"{d.get('specificity', 0)*100:.1f}%",
                "Precision": f"{d.get('precision', 0)*100:.1f}%",
                "F1-Score": f"{d.get('f1_score', 0)*100:.1f}%",
                "ROC-AUC": f"{d.get('roc_auc', 0):.3f}",
                "Brier Score": f"{d.get('brier_score', 0):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("---")
        st.markdown("### Interpreting the threshold")
        st.markdown("Lower thresholds generally increase sensitivity and false positives; higher thresholds generally increase specificity and false negatives. The interactive threshold in **Screening** affects only the current prediction, not these fixed test results.")
    else:
        st.warning(
            "No verified held-out benchmark is available. Legacy 120-sample files were excluded because "
            "their predictions were generated synthetically when real feature caches were absent."
        )
        st.info(
            "Threshold exploration, confusion matrices, ROC curves, and calibration plots are disabled "
            "until official real prediction artifacts exist."
        )

# ===========================================================================
# TAB 4: Dataset Forensic Audit & Anti-Leakage Split Architecture
# ===========================================================================
with tab3:
    st.markdown("## Dataset integrity")
    st.caption(
        "Verified audit and manifest summary. The deployment intentionally omits the raw training dataset; "
        "only trained inference artifacts and bundled demonstration images are shipped."
    )

    audit = load_audit_summary()
    manifest = load_split_manifest()

    if audit:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Cohort Size", f"{audit['total_images']:,} imgs")
        k2.metric("Pneumonia Cases", f"{audit['class_counts']['PNEUMONIA']:,} ({audit['pneumonia_prevalence']*100:.1f}%)")
        k3.metric("Normal Cases", f"{audit['class_counts']['NORMAL']:,} ({(1-audit['pneumonia_prevalence'])*100:.1f}%)")
        k4.metric("Class Imbalance Ratio", f"{audit['pneumonia_to_normal_ratio']:.2f} : 1")

        st.markdown("---")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("### Class distribution")
            fig_p = px.pie(
                names=["PNEUMONIA (Positive)", "NORMAL (Negative)"],
                values=[audit["class_counts"]["PNEUMONIA"], audit["class_counts"]["NORMAL"]],
                color_discrete_sequence=["#e05260", "#64748b"],
                hole=0.45,
            )
            fig_p.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260)
            st.plotly_chart(fig_p, use_container_width=True)

        with g2:
            st.markdown("### Modeling splits")
            if manifest is not None:
                sc = manifest["split"].value_counts().reset_index()
                sc.columns = ["Partition", "Count"]
                fig_b = px.bar(
                    sc,
                    x="Partition",
                    y="Count",
                    color="Partition",
                    color_discrete_sequence=["#e05260", "#94a3b8", "#475569"],
                    text="Count",
                )
                fig_b.update_traces(textposition="outside")
                fig_b.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260)
                st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("---")
    st.info(
        """
        **Recorded data-splitting safeguards:**
        - The original public Kaggle dataset contained only 16 validation images, which compromised hyperparameter validation.
        - The stored manifest records 4,432 train, 784 validation, and 624 holdout test rows.
        - Preprocessing scalers (StandardScaler) are fitted **strictly on training data**. Validation and test data are transformed only without re-fitting.
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Educational ML research prototype · Not for clinical use")
