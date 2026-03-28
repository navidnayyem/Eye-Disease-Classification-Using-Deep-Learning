import os
import sys
import csv
import json
import random
import argparse
import warnings
import logging
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.get_logger().setLevel("ERROR")

############################################################
# CONFIG
############################################################

CONFIG = {
    "MODEL_DIR":   "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/output/models",
    "OUTPUT_DIR":  "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/output",
    "PRED_DIR":    "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/output/predictions",
    "TEST_DIR":    "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/dataset_processed/test",

    "IMAGE_SIZE":      224,
    "DEFAULT_MODEL":   "ResNet50V2",
    "RANDOM_SEED":     None,    # set an int (e.g. 42) for reproducible picks
    "N_RANDOM":        3,       # images to pick per class in auto mode
    "GRADCAM_ALPHA":   0.45,
    "CONF_HIGH":       0.80,
    "CONF_MEDIUM":     0.60,

    "IMAGE_EXTENSIONS": (
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    ),
}

############################################################
# DISEASE METADATA
############################################################

DISEASE_INFO = {
    "cataract": {
        "full_name":   "Cataract",
        "description": "Clouding of the eye's natural lens, causing blurry vision.",
        "severity":    "Treatable",
        "action":      "Consult an ophthalmologist for lens replacement surgery.",
        "color":       "#2980b9",
    },
    "diabetic_retinopathy": {
        "full_name":   "Diabetic Retinopathy",
        "description": "Damage to retinal blood vessels caused by diabetes.",
        "severity":    "Progressive — urgent",
        "action":      "Urgent ophthalmology referral. Blood sugar control is critical.",
        "color":       "#27ae60",
    },
    "glaucoma": {
        "full_name":   "Glaucoma",
        "description": "Optic nerve damage often caused by raised intraocular pressure.",
        "severity":    "Progressive",
        "action":      "Early treatment can prevent further vision loss. See a specialist.",
        "color":       "#8e44ad",
    },
    "normal": {
        "full_name":   "Normal",
        "description": "No signs of the screened eye diseases detected.",
        "severity":    "None",
        "action":      "Routine annual eye examination recommended.",
        "color":       "#16a085",
    },
}

VALID_MODELS = [
    "ResNet50V2", "Xception", "InceptionV3",
    "DenseNet121", "NASNetMobile",
]

############################################################
# CONFIDENCE TIER
############################################################

def confidence_tier(conf):
    if conf >= CONFIG["CONF_HIGH"]:
        return "High confidence", "#27ae60"
    if conf >= CONFIG["CONF_MEDIUM"]:
        return "Moderate confidence — review recommended", "#e67e22"
    return "Low confidence — do NOT rely on this result", "#c0392b"

############################################################
# RANDOM IMAGE SELECTION
############################################################

def pick_random_images(test_dir, n=None, seed=None):
    """
    Walks test_dir/class_name/ for each class and picks n random images.
    Returns list of (image_path, true_class_name) tuples.
    Randomises selection each run unless seed is set.
    """
    if n is None:
        n = CONFIG["N_RANDOM"]
    if seed is not None:
        random.seed(seed)

    exts    = CONFIG["IMAGE_EXTENSIONS"]
    picked  = []
    classes = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    if not classes:
        raise FileNotFoundError(
            f"No class sub-folders found in TEST_DIR:\n  {test_dir}"
        )

    print(f"\n  Selecting {n} random image(s) per class from:")
    print(f"  {test_dir}\n")

    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        images  = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(exts)
        ]
        if not images:
            print(f"  WARNING: no images found in {cls_dir}")
            continue

        chosen = random.sample(images, min(n, len(images)))
        for img_name in chosen:
            picked.append((os.path.join(cls_dir, img_name), cls))
        print(f"  {cls:<25} {len(chosen)} image(s) selected")

    print()
    return picked

############################################################
# HELPERS
############################################################

def load_class_indices():
    path = os.path.join(CONFIG["OUTPUT_DIR"], "class_indices.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"class_indices.json not found:\n  {path}\n"
            "Set OUTPUT_DIR in CONFIG to your training output folder."
        )
    with open(path) as f:
        indices = json.load(f)
    return {v: k for k, v in indices.items()}


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No Conv2D layer found in model.")


def remove_black_border(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return img_bgr
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img_bgr[y:y + h, x:x + w]
    return cropped if cropped.size > 0 else img_bgr


def preprocess_image(img_path):
    size    = CONFIG["IMAGE_SIZE"]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_bgr   = remove_black_border(img_bgr)
    img_bgr   = cv2.resize(img_bgr, (size, size),
                            interpolation=cv2.INTER_AREA)
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb / 255.0, axis=0).astype(np.float32)
    return img_rgb, img_array

############################################################
# GRADCAM++
############################################################

def compute_gradcam_pp(model, img_array, class_idx, layer_name):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            conv_outputs, predictions = grad_model(img_array, training=False)
            loss = predictions[:, class_idx]
        grads = tape1.gradient(loss, conv_outputs)
    grads2 = tape2.gradient(grads, conv_outputs)

    conv_np  = conv_outputs[0].numpy()
    grads_np = grads[0].numpy()

    numerator   = grads_np ** 2
    denominator = (2.0 * grads_np ** 2
                   + np.sum(conv_np * grads_np ** 3,
                             axis=(0, 1), keepdims=True)
                   + 1e-8)
    alphas  = numerator / denominator
    weights = np.sum(alphas * np.maximum(grads_np, 0), axis=(0, 1))
    cam     = np.maximum(np.sum(weights * conv_np, axis=-1), 0)
    cam     = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
    lo, hi  = cam.min(), cam.max()
    cam     = (cam - lo) / (hi - lo + 1e-8)
    return cam.astype(np.float32)


def overlay_heatmap(img_rgb, cam):
    alpha   = CONFIG["GRADCAM_ALPHA"]
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap + (1 - alpha) * img_rgb).astype(np.uint8)

############################################################
# PREDICTION
############################################################

def predict(model, img_array, idx_to_class):
    probs   = model.predict(img_array, verbose=0)[0]
    results = [(idx_to_class[i], float(probs[i]))
               for i in range(len(probs))]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

############################################################
# CONSOLE OUTPUT
############################################################

def print_results(img_path, results, model_name, true_class=None):
    top_class, top_conf = results[0]
    info               = DISEASE_INFO.get(top_class, {})
    tier_label, _      = confidence_tier(top_conf)

    correct = None
    if true_class is not None:
        correct = (top_class == true_class)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Image      : {os.path.basename(img_path)}")
    print(f"  Model      : {model_name}")
    if true_class is not None:
        true_label = DISEASE_INFO.get(true_class, {}).get(
            "full_name", true_class
        )
        result_tag = "CORRECT" if correct else "WRONG"
        print(f"  True class : {true_label}  [{result_tag}]")
    print(sep)
    print(f"  Prediction : {info.get('full_name', top_class)}")
    print(f"  Confidence : {top_conf:.1%}  [{tier_label}]")
    print(f"  Severity   : {info.get('severity', '')}")
    print(f"  Description: {info.get('description', '')}")
    print(f"  Action     : {info.get('action', '')}")
    print(f"\n  All class probabilities:")
    for cls, prob in results:
        filled = int(prob * 30)
        bar    = "█" * filled + " " * (30 - filled)
        name   = DISEASE_INFO.get(cls, {}).get("full_name", cls)
        arrow  = " <- predicted" if cls == top_class else ""
        print(f"    {name:<25} {bar}  {prob:.1%}{arrow}")
    print(sep)
    print(
        "\n  DISCLAIMER: This is an AI screening tool only.\n"
        "  It does NOT replace professional medical diagnosis.\n"
        "  Always consult a qualified ophthalmologist.\n"
    )

############################################################
# FIGURE
############################################################

def save_prediction_figure(img_path, img_rgb, results, model_name,
                            cam=None, out_dir=None, true_class=None):
    top_class, top_conf    = results[0]
    info                   = DISEASE_INFO.get(top_class, {})
    tier_label, tier_color = confidence_tier(top_conf)
    pred_color             = info.get("color", "#2c3e50")

    correct  = (top_class == true_class) if true_class else None
    has_cam  = cam is not None
    n_cols   = 2 if has_cam else 1

    # Fix 1: wider figure + more left margin for bar chart y-labels
    # "Diabetic Retinopathy" is ~20 chars — needs room on the left
    fig = plt.figure(figsize=(5.0 * (n_cols + 1.6), 6.2))
    fig.patch.set_facecolor("white")

    # Fix 2: increase top margin so title + tier text don't overlap
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.05, right=0.97)

    w_ratios = ([1, 1, 1.8] if has_cam else [1, 1.8])
    gs = gridspec.GridSpec(
        1, n_cols + 1,
        figure=fig,
        width_ratios=w_ratios,
        wspace=0.35
    )

    # Original image — title above, not overlapping
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_rgb)
    ax_orig.axis("off")
    ax_orig.set_title("Input fundus image",
                       fontsize=9, color="#555", pad=8)

    # GradCAM++ overlay
    if has_cam:
        ax_cam = fig.add_subplot(gs[0, 1])
        ax_cam.imshow(overlay_heatmap(img_rgb, cam))
        ax_cam.axis("off")
        ax_cam.set_title("GradCAM++ — model attention",
                          fontsize=9, color="#555", pad=8)
        ax_bar = fig.add_subplot(gs[0, 2])
    else:
        ax_bar = fig.add_subplot(gs[0, 1])

    # Fix 3: give the bar chart enough left margin for long y-labels
    ax_bar.yaxis.set_tick_params(labelsize=9, pad=4)

    # Confidence bar chart
    class_names = [
        DISEASE_INFO.get(c, {}).get("full_name", c)
        for c, _ in reversed(results)
    ]
    probs      = [p for _, p in reversed(results)]
    bar_colors = [
        info.get("color", "#888") if c == top_class else "#cccccc"
        for c, _ in reversed(results)
    ]

    bars = ax_bar.barh(class_names, probs,
                        color=bar_colors, height=0.5,
                        edgecolor="none")
    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel("Confidence", fontsize=9)
    ax_bar.set_title("Class probabilities", fontsize=9,
                      color="#555", pad=8)
    ax_bar.tick_params(axis="y", labelsize=9)
    ax_bar.tick_params(axis="x", labelsize=8)
    ax_bar.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    ax_bar.axvline(0.5, color="#aaaaaa", linestyle="--",
                   linewidth=0.7, alpha=0.7)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Value labels — only show if bar is wide enough, else place inside
    for bar, prob in zip(bars, probs):
        x_pos = prob + 0.02
        ha    = "left"
        # if bar nearly fills the axis, label would be clipped — move inside
        if prob > 0.90:
            x_pos = prob - 0.02
            ha    = "right"
        ax_bar.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center", ha=ha,
            fontsize=8.5, color="#333333"
        )

    # Fix 1 continued: title at y=1.0, tier text well below at y=0.93
    # so they never overlap even with a two-line title
    pred_label = info.get("full_name", top_class)
    title      = f"Prediction: {pred_label}  ({top_conf:.1%})"
    if true_class is not None:
        true_label = DISEASE_INFO.get(true_class, {}).get(
            "full_name", true_class
        )
        verdict    = "CORRECT" if correct else "WRONG"
        v_color    = "#27ae60" if correct else "#c0392b"
        title     += f"\nTrue class: {true_label}  [{verdict}]"
        pred_color = v_color

    # suptitle anchored above the subplots_adjust top boundary
    fig.suptitle(title, fontsize=12, fontweight="bold",
                 y=0.97, color=pred_color)

    # tier text sits just below suptitle, well clear of axes
    fig.text(0.5, 0.88, tier_label,
             ha="center", fontsize=9,
             color=tier_color, style="italic")

    # footer lines — anchored below subplots_adjust bottom boundary
    fig.text(
        0.5, 0.06,
        f"Model: {model_name}  |  "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ha="center", fontsize=7.5, color="#888888"
    )
    fig.text(
        0.5, 0.02,
        "AI screening only — not a substitute for "
        "professional ophthalmological diagnosis.",
        ha="center", fontsize=7.5, color="#c0392b", style="italic"
    )

    if out_dir is None:
        out_dir = CONFIG["PRED_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    base     = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{base}_prediction.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

############################################################
# SUMMARY GRID  (all images in one figure)
############################################################

def save_summary_grid(all_rows, out_dir):
    """
    Saves a single summary grid showing all predictions
    with correct/wrong labels — great for portfolios.
    """
    n        = len(all_rows)
    n_cols   = 4
    n_rows   = (n + n_cols - 1) // n_cols
    fig_w    = n_cols * 3.0
    fig_h    = n_rows * 3.5

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # Flatten axes safely
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, row in enumerate(all_rows):
        r      = i // n_cols
        c      = i % n_cols
        ax     = axes[r][c]

        img_bgr = cv2.imread(row["img_path"])
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (224, 224))
            ax.imshow(img_rgb)

        ax.axis("off")

        correct     = row["correct"]
        pred_label  = DISEASE_INFO.get(
            row["prediction"], {}
        ).get("full_name", row["prediction"])
        true_label  = DISEASE_INFO.get(
            row["true_class"], {}
        ).get("full_name", row["true_class"])
        conf        = row["confidence"]
        tick        = "+" if correct else "x"
        t_color     = "#1a7a4a" if correct else "#c0392b"

        ax.set_title(
            f"True: {true_label}\n"
            f"{tick} Pred: {pred_label} ({conf:.0%})",
            fontsize=7.5, color=t_color, pad=4
        )

    # Hide unused axes
    total_cells = n_rows * n_cols
    for j in range(n, total_cells):
        r = j // n_cols
        c = j % n_cols
        axes[r][c].axis("off")

    fig.suptitle(
        f"Prediction summary — {len(all_rows)} images  "
        f"({sum(r['correct'] for r in all_rows)}/{len(all_rows)} correct)",
        fontsize=11, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    path = os.path.join(out_dir, "prediction_summary_grid.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Summary grid -> {path}")
    return path

############################################################
# JSON RESULT
############################################################

def save_json_result(img_path, results, model_name,
                     out_dir, true_class=None):
    top_class, top_conf = results[0]
    info               = DISEASE_INFO.get(top_class, {})
    tier_label, _      = confidence_tier(top_conf)

    result = {
        "image":              os.path.basename(img_path),
        "model":              model_name,
        "timestamp":          datetime.now().isoformat(),
        "true_class":         true_class,
        "correct":            (top_class == true_class)
                              if true_class else None,
        "predicted_class":    top_class,
        "predicted_label":    info.get("full_name", top_class),
        "confidence":         round(top_conf, 4),
        "confidence_tier":    tier_label,
        "severity":           info.get("severity", ""),
        "description":        info.get("description", ""),
        "recommended_action": info.get("action", ""),
        "all_probabilities":  {c: round(p, 4) for c, p in results},
        "disclaimer": (
            "This is an AI screening tool only. "
            "It does not replace professional medical diagnosis. "
            "Always consult a qualified ophthalmologist."
        ),
    }

    os.makedirs(out_dir, exist_ok=True)
    base      = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(out_dir, f"{base}_result.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    return json_path

############################################################
# PROCESS ONE IMAGE
############################################################

def process_image(img_path, model, idx_to_class, layer_name,
                  model_name, gradcam, out_dir, true_class=None):
    try:
        img_rgb, img_array = preprocess_image(img_path)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None

    results = predict(model, img_array, idx_to_class)
    print_results(img_path, results, model_name, true_class)

    cam = None
    if gradcam:
        top_idx = list(idx_to_class.keys())[
            list(idx_to_class.values()).index(results[0][0])
        ]
        cam = compute_gradcam_pp(model, img_array, top_idx, layer_name)

    fig_path = save_prediction_figure(
        img_path, img_rgb, results, model_name,
        cam=cam, out_dir=out_dir, true_class=true_class
    )
    json_path = save_json_result(
        img_path, results, model_name, out_dir, true_class
    )

    print(f"  Figure -> {fig_path}")
    print(f"  JSON   -> {json_path}")

    top_class, top_conf = results[0]
    return {
        "img_path":   img_path,
        "image":      os.path.basename(img_path),
        "true_class": true_class,
        "prediction": top_class,
        "correct":    (top_class == true_class) if true_class else None,
        "confidence": round(top_conf, 4),
        "tier":       confidence_tier(top_conf)[0],
        **{c: round(p, 4) for c, p in results},
    }

############################################################
# RUN AUTO MODE  (3 random per class)
############################################################

def run_auto(model, idx_to_class, layer_name,
             model_name, gradcam, out_dir):
    images = pick_random_images(
        CONFIG["TEST_DIR"],
        n=CONFIG["N_RANDOM"],
        seed=CONFIG["RANDOM_SEED"]
    )

    sep = "=" * 60
    print(f"{sep}")
    print(f"  Auto mode: {len(images)} images selected")
    print(f"  Model    : {model_name}")
    print(f"  GradCAM  : {'Yes' if gradcam else 'No'}")
    print(f"{sep}")

    all_rows = []
    for i, (img_path, true_class) in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}]  true class: {true_class}")
        row = process_image(
            img_path, model, idx_to_class, layer_name,
            model_name, gradcam, out_dir, true_class
        )
        if row:
            all_rows.append(row)

    if not all_rows:
        return

    # CSV summary
    csv_path = os.path.join(out_dir, "random_sample_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[k for k in all_rows[0].keys()
                        if k != "img_path"]
        )
        writer.writeheader()
        for r in all_rows:
            row_out = {k: v for k, v in r.items() if k != "img_path"}
            writer.writerow(row_out)

    # Summary grid
    save_summary_grid(all_rows, out_dir)

    # Final report
    n_total   = len(all_rows)
    n_correct = sum(1 for r in all_rows if r["correct"])
    n_high    = sum(1 for r in all_rows
                    if r["confidence"] >= CONFIG["CONF_HIGH"])
    n_low     = sum(1 for r in all_rows
                    if r["confidence"] < CONFIG["CONF_MEDIUM"])

    print(f"\n{sep}")
    print(f"  RESULTS SUMMARY")
    print(sep)
    print(f"  Total images : {n_total}")
    print(f"  Correct      : {n_correct} / {n_total}  "
          f"({n_correct/n_total:.0%})")
    print(f"  High conf.   : {n_high}  (>= {CONFIG['CONF_HIGH']:.0%})")
    print(f"  Low conf.    : {n_low}   (< {CONFIG['CONF_MEDIUM']:.0%})")
    print(f"  CSV          : {csv_path}")
    print(sep)

    # Per-class breakdown
    classes = sorted(set(r["true_class"] for r in all_rows
                         if r["true_class"]))
    print(f"\n  Per-class breakdown:")
    for cls in classes:
        cls_rows  = [r for r in all_rows if r["true_class"] == cls]
        cls_right = sum(1 for r in cls_rows if r["correct"])
        cls_conf  = sum(r["confidence"] for r in cls_rows) / len(cls_rows)
        label     = DISEASE_INFO.get(cls, {}).get("full_name", cls)
        print(f"    {label:<25}  "
              f"{cls_right}/{len(cls_rows)} correct  "
              f"avg conf: {cls_conf:.1%}")
    print()

############################################################
# RUN SINGLE IMAGE
############################################################

def run_single(img_path, model, idx_to_class, layer_name,
               model_name, gradcam, out_dir):
    if not os.path.isfile(img_path):
        print(f"\nERROR: Image not found: {img_path}")
        sys.exit(1)
    process_image(img_path, model, idx_to_class, layer_name,
                  model_name, gradcam, out_dir)

############################################################
# RUN FOLDER
############################################################

def run_folder(folder_path, model, idx_to_class, layer_name,
               model_name, gradcam, out_dir):
    if not os.path.isdir(folder_path):
        print(f"\nERROR: Folder not found: {folder_path}")
        sys.exit(1)

    images = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(CONFIG["IMAGE_EXTENSIONS"])
    ])

    if not images:
        print(f"  No images found in: {folder_path}")
        return

    print(f"\n  Found {len(images)} image(s)\n")
    all_rows = []

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}]", end=" ")
        row = process_image(img_path, model, idx_to_class,
                            layer_name, model_name, gradcam, out_dir)
        if row:
            all_rows.append(row)

    if not all_rows:
        return

    csv_path = os.path.join(out_dir, "batch_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[k for k in all_rows[0].keys()
                        if k != "img_path"]
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: v for k, v in r.items()
                             if k != "img_path"})

    high = sum(1 for r in all_rows
               if r["confidence"] >= CONFIG["CONF_HIGH"])
    low  = sum(1 for r in all_rows
               if r["confidence"] < CONFIG["CONF_MEDIUM"])

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Batch complete  : {len(all_rows)} images")
    print(f"  High confidence : {high}  (>= {CONFIG['CONF_HIGH']:.0%})")
    print(f"  Low confidence  : {low}   (< {CONFIG['CONF_MEDIUM']:.0%})")
    print(f"  CSV             : {csv_path}")
    print(f"{sep}\n")

############################################################
# ARGUMENT PARSER
############################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Eye Disease Classification — Prediction",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py                     "
            "# auto: 3 random images per class\n"
            "  python predict.py --image retina.jpg  "
            "# single image\n"
            "  python predict.py --image retina.jpg --gradcam\n"
            "  python predict.py --folder ./images/\n"
            "  python predict.py --model InceptionV3\n"
            "  python predict.py --n 5               "
            "# 5 random per class\n"
        )
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--image",  type=str,
                       help="Path to a single fundus image.")
    group.add_argument("--folder", type=str,
                       help="Path to a folder of images (batch mode).")
    parser.add_argument("--model",   type=str, default=None,
                        choices=VALID_MODELS,
                        help=f"Model (default: {CONFIG['DEFAULT_MODEL']}).")
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate GradCAM++ heatmap.")
    parser.add_argument("--out",     type=str, default=None,
                        help="Output directory.")
    parser.add_argument("--n",       type=int, default=None,
                        help="Images per class in auto mode (default: 3).")
    parser.add_argument("--seed",    type=int, default=None,
                        help="Random seed for reproducible image selection.")
    return parser.parse_args()

############################################################
# MAIN
############################################################

def main():
    args = parse_args()

    # Apply CLI overrides to CONFIG
    if args.n    is not None:
        CONFIG["N_RANDOM"]    = args.n
    if args.seed is not None:
        CONFIG["RANDOM_SEED"] = args.seed

    model_name = args.model if args.model else CONFIG["DEFAULT_MODEL"]
    out_dir    = args.out   if args.out   else CONFIG["PRED_DIR"]
    gradcam    = args.gradcam

    os.makedirs(out_dir, exist_ok=True)

    # Load class indices
    try:
        idx_to_class = load_class_indices()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"\nClasses : {list(idx_to_class.values())}")

    # Load model
    model_path = os.path.join(
        CONFIG["MODEL_DIR"], f"{model_name}_best.keras"
    )
    if not os.path.exists(model_path):
        print(
            f"\nERROR: Model not found:\n  {model_path}\n"
            "Check MODEL_DIR in CONFIG."
        )
        sys.exit(1)

    print(f"Loading {model_name} ...")
    model      = load_model(model_path)
    layer_name = find_last_conv_layer(model)
    print(f"Ready.  Last conv layer: {layer_name}\n")

    # Dispatch
    if args.image:
        run_single(args.image, model, idx_to_class,
                   layer_name, model_name, gradcam, out_dir)
    elif args.folder:
        run_folder(args.folder, model, idx_to_class,
                   layer_name, model_name, gradcam, out_dir)
    else:
        # Default: auto mode — 3 random images per class
        run_auto(model, idx_to_class, layer_name,
                 model_name, gradcam, out_dir)


if __name__ == "__main__":
    main()
