############################################################
# GRAD-CAM ANALYSIS v2 — Eye Disease Classification
# Improvements over v1:
#   1. GradCAM++ (sharper, more localised heatmaps)
#   2. Confidence % shown on every grid cell
#   3. Optic disc crop preprocessing for glaucoma
#   4. Cleaner grid layout with class confidence summary
#   5. Side-by-side v1 vs v2 comparison output
############################################################

import os
import json
import warnings
import logging

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.get_logger().setLevel("ERROR")

############################################################
# CONFIG
############################################################

CONFIG = {
    "PROCESSED_DATASET": "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/dataset_processed",
    "MODEL_DIR":         "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/output/models",
    "OUTPUT_DIR":        "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/output",
    "GRADCAM_DIR":       "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/output/gradcam",
    "IMAGE_SIZE":        224,
    "BATCH_SIZE":        16,

    "MODELS_TO_ANALYZE": ["ResNet50V2"],  # or None for all

    # Samples per class in the main grid
    "N_PER_CLASS": 4,

    # Save individual overlay PNGs per sample
    "SAVE_INDIVIDUAL": True,

    # Optic disc crop: tight crop around the brightest region
    # Applied only when visualising glaucoma samples
    "OPTIC_DISC_CROP": True,
    "OPTIC_DISC_MARGIN": 60,   # pixels around detected disc centre

    # GradCAM++ vs standard GradCAM
    "USE_GRADCAM_PLUS_PLUS": True,

    # Heatmap blend alpha (0=original only, 1=heatmap only)
    "OVERLAY_ALPHA": 0.45,
}

############################################################
# OPTIC DISC DETECTION + CROP
############################################################

def detect_optic_disc_centre(img_rgb):
    """
    Rough optic disc localisation via the brightest region in the green
    channel (the disc appears as a bright circular region).
    Returns (cx, cy) pixel centre, or None if detection is unreliable.
    """
    green = img_rgb[:, :, 1].astype(np.float32)

    # Blur heavily to suppress vessels and noise
    blurred = cv2.GaussianBlur(green, (51, 51), 0)

    # Find brightest point
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    cx, cy = max_loc  # OpenCV returns (col, row)

    h, w = img_rgb.shape[:2]

    # Sanity check: reject if centre is too close to the image edge
    margin = 0.1
    if (cx < w * margin or cx > w * (1 - margin) or
            cy < h * margin or cy > h * (1 - margin)):
        return None

    return cx, cy


def crop_optic_disc(img_rgb, margin=None):
    """
    Returns a square crop centred on the detected optic disc.
    Falls back to the full image if detection fails.
    """
    if margin is None:
        margin = CONFIG["OPTIC_DISC_MARGIN"]

    centre = detect_optic_disc_centre(img_rgb)
    if centre is None:
        return img_rgb, False

    cx, cy = centre
    h, w = img_rgb.shape[:2]

    x1 = max(0, cx - margin)
    y1 = max(0, cy - margin)
    x2 = min(w, cx + margin)
    y2 = min(h, cy + margin)

    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return img_rgb, False

    return crop, True


############################################################
# FIND LAST CONV LAYER
############################################################

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


############################################################
# GRADCAM++ CORE
############################################################

def compute_gradcam_pp(model, img_array, class_idx, layer_name):
    """
    GradCAM++ — sharper and more localised than standard GradCAM.
    Weights each spatial location by the second-order gradient importance.

    img_array : float32 (1, H, W, 3) in [0, 1]
    Returns   : float32 (H, W) heatmap in [0, 1]
    """
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            conv_outputs, predictions = grad_model(img_array, training=False)
            loss = predictions[:, class_idx]
        grads = tape1.gradient(loss, conv_outputs)       # 1st order
    grads2 = tape2.gradient(grads, conv_outputs)         # 2nd order

    conv_np  = conv_outputs[0].numpy()   # (h, w, C)
    grads_np = grads[0].numpy()          # (h, w, C)
    grads2_np = grads2[0].numpy()        # (h, w, C)

    # GradCAM++ alpha weights
    # alpha = grads^2 / (2*grads^2 + sum(A * grads^3) + eps)
    numerator   = grads_np ** 2
    denominator = (2.0 * grads_np ** 2
                   + np.sum(conv_np * grads_np ** 3,
                             axis=(0, 1), keepdims=True)
                   + 1e-8)
    alphas = numerator / denominator          # (h, w, C)

    # Weight by ReLU of gradients
    weights = np.sum(alphas * np.maximum(grads_np, 0),
                     axis=(0, 1))             # (C,)

    cam = np.sum(weights * conv_np, axis=-1)  # (h, w)
    cam = np.maximum(cam, 0)

    # Resize to input resolution
    size = img_array.shape[2]
    cam = cv2.resize(cam, (size, img_array.shape[1]))

    # Normalise
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32)


def compute_gradcam(model, img_array, class_idx, layer_name):
    """Standard GradCAM (fallback if GradCAM++ is disabled)."""
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_map = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_map), axis=-1)
    cam = np.maximum(cam.numpy(), 0)
    cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
    return cam.astype(np.float32)


def get_cam(model, img_array, class_idx, layer_name):
    if CONFIG["USE_GRADCAM_PLUS_PLUS"]:
        return compute_gradcam_pp(model, img_array, class_idx, layer_name)
    return compute_gradcam(model, img_array, class_idx, layer_name)


############################################################
# OVERLAY HELPER
############################################################

def overlay_heatmap(original_rgb, cam, alpha=None):
    if alpha is None:
        alpha = CONFIG["OVERLAY_ALPHA"]
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_rgb + (1 - alpha) * original_rgb).astype(np.uint8)
    return blended


############################################################
# LOAD IMAGE HELPER
############################################################

def load_image(img_path, size=None):
    """Returns uint8 RGB, resized to (size, size)."""
    if size is None:
        size = CONFIG["IMAGE_SIZE"]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size))
    return img_resized


def preprocess(img_rgb):
    """uint8 (H,W,3) → float32 (1,H,W,3) in [0,1]."""
    return np.expand_dims(img_rgb / 255.0, axis=0).astype(np.float32)


############################################################
# COLLECT SAMPLES
############################################################

def collect_samples(test_gen, classes, n_per_class):
    samples = {cls: [] for cls in classes}
    for path, label_idx in zip(test_gen.filepaths, test_gen.classes):
        cls = classes[label_idx]
        if len(samples[cls]) < n_per_class:
            samples[cls].append((path, int(label_idx)))
    return samples


############################################################
# 1. MAIN GRAD-CAM GRID  (with confidence %)
############################################################

def save_gradcam_grid(model_name, model, test_gen, classes,
                      layer_name, out_dir):
    """
    Grid: rows = classes, columns = N_PER_CLASS × (original | overlay)
    Each overlay subtitle now shows: [+/-] pred: <class> (XX%)
    Glaucoma samples optionally show an optic-disc crop inset.
    """
    n        = CONFIG["N_PER_CLASS"]
    n_cls    = len(classes)
    size     = CONFIG["IMAGE_SIZE"]
    n_cols   = n * 2

    test_gen.reset()
    samples  = collect_samples(test_gen, classes, n)

    test_gen.reset()
    y_prob   = model.predict(test_gen, verbose=0)
    y_pred   = np.argmax(y_prob, axis=1)
    pred_map = dict(zip(test_gen.filepaths, zip(y_pred, y_prob)))

    fig = plt.figure(figsize=(n_cols * 2.4, n_cls * 2.8))
    fig.patch.set_facecolor("white")
    outer = gridspec.GridSpec(n_cls, n_cols, figure=fig,
                              hspace=0.4, wspace=0.06)

    for row_idx, cls in enumerate(classes):
        for col_pair, (img_path, label_idx) in enumerate(samples[cls]):

            img_rgb   = load_image(img_path, size)
            img_array = preprocess(img_rgb)

            # Optic disc crop for glaucoma — show inset below overlay
            disc_crop = None
            if CONFIG["OPTIC_DISC_CROP"] and cls == "glaucoma":
                crop, found = crop_optic_disc(img_rgb)
                if found:
                    disc_crop = cv2.resize(crop, (size // 2, size // 2))

            cam     = get_cam(model, img_array, label_idx, layer_name)
            overlay = overlay_heatmap(img_rgb, cam)

            pred_idx  = pred_map[img_path][0]
            prob_vec  = pred_map[img_path][1]
            correct   = (pred_idx == label_idx)
            pred_name = classes[pred_idx]
            conf      = prob_vec[pred_idx]
            tick      = "+" if correct else "✗"
            color     = "#1a7a4a" if correct else "#c0392b"

            # --- original ---
            ax_o = fig.add_subplot(outer[row_idx, col_pair * 2])
            ax_o.imshow(img_rgb)
            ax_o.axis("off")
            if col_pair == 0:
                ax_o.set_ylabel(
                    cls.replace("_", "\n"),
                    fontsize=9, fontweight="bold",
                    rotation=90, labelpad=6, va="center"
                )
            if row_idx == 0:
                ax_o.set_title("original", fontsize=7,
                               color="#888", pad=3)

            # --- overlay ---
            ax_c = fig.add_subplot(outer[row_idx, col_pair * 2 + 1])
            ax_c.imshow(overlay)
            ax_c.axis("off")
            ax_c.set_title(
                f"{tick} pred: {pred_name}\n({conf:.0%})",
                fontsize=7, color=color, pad=3
            )

            # Optic disc crop inset (bottom-right corner of overlay axes)
            if disc_crop is not None:
                inset_ax = ax_c.inset_axes([0.6, 0.0, 0.4, 0.4])
                inset_ax.imshow(disc_crop)
                inset_ax.axis("off")
                for spine in inset_ax.spines.values():
                    spine.set_edgecolor("yellow")
                    spine.set_linewidth(1.2)

            # Save individual
            if CONFIG["SAVE_INDIVIDUAL"]:
                ind_dir = os.path.join(out_dir, "individual",
                                       model_name, cls)
                os.makedirs(ind_dir, exist_ok=True)
                base    = os.path.splitext(os.path.basename(img_path))[0]
                cv2.imwrite(
                    os.path.join(ind_dir, f"{base}_gradcam.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                )

    method = "GradCAM++" if CONFIG["USE_GRADCAM_PLUS_PLUS"] else "GradCAM"
    fig.suptitle(
        f"{method} — {model_name}   "
        f"(green = correct prediction, red = wrong)\n"
        f"Glaucoma rows include optic-disc crop inset (yellow box)",
        fontsize=10, fontweight="bold", y=1.01
    )

    for ext, dpi in [(".pdf", 300), (".png", 150)]:
        path = os.path.join(out_dir, f"{model_name}_gradcam_grid{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved grid  -> {path}")

    plt.close(fig)


############################################################
# 2. FAILURE CASES  (with confidence %)
############################################################

def save_failure_cases(model_name, model, test_gen, classes,
                       layer_name, out_dir, n=6):
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    wrong_idx  = np.where(y_pred != y_true)[0]
    wrong_conf = y_prob[wrong_idx, y_pred[wrong_idx]]
    top        = wrong_idx[np.argsort(-wrong_conf)[:n]]

    if len(top) == 0:
        print(f"  No failures for {model_name} — skipping.")
        return

    size   = CONFIG["IMAGE_SIZE"]
    n_show = len(top)
    fig, axes = plt.subplots(2, n_show,
                             figsize=(n_show * 2.4, 6.0))
    fig.subplots_adjust(hspace=0.45)
    fig.patch.set_facecolor("white")
    method = "GradCAM++" if CONFIG["USE_GRADCAM_PLUS_PLUS"] else "GradCAM"
    fig.suptitle(
        f"{model_name} — top {n_show} most confident wrong predictions  [{method}]",
        fontsize=10, fontweight="bold"
    )

    for col, idx in enumerate(top):
        img_path  = test_gen.filepaths[idx]
        true_cls  = classes[y_true[idx]]
        pred_cls  = classes[y_pred[idx]]
        conf      = y_prob[idx, y_pred[idx]]
        true_conf = y_prob[idx, y_true[idx]]

        img_rgb   = load_image(img_path, size)
        img_array = preprocess(img_rgb)

        cam     = get_cam(model, img_array, int(y_pred[idx]), layer_name)
        overlay = overlay_heatmap(img_rgb, cam)

        axes[0, col].imshow(img_rgb)
        axes[0, col].axis("off")
        axes[0, col].set_title(
            f"true: {true_cls}\n(model conf: {true_conf:.0%})",
            fontsize=7, color="#1a7a4a"
        )

        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")
        axes[1, col].set_title(
            f"pred: {pred_cls} ({conf:.0%})",
            fontsize=7, color="#c0392b"
        )

    fig.tight_layout()
    for ext, dpi in [(".pdf", 300), (".png", 150)]:
        path = os.path.join(out_dir, f"{model_name}_failure_cases{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved failures -> {path}")
    plt.close(fig)


############################################################
# 3. PER-CLASS CONFIDENCE PLOT  (improved styling)
############################################################

def save_confidence_plot(model_name, model, test_gen, classes, out_dir):
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=0)
    y_true = test_gen.classes

    # Mean confidence for correct predictions only vs all predictions
    mean_all     = []
    mean_correct = []
    for i, cls in enumerate(classes):
        mask         = y_true == i
        correct_mask = mask & (np.argmax(y_prob, axis=1) == i)
        mean_all.append(float(np.mean(y_prob[mask, i])) if mask.sum() > 0 else 0)
        mean_correct.append(
            float(np.mean(y_prob[correct_mask, i]))
            if correct_mask.sum() > 0 else 0
        )

    x      = np.arange(len(classes))
    width  = 0.38
    colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width / 2, mean_all,     width,
                   color=colors[:len(classes)], alpha=0.55,
                   label="All samples", zorder=3)
    bars2 = ax.bar(x + width / 2, mean_correct, width,
                   color=colors[:len(classes)], alpha=1.0,
                   label="Correct predictions only", zorder=3)

    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("Mean confidence (true class probability)", fontsize=10)
    ax.set_title(
        f"{model_name} — per-class confidence on test set",
        fontsize=11, fontweight="bold"
    )
    ax.axhline(0.5, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6, zorder=2)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    for bar, val in zip(list(bars1) + list(bars2),
                        mean_all + mean_correct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8
        )

    fig.tight_layout()
    path = os.path.join(out_dir, f"{model_name}_confidence.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confidence -> {path}")


############################################################
# 4. GRADCAM vs GRADCAM++ COMPARISON  (new in v2)
############################################################

def save_method_comparison(model_name, model, test_gen, classes,
                            layer_name, out_dir, n_samples=3):
    """
    Side-by-side: original | GradCAM | GradCAM++ for n_samples per class.
    Shows clearly why GradCAM++ produces sharper localisation.
    """
    size    = CONFIG["IMAGE_SIZE"]
    n_cls   = len(classes)
    n_cols  = n_samples * 3   # original + v1 + v2

    test_gen.reset()
    samples = collect_samples(test_gen, classes, n_samples)

    test_gen.reset()
    y_prob   = model.predict(test_gen, verbose=0)
    y_pred   = np.argmax(y_prob, axis=1)
    pred_map = dict(zip(test_gen.filepaths, zip(y_pred, y_prob)))

    fig = plt.figure(figsize=(n_cols * 2.2, n_cls * 2.6))
    fig.patch.set_facecolor("white")
    outer = gridspec.GridSpec(n_cls, n_cols, figure=fig,
                              hspace=0.4, wspace=0.06)

    col_headers = (["original", "GradCAM", "GradCAM++"] * n_samples)

    for row_idx, cls in enumerate(classes):
        for col_sample, (img_path, label_idx) in enumerate(samples[cls]):

            img_rgb   = load_image(img_path, size)
            img_array = preprocess(img_rgb)

            cam_v1 = compute_gradcam(model, img_array, label_idx, layer_name)
            cam_v2 = compute_gradcam_pp(model, img_array, label_idx, layer_name)

            ov_v1  = overlay_heatmap(img_rgb, cam_v1)
            ov_v2  = overlay_heatmap(img_rgb, cam_v2)

            pred_idx = pred_map[img_path][0]
            conf     = pred_map[img_path][1][pred_idx]
            correct  = (pred_idx == label_idx)
            tick     = "+" if correct else "✗"
            color    = "#1a7a4a" if correct else "#c0392b"

            base_col = col_sample * 3

            # original
            ax0 = fig.add_subplot(outer[row_idx, base_col])
            ax0.imshow(img_rgb)
            ax0.axis("off")
            if col_sample == 0:
                ax0.set_ylabel(cls.replace("_", "\n"),
                               fontsize=9, fontweight="bold",
                               rotation=90, labelpad=6, va="center")
            if row_idx == 0:
                ax0.set_title("original", fontsize=7, color="#888", pad=3)

            # GradCAM v1
            ax1 = fig.add_subplot(outer[row_idx, base_col + 1])
            ax1.imshow(ov_v1)
            ax1.axis("off")
            if row_idx == 0:
                ax1.set_title("GradCAM", fontsize=7, color="#555", pad=3)

            # GradCAM++
            ax2 = fig.add_subplot(outer[row_idx, base_col + 2])
            ax2.imshow(ov_v2)
            ax2.axis("off")
            ax2.set_title(
                f"{tick} {conf:.0%}",
                fontsize=7, color=color, pad=3
            )
            if row_idx == 0:
                # Add method label above the confidence tick
                ax2.set_title(
                    f"GradCAM++\n{tick} {conf:.0%}",
                    fontsize=7, color=color, pad=3
                )

    fig.suptitle(
        f"{model_name} — GradCAM vs GradCAM++ comparison\n"
        "GradCAM++ produces sharper, more localised activation maps",
        fontsize=10, fontweight="bold", y=1.01
    )

    for ext, dpi in [(".pdf", 300), (".png", 150)]:
        path = os.path.join(out_dir, f"{model_name}_method_comparison{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved comparison -> {path}")
    plt.close(fig)


############################################################
# 5. OPTIC DISC CROP ANALYSIS  (glaucoma-specific, new in v2)
############################################################

def save_optic_disc_analysis(model_name, model, test_gen, classes,
                              layer_name, out_dir, n=6):
    """
    For glaucoma samples only: show
      original | full Grad-CAM | disc crop | disc Grad-CAM
    Highlights whether the model is correctly focusing on the disc.
    """
    if "glaucoma" not in classes:
        return

    glaucoma_idx = classes.index("glaucoma")
    size         = CONFIG["IMAGE_SIZE"]

    glaucoma_samples = [
        (p, int(l))
        for p, l in zip(test_gen.filepaths, test_gen.classes)
        if l == glaucoma_idx
    ][:n]

    test_gen.reset()
    y_prob   = model.predict(test_gen, verbose=0)
    y_pred   = np.argmax(y_prob, axis=1)
    pred_map = dict(zip(test_gen.filepaths, zip(y_pred, y_prob)))

    fig, axes = plt.subplots(n, 4, figsize=(10, n * 2.5))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.patch.set_facecolor("white")

    col_labels = ["Original", "Full GradCAM++",
                  "Optic disc crop", "Disc GradCAM++"]
    for col, lbl in enumerate(col_labels):
        axes[0, col].set_title(lbl, fontsize=9, fontweight="bold", pad=4)

    for row, (img_path, label_idx) in enumerate(glaucoma_samples):
        img_rgb   = load_image(img_path, size)
        img_array = preprocess(img_rgb)

        # Full image CAM
        cam_full    = get_cam(model, img_array, label_idx, layer_name)
        overlay_full = overlay_heatmap(img_rgb, cam_full)

        # Optic disc crop
        crop, found = crop_optic_disc(img_rgb,
                                      CONFIG["OPTIC_DISC_MARGIN"])
        disc_rgb    = cv2.resize(crop, (size, size)) if found else img_rgb

        disc_array  = preprocess(disc_rgb)
        cam_disc    = get_cam(model, disc_array, label_idx, layer_name)
        overlay_disc = overlay_heatmap(disc_rgb, cam_disc)

        pred_idx  = pred_map[img_path][0]
        conf      = pred_map[img_path][1][pred_idx]
        correct   = (pred_idx == label_idx)
        pred_name = classes[pred_idx]
        tick      = "+" if correct else "✗"
        color     = "#1a7a4a" if correct else "#c0392b"

        for col, im in enumerate([img_rgb, overlay_full,
                                   disc_rgb, overlay_disc]):
            axes[row, col].imshow(im)
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(
            f"sample {row+1}", fontsize=8, rotation=90,
            labelpad=4, va="center"
        )
        axes[row, 1].set_xlabel(
            f"{tick} pred: {pred_name} ({conf:.0%})",
            fontsize=8, color=color, labelpad=3
        )

    fig.suptitle(
        f"{model_name} — Glaucoma optic disc analysis\n"
        "Cropping around the disc forces the model to focus on "
        "the cup-to-disc ratio",
        fontsize=10, fontweight="bold"
    )

    fig.tight_layout()
    for ext, dpi in [(".pdf", 300), (".png", 150)]:
        path = os.path.join(out_dir, f"{model_name}_optic_disc{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved optic disc -> {path}")
    plt.close(fig)


############################################################
# BUILD TEST GENERATOR
############################################################

def build_test_generator():
    size    = CONFIG["IMAGE_SIZE"]
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return datagen.flow_from_directory(
        os.path.join(CONFIG["PROCESSED_DATASET"], "test"),
        target_size=(size, size),
        batch_size=CONFIG["BATCH_SIZE"],
        class_mode="categorical",
        shuffle=False
    )


############################################################
# MAIN
############################################################

def main():
    os.makedirs(CONFIG["GRADCAM_DIR"], exist_ok=True)

    class_indices_path = os.path.join(CONFIG["OUTPUT_DIR"], "class_indices.json")
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(
            f"class_indices.json not found at {class_indices_path}"
        )

    with open(class_indices_path) as f:
        class_indices = json.load(f)
    classes = [k for k, _ in sorted(class_indices.items(), key=lambda x: x[1])]
    print(f"Classes : {classes}")

    method = "GradCAM++" if CONFIG["USE_GRADCAM_PLUS_PLUS"] else "GradCAM"
    print(f"Method  : {method}\n")

    models_to_run = CONFIG["MODELS_TO_ANALYZE"]
    if models_to_run is None:
        models_to_run = [
            f.replace("_best.keras", "")
            for f in sorted(os.listdir(CONFIG["MODEL_DIR"]))
            if f.endswith("_best.keras")
        ]

    for model_name in models_to_run:
        model_path = os.path.join(CONFIG["MODEL_DIR"],
                                   f"{model_name}_best.keras")
        if not os.path.exists(model_path):
            print(f"WARNING: {model_path} not found — skipping.")
            continue

        print(f"{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")

        model      = load_model(model_path)
        layer_name = find_last_conv_layer(model)
        print(f"  Conv layer : {layer_name}")

        test_gen = build_test_generator()
        out_dir  = CONFIG["GRADCAM_DIR"]

        print("  [1/4] Grad-CAM grid...")
        save_gradcam_grid(model_name, model, test_gen, classes,
                          layer_name, out_dir)

        print("  [2/4] Failure cases...")
        save_failure_cases(model_name, model, test_gen, classes,
                           layer_name, out_dir)

        print("  [3/4] Confidence plot...")
        save_confidence_plot(model_name, model, test_gen,
                             classes, out_dir)

        print("  [4/4] Method comparison + optic disc analysis...")
        save_method_comparison(model_name, model, test_gen, classes,
                               layer_name, out_dir, n_samples=3)
        save_optic_disc_analysis(model_name, model, test_gen, classes,
                                 layer_name, out_dir, n=6)

        del model
        tf.keras.backend.clear_session()
        print()

    print("All outputs saved to:")
    print(f"  {CONFIG['GRADCAM_DIR']}")


if __name__ == "__main__":
    main()
