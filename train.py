############################################################
# ENV / LOGGING SETUP (MUST BE BEFORE TF / PYLOT IMPORTS)
############################################################
import json
import os
import sys
import warnings
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

############################################################
# IMPORTS
############################################################

import cv2
import random
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    Callback
)
from tensorflow.keras.applications import (
    Xception,
    InceptionV3,
    DenseNet121,
    ResNet50V2,
    NASNetMobile
)

tf.get_logger().setLevel("ERROR")


############################################################
# CONFIG
############################################################

CONFIG = {
    "RAW_DATASET": "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/dataset",
    "PROCESSED_DATASET": "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/dataset_processed",
    "OUTPUT_DIR": "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/output",
    "MODEL_DIR": "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/output/models",
    "PLOT_DIR": "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/output/plots",
    "FULL_TERMINAL_LOG": "/media/dell-server/0441e02b-6b72-4551-a5a0-44018119a50d/output/full_terminal_output.txt",

    "IMAGE_SIZE": 224,
    "BATCH_SIZE": 16,
    "EPOCHS": 50,
    "LEARNING_RATE": 1e-4,
    "TRAIN_SPLIT": 0.70,
    "VAL_SPLIT": 0.15,
    "TEST_SPLIT": 0.15,
    "PATIENCE": 10,
    "SEED": 42,
    "FREEZE_BASE_MODEL": True,

    "IMAGE_EXTENSIONS": (
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    )
}


############################################################
# FULL TERMINAL OUTPUT LOGGER
############################################################

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


LOG_FILE_HANDLE = None
ORIGINAL_STDOUT = sys.stdout
ORIGINAL_STDERR = sys.stderr


def start_full_terminal_logging():
    global LOG_FILE_HANDLE

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    LOG_FILE_HANDLE = open(
        CONFIG["FULL_TERMINAL_LOG"],
        "w",
        buffering=1,
        encoding="utf-8"
    )

    sys.stdout = Tee(ORIGINAL_STDOUT, LOG_FILE_HANDLE)
    sys.stderr = Tee(ORIGINAL_STDERR, LOG_FILE_HANDLE)


def stop_full_terminal_logging():
    global LOG_FILE_HANDLE

    try:
        print("TERMINAL FULL OUTPUT SAVED")
    except Exception:
        pass

    sys.stdout = ORIGINAL_STDOUT
    sys.stderr = ORIGINAL_STDERR

    if LOG_FILE_HANDLE is not None:
        LOG_FILE_HANDLE.close()
        LOG_FILE_HANDLE = None


############################################################
# REPRODUCIBILITY
############################################################

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


############################################################
# GPU SETUP
############################################################

def setup_gpu():
    print("\nChecking GPU availability\n")

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print("GPUs detected:")
        for g in gpus:
            print(g.name)

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

        strategy = tf.distribute.MirroredStrategy()
    else:
        print("Running on CPU")
        strategy = tf.distribute.get_strategy()

    print("Replicas:", strategy.num_replicas_in_sync)
    return strategy


############################################################
# UTILITIES
############################################################

def create_directories():
    for d in [
        CONFIG["PROCESSED_DATASET"],
        CONFIG["OUTPUT_DIR"],
        CONFIG["MODEL_DIR"],
        CONFIG["PLOT_DIR"]
    ]:
        os.makedirs(d, exist_ok=True)


def list_images(folder):
    if not os.path.exists(folder):
        return []

    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(CONFIG["IMAGE_EXTENSIONS"])
    ]


def get_model_plot_dir(model_name):
    model_plot_dir = os.path.join(CONFIG["PLOT_DIR"], model_name)
    os.makedirs(model_plot_dir, exist_ok=True)
    return model_plot_dir


############################################################
# CHECK IF DATASET ALREADY PROCESSED
############################################################

def processed_dataset_exists():
    train_dir = os.path.join(CONFIG["PROCESSED_DATASET"], "train")

    if not os.path.exists(train_dir):
        return False

    for _, _, files in os.walk(train_dir):
        if len(files) > 10:
            return True

    return False


############################################################
# REMOVE BLACK BORDER
############################################################

def remove_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]

    if cropped.size == 0:
        return img

    return cropped


############################################################
# DATASET INSPECTION
############################################################

def inspect_dataset():
    print("\nDataset statistics\n")

    classes = []

    if not os.path.exists(CONFIG["RAW_DATASET"]):
        raise FileNotFoundError(f"RAW_DATASET not found: {CONFIG['RAW_DATASET']}")

    for disease in sorted(os.listdir(CONFIG["RAW_DATASET"])):
        path = os.path.join(CONFIG["RAW_DATASET"], disease)

        if os.path.isdir(path):
            imgs = list_images(path)
            classes.append(disease)
            print(f"{disease}: {len(imgs)}")

    if not classes:
        raise ValueError("No class folders found in RAW_DATASET.")

    total_split = CONFIG["TRAIN_SPLIT"] + CONFIG["VAL_SPLIT"] + CONFIG["TEST_SPLIT"]
    if not np.isclose(total_split, 1.0):
        raise ValueError("TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must equal 1.0")

    return classes


############################################################
# SPLIT DATASET
############################################################

def split_dataset(classes):
    print("\nSplitting dataset...\n")

    for disease in classes:
        src = os.path.join(CONFIG["RAW_DATASET"], disease)
        images = list_images(src)

        if len(images) == 0:
            print(f"Warning: no images found in class '{disease}'")
            continue

        random.shuffle(images)

        n = len(images)
        train_end = int(n * CONFIG["TRAIN_SPLIT"])
        val_end = int(n * (CONFIG["TRAIN_SPLIT"] + CONFIG["VAL_SPLIT"]))

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        splits = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs
        }

        for split_name, split_data in splits.items():
            dest = os.path.join(CONFIG["PROCESSED_DATASET"], split_name, disease)
            os.makedirs(dest, exist_ok=True)

            for img in split_data:
                src_path = os.path.join(src, img)
                dst_path = os.path.join(dest, img)

                if not os.path.exists(dst_path):
                    import shutil
                    shutil.copy(src_path, dst_path)


############################################################
# PREPROCESS IMAGES
############################################################

def preprocess_images(classes):
    print("\nPreprocessing images...\n")

    size = CONFIG["IMAGE_SIZE"]

    for split in ["train", "val", "test"]:
        for disease in classes:
            folder = os.path.join(CONFIG["PROCESSED_DATASET"], split, disease)

            for img_name in list_images(folder):
                path = os.path.join(folder, img_name)

                img = cv2.imread(path)
                if img is None:
                    print(f"Warning: failed to read {path}")
                    continue

                img = remove_black_border(img)
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path, img)


############################################################
# DATA GENERATORS
############################################################

def build_generators():
    size = CONFIG["IMAGE_SIZE"]
    batch_size = CONFIG["BATCH_SIZE"]
    seed = CONFIG["SEED"]

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train = train_datagen.flow_from_directory(
        os.path.join(CONFIG["PROCESSED_DATASET"], "train"),
        target_size=(size, size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed
    )

    val = eval_datagen.flow_from_directory(
        os.path.join(CONFIG["PROCESSED_DATASET"], "val"),
        target_size=(size, size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test = eval_datagen.flow_from_directory(
        os.path.join(CONFIG["PROCESSED_DATASET"], "test"),
        target_size=(size, size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train, val, test


############################################################
# MODEL BUILDER
############################################################

def build_model(base_model, num_classes):
    base_model.trainable = not CONFIG["FREEZE_BASE_MODEL"]

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["LEARNING_RATE"]),
        loss="categorical_crossentropy",
        metrics=["accuracy", AUC(name="auc")]
    )

    return model


############################################################
# CUSTOM CALLBACK
############################################################

class PrintLR(Callback):
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = None

        if hasattr(optimizer, "learning_rate"):
            try:
                lr = tf.keras.backend.get_value(optimizer.learning_rate)
            except Exception:
                lr = optimizer.learning_rate
        elif hasattr(optimizer, "lr"):
            try:
                lr = tf.keras.backend.get_value(optimizer.lr)
            except Exception:
                lr = optimizer.lr

        if lr is not None:
            print(f"Learning Rate: {float(lr):.8f}")
        else:
            print("Learning Rate: unavailable")


############################################################
# SAFE PLOTTING HELPERS
############################################################

def save_confusion_matrix(cm, labels, path, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true_bin, y_prob, labels, path, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{labels[i]} AUC = {roc_auc:.3f}")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_history_plot(values1, values2, label1, label2, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(values1, label=label1)
    ax.plot(values2, label=label2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


############################################################
# EVALUATION
############################################################

def evaluate_model(name, model, test):
    print(f"\nEvaluating {name}\n")

    test.reset()
    y_true = test.classes
    y_prob = model.predict(test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    labels = list(test.class_indices.keys())
    model_plot_dir = get_model_plot_dir(name)

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(
        cm=cm,
        labels=labels,
        path=os.path.join(model_plot_dir, f"{name}_confusion_matrix.pdf"),
        title=f"{name} Confusion Matrix"
    )

    report_text = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(report_text)

    y_true_bin = label_binarize(y_true, classes=range(len(labels)))
    save_roc_curve(
        y_true_bin=y_true_bin,
        y_prob=y_prob,
        labels=labels,
        path=os.path.join(model_plot_dir, f"{name}_roc_curve.pdf"),
        title=f"{name} ROC Curve"
    )

    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_auc = []
    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        per_class_auc.append(auc(fpr, tpr))

    mean_auc = float(np.mean(per_class_auc))

    return {
        "test_accuracy": float(acc),
        "test_precision_macro": float(precision_macro),
        "test_recall_macro": float(recall_macro),
        "test_f1_macro": float(f1_macro),
        "test_precision_weighted": float(precision_weighted),
        "test_recall_weighted": float(recall_weighted),
        "test_f1_weighted": float(f1_weighted),
        "test_mean_auc_ovr": mean_auc
    }


############################################################
# TRAINING HISTORY PLOTS
############################################################

def plot_training_history(name, history):
    history_dict = history.history
    model_plot_dir = get_model_plot_dir(name)

    if "accuracy" in history_dict and "val_accuracy" in history_dict:
        save_history_plot(
            history_dict["accuracy"],
            history_dict["val_accuracy"],
            "train_accuracy",
            "val_accuracy",
            f"{name} Accuracy",
            "Accuracy",
            os.path.join(model_plot_dir, f"{name}_accuracy.pdf")
        )

    if "loss" in history_dict and "val_loss" in history_dict:
        save_history_plot(
            history_dict["loss"],
            history_dict["val_loss"],
            "train_loss",
            "val_loss",
            f"{name} Loss",
            "Loss",
            os.path.join(model_plot_dir, f"{name}_loss.pdf")
        )

    if "auc" in history_dict and "val_auc" in history_dict:
        save_history_plot(
            history_dict["auc"],
            history_dict["val_auc"],
            "train_auc",
            "val_auc",
            f"{name} AUC",
            "AUC",
            os.path.join(model_plot_dir, f"{name}_auc.pdf")
        )


############################################################
# RESULTS SUMMARY
############################################################

def build_result_row(name, history, eval_metrics):
    history_dict = history.history
    best_epoch_idx = int(np.argmax(history_dict["val_auc"]))

    row = {
        "model_name": name,
        "best_epoch": best_epoch_idx + 1,
        "best_val_auc": float(history_dict["val_auc"][best_epoch_idx]),
        "best_val_accuracy": float(history_dict["val_accuracy"][best_epoch_idx]),
        "best_val_loss": float(history_dict["val_loss"][best_epoch_idx]),
        "train_auc_at_best_epoch": float(history_dict["auc"][best_epoch_idx]),
        "train_accuracy_at_best_epoch": float(history_dict["accuracy"][best_epoch_idx]),
        "train_loss_at_best_epoch": float(history_dict["loss"][best_epoch_idx]),
    }

    row.update(eval_metrics)
    return row


def save_results_summary(results):
    df = pd.DataFrame(results)
    df = df.sort_values(by="best_val_auc", ascending=False).reset_index(drop=True)

    summary_csv_path = os.path.join(CONFIG["OUTPUT_DIR"], "results_summary.csv")
    df.to_csv(summary_csv_path, index=False)

    print("FINAL RESULTS SUMMARY")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    print(f"\nResults summary saved to: {summary_csv_path}\n")

    return df


############################################################
# TRAINING
############################################################

def train_models(strategy, train, val, test):
    num_classes = train.num_classes
    input_shape = (CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"], 3)

    model_factories = {
        "Xception": lambda: Xception(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        ),
        "InceptionV3": lambda: InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        ),
        "DenseNet121": lambda: DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        ),
        "ResNet50V2": lambda: ResNet50V2(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        ),
        "NASNetMobile": lambda: NASNetMobile(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape
        )
    }

    results = []

    for name, factory in model_factories.items():
        print(f"\nTraining {name}\n")

        best_model_path = os.path.join(CONFIG["MODEL_DIR"], f"{name}_best.keras")

        with strategy.scope():
            base = factory()
            model = build_model(base, num_classes)

        checkpoint = ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        early = EarlyStopping(
            monitor="val_auc",
            patience=CONFIG["PATIENCE"],
            mode="max",
            restore_best_weights=True,
            verbose=1
        )

        reduce = ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.3,
            patience=5,
            min_lr=1e-6,
            mode="max",
            verbose=1
        )

        history = model.fit(
            train,
            validation_data=val,
            epochs=CONFIG["EPOCHS"],
            callbacks=[checkpoint, early, reduce, PrintLR()],
            verbose=2
        )

        plot_training_history(name, history)

        best_model = load_model(best_model_path)
        eval_metrics = evaluate_model(name, best_model, test)

        result_row = build_result_row(name, history, eval_metrics)
        results.append(result_row)

        del model
        del best_model
        tf.keras.backend.clear_session()

    save_results_summary(results)


############################################################
# MAIN
############################################################

def main():
    create_directories()
    start_full_terminal_logging()

    try:
        set_seed(CONFIG["SEED"])
        strategy = setup_gpu()

        classes = inspect_dataset()

        if processed_dataset_exists():
            print("\nProcessed dataset already exists — skipping preprocessing.\n")
        else:
            split_dataset(classes)
            preprocess_images(classes)

        train, val, test = build_generators()
        class_indices_path = os.path.join(CONFIG["OUTPUT_DIR"], "class_indices.json")
        with open(class_indices_path, "w", encoding="utf-8") as f:
            json.dump(train.class_indices, f, indent=2, sort_keys=True)
        print(f"Saved class_indices.json to: {class_indices_path}")
        print(f"class_indices: {train.class_indices}")
        train_models(strategy, train, val, test)

    except Exception as e:
        print("UNHANDLED EXCEPTION")
        print(str(e))
        print(traceback.format_exc())
        raise

    finally:
        stop_full_terminal_logging()


if __name__ == "__main__":
    main()
