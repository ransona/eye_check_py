import sys
import os
import getpass
import json
import time
import traceback
import pickle
import shutil
import cv2
import numpy as np

# PyQt5 imports
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSlider, QMessageBox, QProgressDialog, QShortcut,
    QDialog, QListWidget, QComboBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence

# Matplotlib integration in PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector

# Import your custom module that provides file paths.
import organise_paths


class VideoDisplayLabel(QLabel):
    def __init__(self, side_name, parent=None):
        super().__init__(parent)
        self.side_name = side_name
        self.selection_enabled = False
        self._drag_start = None
        self._drag_end = None
        self._on_roi_selected = None
        self.setAlignment(Qt.AlignCenter)

    def set_roi_callback(self, callback):
        self._on_roi_selected = callback

    def set_selection_enabled(self, enabled):
        self.selection_enabled = bool(enabled)
        if not self.selection_enabled:
            self._drag_start = None
            self._drag_end = None
        self.update()

    def mousePressEvent(self, event):
        if self.selection_enabled and event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self._drag_end = event.pos()
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selection_enabled and self._drag_start is not None:
            self._drag_end = event.pos()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.selection_enabled and self._drag_start is not None and event.button() == Qt.LeftButton:
            self._drag_end = event.pos()
            rect = QtCore.QRect(self._drag_start, self._drag_end).normalized()
            self._drag_start = None
            self._drag_end = None
            self.update()
            if self._on_roi_selected is not None:
                self._on_roi_selected(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection_enabled and self._drag_start is not None and self._drag_end is not None:
            painter = QtGui.QPainter(self)
            pen = QtGui.QPen(QtGui.QColor(255, 255, 0), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(QtCore.QRect(self._drag_start, self._drag_end).normalized())
            painter.end()


def _build_cnn_model(arch_name: str):
    try:
        import torchvision.models as tv_models
        import torch.nn as nn
        if arch_name == "resnet18":
            model = tv_models.resnet18(weights=None)
            n_features = model.fc.in_features
            model.fc = nn.Linear(n_features, 1)
            return model
    except Exception:
        pass

    # Fallback if torchvision is unavailable.
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 1),
    )


def _preprocess_gray_for_cnn(gray_img: np.ndarray, input_size: int = 224) -> np.ndarray:
    gray = np.asarray(gray_img, dtype=np.float32)
    resized = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_AREA)
    resized = np.clip(resized, 0, 255) / 255.0
    chw = np.stack([resized, resized, resized], axis=0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    chw = (chw - mean) / std
    return chw.astype(np.float32)


class ClassifierTrainWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, classifier_dir, arch_name="resnet18", epochs=8, batch_size=32):
        super().__init__()
        self.classifier_dir = classifier_dir
        self.arch_name = arch_name
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

    def _load_examples(self):
        present_dir = os.path.join(self.classifier_dir, "present")
        absent_dir = os.path.join(self.classifier_dir, "absent")
        present_files = sorted(
            [os.path.join(present_dir, x) for x in os.listdir(present_dir) if x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        )
        absent_files = sorted(
            [os.path.join(absent_dir, x) for x in os.listdir(absent_dir) if x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        )
        return present_files, absent_files

    def run(self):
        try:
            import torch
            from torch import nn
            from torch.utils.data import Dataset, DataLoader

            present_files, absent_files = self._load_examples()
            n_pos = len(present_files)
            n_neg = len(absent_files)
            if n_pos < 2 or n_neg < 2:
                raise RuntimeError("Need at least 2 present and 2 absent frames to train.")

            rng = np.random.default_rng(42)
            rng.shuffle(present_files)
            rng.shuffle(absent_files)
            n_val_pos = max(1, int(round(0.2 * n_pos)))
            n_val_neg = max(1, int(round(0.2 * n_neg)))

            val_pos = present_files[:n_val_pos]
            train_pos = present_files[n_val_pos:]
            val_neg = absent_files[:n_val_neg]
            train_neg = absent_files[n_val_neg:]
            if len(train_pos) == 0 or len(train_neg) == 0:
                raise RuntimeError("Not enough training examples after split; add more examples.")

            train_items = [(p, 1.0) for p in train_pos] + [(p, 0.0) for p in train_neg]
            val_items = [(p, 1.0) for p in val_pos] + [(p, 0.0) for p in val_neg]
            rng.shuffle(train_items)
            rng.shuffle(val_items)

            class FrameDataset(Dataset):
                def __init__(self, items, input_size=224):
                    self.items = items
                    self.input_size = input_size

                def __len__(self):
                    return len(self.items)

                def __getitem__(self, idx):
                    path, label = self.items[idx]
                    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if gray is None:
                        raise RuntimeError(f"Could not read training image: {path}")
                    arr = _preprocess_gray_for_cnn(gray, input_size=self.input_size)
                    x = torch.from_numpy(arr)
                    y = torch.tensor([label], dtype=torch.float32)
                    return x, y

            input_size = 224
            train_ds = FrameDataset(train_items, input_size=input_size)
            val_ds = FrameDataset(val_items, input_size=input_size)
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.progress.emit(f"Training on device: {device}")
            model = _build_cnn_model(self.arch_name).to(device)

            pos_weight_value = max(1e-6, float(len(train_neg)) / float(len(train_pos)))
            pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            use_amp = torch.cuda.is_available()
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            best_val_acc = -1.0
            best_state = None
            best_epoch = 0

            for epoch in range(1, self.epochs + 1):
                model.train()
                running_loss = 0.0
                seen = 0
                for x, y in train_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = model(x)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += float(loss.item()) * x.size(0)
                    seen += x.size(0)

                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        logits = model(x)
                        probs = torch.sigmoid(logits)
                        preds = (probs >= 0.5).float()
                        correct += int((preds == y).sum().item())
                        total += int(y.numel())
                val_acc = (correct / total) if total > 0 else 0.0
                train_loss = running_loss / max(1, seen)
                self.progress.emit(f"Epoch {epoch}/{self.epochs} | loss={train_loss:.4f} | val_acc={val_acc:.3f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if best_state is None:
                raise RuntimeError("Training failed to produce a valid model state.")

            models_dir = os.path.join(self.classifier_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(models_dir, f"{stamp}_{self.arch_name}.pt")
            payload = {
                "arch_name": self.arch_name,
                "input_size": 224,
                "threshold": 0.5,
                "state_dict": best_state,
                "metrics": {
                    "best_val_acc": float(best_val_acc),
                    "best_epoch": int(best_epoch),
                    "n_train": int(len(train_items)),
                    "n_val": int(len(val_items)),
                    "n_present_total": int(n_pos),
                    "n_absent_total": int(n_neg),
                },
            }
            torch.save(payload, model_path)
            self.finished.emit({"model_path": model_path, "metrics": payload["metrics"]})
        except Exception:
            self.failed.emit(traceback.format_exc())


class ClassifierBuilderWindow(QDialog):
    def __init__(self, app_window):
        super().__init__(app_window)
        self.app_window = app_window
        self.setWindowTitle("Build Classifier")
        self.setModal(False)
        self.resize(980, 620)

        self.classifier_dir = None
        self.training_thread = None
        self.training_worker = None

        self._init_ui()
        self.rootPathEdit.setText(os.path.expanduser("~/data/eye_view_gui/classifiers/"))

    def _init_ui(self):
        layout = QVBoxLayout(self)

        path_row = QHBoxLayout()
        self.rootPathEdit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_root)
        path_row.addWidget(QLabel("Classifier root"))
        path_row.addWidget(self.rootPathEdit)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        name_row = QHBoxLayout()
        self.classifierNameEdit = QLineEdit()
        self.classifierNameEdit.setPlaceholderText("e.g. white_secretion")
        self.initBtn = QPushButton("Initialize classifier")
        self.initBtn.clicked.connect(self._initialize_classifier)
        self.loadBtn = QPushButton("Load existing classifier")
        self.loadBtn.clicked.connect(self._load_existing_classifier)
        name_row.addWidget(QLabel("Classifier name"))
        name_row.addWidget(self.classifierNameEdit)
        name_row.addWidget(self.initBtn)
        name_row.addWidget(self.loadBtn)
        layout.addLayout(name_row)

        self.classifierPathLabel = QLabel("Classifier path: (not initialized)")
        layout.addWidget(self.classifierPathLabel)

        act_row = QHBoxLayout()
        self.eyeToggle = QComboBox()
        self.eyeToggle.addItems(["Left eye", "Right eye"])
        self.surroundCountEdit = QLineEdit("6")
        self.surroundCountEdit.setFixedWidth(60)
        self.surroundSpacingEdit = QLineEdit("3")
        self.surroundSpacingEdit.setFixedWidth(60)
        self.addPresentBtn = QPushButton("Add feature present frame")
        self.addAbsentBtn = QPushButton("Add feature absent frame")
        self.deleteBtn = QPushButton("Delete selected example frame")
        self.addPresentBtn.clicked.connect(lambda: self._add_example("present"))
        self.addAbsentBtn.clicked.connect(lambda: self._add_example("absent"))
        self.deleteBtn.clicked.connect(self._delete_selected_example)
        act_row.addWidget(QLabel("Copy eye"))
        act_row.addWidget(self.eyeToggle)
        act_row.addWidget(QLabel("Surrounding"))
        act_row.addWidget(self.surroundCountEdit)
        act_row.addWidget(QLabel("Spacing"))
        act_row.addWidget(self.surroundSpacingEdit)
        act_row.addWidget(self.addPresentBtn)
        act_row.addWidget(self.addAbsentBtn)
        act_row.addWidget(self.deleteBtn)
        layout.addLayout(act_row)

        center_row = QHBoxLayout()
        self.presentList = QListWidget()
        self.absentList = QListWidget()
        self.presentList.currentTextChanged.connect(lambda _: self._preview_selected("present"))
        self.absentList.currentTextChanged.connect(lambda _: self._preview_selected("absent"))
        center_row.addWidget(self.presentList)
        center_row.addWidget(self.absentList)

        self.previewLabel = QLabel("Example preview")
        self.previewLabel.setAlignment(Qt.AlignCenter)
        self.previewLabel.setMinimumSize(320, 240)
        self.previewLabel.setStyleSheet("border: 1px solid #777;")
        center_row.addWidget(self.previewLabel)
        layout.addLayout(center_row)

        train_row = QHBoxLayout()
        self.trainBtn = QPushButton("Train classifier")
        self.trainBtn.clicked.connect(self._train_classifier)
        self.modelList = QListWidget()
        self.classifyBtn = QPushButton("Classify current frame")
        self.classifyBtn.clicked.connect(self._classify_current_frame)
        train_col = QVBoxLayout()
        train_col.addWidget(self.trainBtn)
        train_col.addWidget(QLabel("Trained models"))
        train_col.addWidget(self.modelList)
        train_col.addWidget(self.classifyBtn)
        train_row.addLayout(train_col)

        self.logText = QtWidgets.QPlainTextEdit()
        self.logText.setReadOnly(True)
        train_row.addWidget(self.logText)
        layout.addLayout(train_row)

    def _log(self, text):
        self.logText.appendPlainText(text)

    def _browse_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select classifier root", self.rootPathEdit.text().strip() or os.path.expanduser("~"))
        if path:
            self.rootPathEdit.setText(path)

    def _initialize_classifier(self):
        root = os.path.expanduser(self.rootPathEdit.text().strip())
        name = self.classifierNameEdit.text().strip()
        if not root or not name:
            QMessageBox.warning(self, "Input Error", "Please set classifier root and classifier name.")
            return

        self.classifier_dir = os.path.join(root, name)
        existed = os.path.isdir(self.classifier_dir)
        os.makedirs(os.path.join(self.classifier_dir, "present"), exist_ok=True)
        os.makedirs(os.path.join(self.classifier_dir, "absent"), exist_ok=True)
        os.makedirs(os.path.join(self.classifier_dir, "models"), exist_ok=True)
        meta_path = os.path.join(self.classifier_dir, "meta.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                json.dump({"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

        self.classifierPathLabel.setText(f"Classifier path: {self.classifier_dir}")
        self._refresh_example_lists()
        self._refresh_model_list()
        if existed:
            self._log(f"Loaded existing classifier at {self.classifier_dir}")
        else:
            self._log(f"Initialized classifier at {self.classifier_dir}")

    def _load_existing_classifier(self):
        root = os.path.expanduser(self.rootPathEdit.text().strip())
        name = self.classifierNameEdit.text().strip()
        if not root or not name:
            QMessageBox.warning(self, "Input Error", "Please set classifier root and classifier name.")
            return
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            QMessageBox.warning(self, "Not found", f"Classifier does not exist:\n{path}")
            return
        self.classifier_dir = path
        os.makedirs(os.path.join(self.classifier_dir, "present"), exist_ok=True)
        os.makedirs(os.path.join(self.classifier_dir, "absent"), exist_ok=True)
        os.makedirs(os.path.join(self.classifier_dir, "models"), exist_ok=True)
        self.classifierPathLabel.setText(f"Classifier path: {self.classifier_dir}")
        self._refresh_example_lists()
        self._refresh_model_list()
        self._log(f"Loaded existing classifier at {self.classifier_dir}")

    def _index_path(self):
        if self.classifier_dir is None:
            return None
        return os.path.join(self.classifier_dir, "examples_index.json")

    def _load_examples_index(self):
        path = self._index_path()
        if path is None or (not os.path.exists(path)):
            return {"records": []}
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "records" not in data or not isinstance(data["records"], list):
                return {"records": []}
            return data
        except Exception:
            return {"records": []}

    def _save_examples_index(self, data):
        path = self._index_path()
        if path is None:
            return
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _source_key(self, exp_id, side_key, frame_idx):
        return f"{exp_id}|{side_key}|{int(frame_idx)}"

    def _indexed_source_keys(self):
        data = self._load_examples_index()
        keys = set()
        for rec in data.get("records", []):
            key = self._source_key(rec.get("exp_id", ""), rec.get("side", ""), rec.get("frame_idx", -1))
            keys.add(key)
        return keys

    def _current_side_key(self):
        return "left" if self.eyeToggle.currentIndex() == 0 else "right"

    def _save_example_frame(self, side_key, target_dir, frame_idx):
        gray = self.app_window.get_raw_video_frame(side_key, frame_index=frame_idx)
        if gray is None:
            raise RuntimeError("Could not read current video frame.")
        exp_id = str(getattr(self.app_window, "expID", "unknown"))
        safe_exp = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in exp_id)
        stamp = int(time.time() * 1000)
        fname = f"{safe_exp}_{side_key}_frame_{frame_idx:06d}_{stamp}.png"
        path = os.path.join(target_dir, fname)
        if not cv2.imwrite(path, gray):
            raise RuntimeError(f"Could not write frame to {path}")
        return path

    def _add_example(self, category):
        if self.classifier_dir is None:
            QMessageBox.warning(self, "Not initialized", "Initialize classifier first.")
            return
        if not self.app_window.loaded:
            QMessageBox.warning(self, "No data", "Load data in the main GUI first.")
            return
        target_dir = os.path.join(self.classifier_dir, category)
        side_key = self._current_side_key()
        exp_id = str(getattr(self.app_window, "expID", "unknown"))
        try:
            n_surrounding = int(float(self.surroundCountEdit.text().strip()))
            spacing = int(float(self.surroundSpacingEdit.text().strip()))
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Surrounding and spacing must be integers.")
            return
        n_surrounding = max(0, n_surrounding)
        spacing = max(1, spacing)

        center_idx = int(self.app_window.slider.value())
        max_idx = int(self.app_window.total_frames - 1)
        n_pairs = n_surrounding // 2
        offsets = [0]
        for k in range(1, n_pairs + 1):
            d = k * spacing
            offsets.extend([-d, d])
        if n_surrounding % 2 == 1:
            offsets.append((n_pairs + 1) * spacing)

        frame_indices = []
        seen = set()
        for off in offsets:
            idx = max(0, min(max_idx, center_idx + off))
            if idx not in seen:
                seen.add(idx)
                frame_indices.append(idx)
        try:
            index_data = self._load_examples_index()
            existing_keys = self._indexed_source_keys()
            duplicate_indices = []
            add_indices = []
            for frame_idx in frame_indices:
                key = self._source_key(exp_id, side_key, frame_idx)
                if key in existing_keys:
                    duplicate_indices.append(frame_idx)
                else:
                    add_indices.append(frame_idx)

            if duplicate_indices:
                QMessageBox.warning(
                    self,
                    "Duplicate example(s)",
                    f"Skipped {len(duplicate_indices)} duplicate source frame(s) for {exp_id}/{side_key}: "
                    + ", ".join(str(x) for x in duplicate_indices[:10])
                    + ("..." if len(duplicate_indices) > 10 else ""),
                )

            saved = []
            for frame_idx in add_indices:
                path = self._save_example_frame(side_key, target_dir, frame_idx=frame_idx)
                saved.append(path)
                index_data["records"].append(
                    {
                        "exp_id": exp_id,
                        "side": side_key,
                        "frame_idx": int(frame_idx),
                        "category": category,
                        "file": os.path.basename(path),
                        "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            self._save_examples_index(index_data)
            self._refresh_example_lists()
            self._log(
                f"Saved {len(saved)} {category} example(s) from frame {center_idx} "
                f"(surrounding={n_surrounding}, spacing={spacing})."
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _refresh_example_lists(self):
        self.presentList.clear()
        self.absentList.clear()
        if self.classifier_dir is None:
            return
        for category, widget in (("present", self.presentList), ("absent", self.absentList)):
            d = os.path.join(self.classifier_dir, category)
            if not os.path.isdir(d):
                continue
            files = sorted([x for x in os.listdir(d) if x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))])
            widget.addItems(files)

    def _selected_example_path(self):
        if self.presentList.hasFocus() and self.presentList.currentItem() is not None:
            return os.path.join(self.classifier_dir, "present", self.presentList.currentItem().text())
        if self.absentList.hasFocus() and self.absentList.currentItem() is not None:
            return os.path.join(self.classifier_dir, "absent", self.absentList.currentItem().text())
        if self.presentList.currentItem() is not None:
            return os.path.join(self.classifier_dir, "present", self.presentList.currentItem().text())
        if self.absentList.currentItem() is not None:
            return os.path.join(self.classifier_dir, "absent", self.absentList.currentItem().text())
        return None

    def _delete_selected_example(self):
        if self.classifier_dir is None:
            return
        path = self._selected_example_path()
        if path is None or not os.path.exists(path):
            QMessageBox.information(self, "Delete", "No example selected.")
            return
        os.remove(path)
        data = self._load_examples_index()
        fname = os.path.basename(path)
        data["records"] = [r for r in data.get("records", []) if r.get("file") != fname]
        self._save_examples_index(data)
        self._refresh_example_lists()
        self.previewLabel.setPixmap(QtGui.QPixmap())
        self.previewLabel.setText("Example preview")
        self._log(f"Deleted example: {os.path.basename(path)}")

    def _preview_selected(self, category):
        if self.classifier_dir is None:
            return
        widget = self.presentList if category == "present" else self.absentList
        item = widget.currentItem()
        if item is None:
            return
        path = os.path.join(self.classifier_dir, category, item.text())
        pix = QtGui.QPixmap(path)
        if pix.isNull():
            self.previewLabel.setText("Failed to load preview")
            return
        self.previewLabel.setPixmap(pix.scaled(self.previewLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refresh_model_list(self):
        self.modelList.clear()
        if self.classifier_dir is None:
            return
        d = os.path.join(self.classifier_dir, "models")
        if not os.path.isdir(d):
            return
        files = sorted([x for x in os.listdir(d) if x.lower().endswith(".pt")])
        self.modelList.addItems(files)

    def _train_classifier(self):
        if self.classifier_dir is None:
            QMessageBox.warning(self, "Not initialized", "Initialize classifier first.")
            return
        if self.training_thread is not None:
            QMessageBox.information(self, "Training", "Training already in progress.")
            return

        self.training_thread = QtCore.QThread(self)
        self.training_worker = ClassifierTrainWorker(self.classifier_dir, arch_name="resnet18", epochs=8, batch_size=32)
        self.training_worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.progress.connect(self._log)
        self.training_worker.finished.connect(self._on_training_finished)
        self.training_worker.failed.connect(self._on_training_failed)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.failed.connect(self.training_thread.quit)
        self.training_thread.finished.connect(self._cleanup_training_thread)
        self.trainBtn.setEnabled(False)
        self._log("Started training...")
        self.training_thread.start()

    def _cleanup_training_thread(self):
        if self.training_worker is not None:
            self.training_worker.deleteLater()
        if self.training_thread is not None:
            self.training_thread.deleteLater()
        self.training_worker = None
        self.training_thread = None
        self.trainBtn.setEnabled(True)

    def _on_training_finished(self, result):
        model_path = result.get("model_path", "")
        metrics = result.get("metrics", {})
        self._log(f"Training complete: {model_path}")
        self._log(f"Best val acc: {metrics.get('best_val_acc', 0.0):.3f}")
        self._refresh_model_list()

    def _on_training_failed(self, err):
        self._log("Training failed.")
        self._log(err)
        QMessageBox.critical(self, "Training Error", err.splitlines()[-1] if err else "Unknown training error")

    def _classify_current_frame(self):
        if self.classifier_dir is None:
            QMessageBox.warning(self, "Not initialized", "Initialize classifier first.")
            return
        item = self.modelList.currentItem()
        if item is None:
            QMessageBox.warning(self, "No model", "Select a trained model.")
            return
        if not self.app_window.loaded:
            QMessageBox.warning(self, "No data", "Load data in the main GUI first.")
            return

        model_path = os.path.join(self.classifier_dir, "models", item.text())
        side_key = self._current_side_key()
        gray = self.app_window.get_raw_video_frame(side_key)
        if gray is None:
            QMessageBox.warning(self, "Frame Error", "Could not read current frame.")
            return

        try:
            import torch

            payload = torch.load(model_path, map_location="cpu")
            arch_name = payload.get("arch_name", "resnet18")
            input_size = int(payload.get("input_size", 224))
            threshold = float(payload.get("threshold", 0.5))

            model = _build_cnn_model(arch_name)
            model.load_state_dict(payload["state_dict"])
            model.eval()

            x = _preprocess_gray_for_cnn(gray, input_size=input_size)
            x = torch.from_numpy(x).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                prob = float(torch.sigmoid(logits).item())
            has_feature = prob >= threshold
            text = "Feature present" if has_feature else "Feature absent"
            QMessageBox.information(self, "Classifier result", f"{text}\nConfidence: {prob:.3f}")
            self._log(f"Inference ({item.text()}): prob={prob:.3f} -> {text}")
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", str(e))


class VideoAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.loaded = False
        self.playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.playFrame)
        self.vlines = []

        # QC editing state
        self.current_eye = None            # 'left' or 'right'
        self.current_start_idx = None      # int

        # Drag-select state
        self.selection_eye = None          # 'left' or 'right'
        self.selection_range = None        # (start, end) in frame indices
        self.selection_patch = None        # matplotlib patch to show selection
        self.left_span = None
        self.right_span = None

        # Zoom-to-eye display state
        self.zoom_to_eye_enabled = False
        self.zoom_roi_left = None
        self.zoom_roi_right = None
        self.manual_zoom_queue = []
        self.manual_zoom_side = None
        self.last_display_meta = {
            'left': {'shape': None, 'offset': (0, 0)},
            'right': {'shape': None, 'offset': (0, 0)},
        }
        self.classifier_builder_window = None

        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Video Analysis GUI")
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        mainLayout = QVBoxLayout(centralWidget)
        
        # --- Top Input Fields ---
        inputLayout = QHBoxLayout()
        self.userIdEdit = QLineEdit()
        self.userIdEdit.setPlaceholderText("Enter User ID")
        self.userIdEdit.setText(getpass.getuser())
        self.expIdEdit = QLineEdit()
        self.expIdEdit.setPlaceholderText("Enter Experiment ID")
        self.loadButton = QPushButton("Load Data")
        self.loadButton.clicked.connect(self.loadData)
        inputLayout.addWidget(QLabel("User ID:"))
        inputLayout.addWidget(self.userIdEdit)
        inputLayout.addWidget(QLabel("Experiment ID:"))
        inputLayout.addWidget(self.expIdEdit)
        inputLayout.addWidget(self.loadButton)
        mainLayout.addLayout(inputLayout)
        
        # --- Video Control Buttons and Slider ---
        controlLayout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.updateFrame)
        self.playButton = QPushButton("Play")
        self.playButton.setEnabled(False)
        self.playButton.clicked.connect(self.startPlayback)
        self.stopButton = QPushButton("Stop")
        self.stopButton.setEnabled(False)
        self.stopButton.clicked.connect(self.stopPlayback)

        # Frame jump edit + center-of-view button
        self.frameJumpEdit = QLineEdit()
        self.frameJumpEdit.setPlaceholderText("Frame #")
        self.frameJumpEdit.setFixedWidth(100)
        self.frameJumpEdit.setEnabled(False)
        self.frameJumpEdit.returnPressed.connect(self.jumpToTypedFrame)

        self.centerViewBtn = QPushButton("Jump to View Center")
        self.centerViewBtn.setEnabled(False)
        self.centerViewBtn.clicked.connect(self.jumpToViewCenter)
        self.zoomEyeBtn = QPushButton("Zoom to Eye")
        self.zoomEyeBtn.setCheckable(True)
        self.zoomEyeBtn.setEnabled(False)
        self.zoomEyeBtn.toggled.connect(self.toggleZoomToEye)
        self.buildClassifierBtn = QPushButton("Build Classifier")
        self.buildClassifierBtn.setEnabled(False)
        self.buildClassifierBtn.clicked.connect(self.openClassifierBuilder)

        controlLayout.addWidget(self.slider)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.stopButton)
        controlLayout.addWidget(QLabel("Current / Go to:"))
        controlLayout.addWidget(self.frameJumpEdit)
        controlLayout.addWidget(self.centerViewBtn)
        controlLayout.addWidget(self.zoomEyeBtn)
        controlLayout.addWidget(self.buildClassifierBtn)
        mainLayout.addLayout(controlLayout)

        # --- QC Editing Buttons ---
        qcLayout = QHBoxLayout()
        self.startLeftBtn = QPushButton("Set Start (Left)")
        self.startLeftBtn.setEnabled(False)
        self.startLeftBtn.clicked.connect(self.setStartLeft)

        self.startRightBtn = QPushButton("Set Start (Right)")
        self.startRightBtn.setEnabled(False)
        self.startRightBtn.clicked.connect(self.setStartRight)

        self.setEndBtn = QPushButton("Set End")
        self.setEndBtn.setEnabled(False)
        self.setEndBtn.clicked.connect(self.setEndRange)

        # NEW: Blank button to apply last drag selection
        self.blankBtn = QPushButton("Blank")
        self.blankBtn.setEnabled(False)
        self.blankBtn.clicked.connect(self.apply_current_selection)

        self.saveBtn = QPushButton("Save Changes")
        self.saveBtn.setEnabled(False)
        self.saveBtn.clicked.connect(self.saveChanges)

        qcLayout.addWidget(self.startLeftBtn)
        qcLayout.addWidget(self.startRightBtn)
        qcLayout.addWidget(self.setEndBtn)
        qcLayout.addWidget(self.blankBtn)
        qcLayout.addWidget(self.saveBtn)
        mainLayout.addLayout(qcLayout)

        # --- Processing / Filtering Controls ---
        procLayout = QHBoxLayout()

        # Median filter controls
        procLayout.addWidget(QLabel("Median window:"))
        self.medianWinEdit = QLineEdit("5")
        self.medianWinEdit.setFixedWidth(60)
        self.medianWinEdit.setEnabled(False)
        self.applyMedianBtn = QPushButton("Apply Median Filter")
        self.applyMedianBtn.setEnabled(False)
        self.applyMedianBtn.clicked.connect(self.applyMedianFilter)
        procLayout.addWidget(self.medianWinEdit)
        procLayout.addWidget(self.applyMedianBtn)

        procLayout.addSpacing(20)

        # NaN-gap interpolation controls
        procLayout.addWidget(QLabel("Max gap (frames):"))
        self.maxGapEdit = QLineEdit("10")
        self.maxGapEdit.setFixedWidth(60)
        self.maxGapEdit.setEnabled(False)
        self.fillGapsBtn = QPushButton("Fill NaN Gaps")
        self.fillGapsBtn.setEnabled(False)
        self.fillGapsBtn.clicked.connect(self.fillNanGaps)
        procLayout.addWidget(self.maxGapEdit)
        procLayout.addWidget(self.fillGapsBtn)

        mainLayout.addLayout(procLayout)
        
        # --- Video Display Widgets ---
        videoLayout = QHBoxLayout()
        leftCol = QVBoxLayout()
        rightCol = QVBoxLayout()

        self.leftVideoHeader = QLabel("LEFT EYE")
        self.leftVideoHeader.setAlignment(Qt.AlignCenter)
        self.rightVideoHeader = QLabel("RIGHT EYE")
        self.rightVideoHeader.setAlignment(Qt.AlignCenter)
        left_header_font = self.leftVideoHeader.font()
        left_header_font.setPointSize(max(1, int(round(left_header_font.pointSizeF() * 3))))
        self.leftVideoHeader.setFont(left_header_font)
        right_header_font = self.rightVideoHeader.font()
        right_header_font.setPointSize(max(1, int(round(right_header_font.pointSizeF() * 3))))
        self.rightVideoHeader.setFont(right_header_font)

        self.leftBlackLabel = QLabel("Black: 0")
        self.leftWhiteLabel = QLabel("White: 255")
        self.leftBlackSlider = QSlider(Qt.Horizontal)
        self.leftBlackSlider.setRange(0, 254)
        self.leftBlackSlider.setValue(0)
        self.leftBlackSlider.setEnabled(False)
        self.leftBlackSlider.valueChanged.connect(self._on_video_levels_changed)
        self.leftWhiteSlider = QSlider(Qt.Horizontal)
        self.leftWhiteSlider.setRange(1, 255)
        self.leftWhiteSlider.setValue(255)
        self.leftWhiteSlider.setEnabled(False)
        self.leftWhiteSlider.valueChanged.connect(self._on_video_levels_changed)

        self.rightBlackLabel = QLabel("Black: 0")
        self.rightWhiteLabel = QLabel("White: 255")
        self.rightBlackSlider = QSlider(Qt.Horizontal)
        self.rightBlackSlider.setRange(0, 254)
        self.rightBlackSlider.setValue(0)
        self.rightBlackSlider.setEnabled(False)
        self.rightBlackSlider.valueChanged.connect(self._on_video_levels_changed)
        self.rightWhiteSlider = QSlider(Qt.Horizontal)
        self.rightWhiteSlider.setRange(1, 255)
        self.rightWhiteSlider.setValue(255)
        self.rightWhiteSlider.setEnabled(False)
        self.rightWhiteSlider.valueChanged.connect(self._on_video_levels_changed)

        self.leftVideoLabel = VideoDisplayLabel("left")
        self.leftVideoLabel.setFixedSize(320, 240)
        self.leftVideoLabel.setText("Left Eye Video")
        self.leftVideoLabel.set_roi_callback(lambda rect: self._on_manual_roi_selected('left', rect))

        self.rightVideoLabel = VideoDisplayLabel("right")
        self.rightVideoLabel.setFixedSize(320, 240)
        self.rightVideoLabel.setText("Right Eye Video")
        self.rightVideoLabel.set_roi_callback(lambda rect: self._on_manual_roi_selected('right', rect))

        leftCol.addWidget(self.leftVideoHeader, alignment=Qt.AlignHCenter)
        leftCol.addWidget(self.leftBlackLabel)
        leftCol.addWidget(self.leftBlackSlider)
        leftCol.addWidget(self.leftWhiteLabel)
        leftCol.addWidget(self.leftWhiteSlider)
        leftCol.addWidget(self.leftVideoLabel, alignment=Qt.AlignHCenter)
        rightCol.addWidget(self.rightVideoHeader, alignment=Qt.AlignHCenter)
        rightCol.addWidget(self.rightBlackLabel)
        rightCol.addWidget(self.rightBlackSlider)
        rightCol.addWidget(self.rightWhiteLabel)
        rightCol.addWidget(self.rightWhiteSlider)
        rightCol.addWidget(self.rightVideoLabel, alignment=Qt.AlignHCenter)
        videoLayout.addLayout(leftCol)
        videoLayout.addLayout(rightCol)
        mainLayout.addLayout(videoLayout)
        
        # --- Percentile Control Panel ---
        percentileLayout = QHBoxLayout()
        self.lowerPercentileEdit = QLineEdit("0")
        self.upperPercentileEdit = QLineEdit("99")
        updatePercentileButton = QPushButton("Update Y-Limits")
        updatePercentileButton.clicked.connect(self.plotPupilProperties)
        percentileLayout.addWidget(QLabel("Lower Percentile:"))
        percentileLayout.addWidget(self.lowerPercentileEdit)
        percentileLayout.addWidget(QLabel("Upper Percentile:"))
        percentileLayout.addWidget(self.upperPercentileEdit)
        percentileLayout.addWidget(updatePercentileButton)
        mainLayout.addLayout(percentileLayout)
        
        # --- Matplotlib Canvas for Pupil Property Plots ---
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        mainLayout.addWidget(self.canvas)
        
        # --- Matplotlib Navigation Toolbar (for zooming and panning) ---
        self.toolbar = NavigationToolbar(self.canvas, self)
        mainLayout.addWidget(self.toolbar)

        # Reliable key shortcuts at the Qt layer (avoid focus issues with mpl canvas)
        self.shortcut_blank_b = QShortcut(QKeySequence("B"), self)
        self.shortcut_blank_b.activated.connect(self.apply_current_selection)
        self.shortcut_blank_enter = QShortcut(QKeySequence(Qt.Key_Return), self)
        self.shortcut_blank_enter.activated.connect(self.apply_current_selection)
        self.shortcut_blank_enter2 = QShortcut(QKeySequence(Qt.Key_Enter), self)
        self.shortcut_blank_enter2.activated.connect(self.apply_current_selection)

        # (We keep mpl key hook too; harmless backup)
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def loadData(self):
        """
        Loads data based on the entered User ID and Experiment ID.
        Sets up file paths, copies videos if necessary, loads pickle data,
        configures the slider, and plots the pupil properties.
        """
        self.userID = self.userIdEdit.text().strip()
        self.expID = self.expIdEdit.text().strip()
        if not self.userID or not self.expID:
            QMessageBox.warning(self, "Input Error", "Please enter both User ID and Experiment ID")
            return

        # Get paths using the shared path resolver.
        self.animalID, self.remote_repository_root, self.processed_root, \
            self.exp_dir_processed, self.exp_dir_raw = organise_paths.find_paths(self.userID, self.expID)
        self.exp_dir_processed_recordings = os.path.join(self.exp_dir_processed, 'recordings')
        self.exp_dir_processed_cut = os.path.join(self.exp_dir_processed, 'cut')        
        # Video file paths.
        self.video_path_left = os.path.join(self.exp_dir_processed, f"{self.expID}_eye1_left.avi")
        self.video_path_right = os.path.join(self.exp_dir_processed, f"{self.expID}_eye1_right.avi")
        
        # Check video existence; attempt to copy from raw if not found.
        if not os.path.isfile(self.video_path_left):
            try:
                print("Copying eye videos if necessary")
                shutil.copyfile(os.path.join(self.exp_dir_raw, f"{self.expID}_eye1_left.avi"), self.video_path_left)
                shutil.copyfile(os.path.join(self.exp_dir_raw, f"{self.expID}_eye1_right.avi"), self.video_path_right)
                print("Copy complete!")
            except Exception as e:
                print("Cropped eye videos not found on server:", e)
                QMessageBox.critical(self, "File Error", "Eye videos not found. Please check the paths.")
                return
        
        # Load pickle data.
        try:
            with open(os.path.join(self.exp_dir_processed_recordings, 'dlcEyeLeft.pickle'), "rb") as file:
                self.left_eyedat = pickle.load(file)
            with open(os.path.join(self.exp_dir_processed_recordings, 'dlcEyeRight.pickle'), "rb") as file:
                self.right_eyedat = pickle.load(file)
        except Exception as e:
            QMessageBox.critical(self, "Data Error", "Error loading pupil data: " + str(e))
            return

        # Ensure QC arrays exist (0 = default ok)
        self._ensure_qc_field(self.left_eyedat)
        self._ensure_qc_field(self.right_eyedat)
        
        # Open left video to get the total number of frames.
        cap = cv2.VideoCapture(self.video_path_left)
        if not cap.isOpened():
            QMessageBox.critical(self, "Video Error", "Could not open left video file.")
            return
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.setEnabled(True)
        self.playButton.setEnabled(True)
        self.stopButton.setEnabled(True)

        # Enable QC and navigation controls
        self.startLeftBtn.setEnabled(True)
        self.startRightBtn.setEnabled(True)
        self.setEndBtn.setEnabled(True)
        self.blankBtn.setEnabled(True)
        self.saveBtn.setEnabled(True)
        self.frameJumpEdit.setEnabled(True)
        self.centerViewBtn.setEnabled(True)
        self.zoomEyeBtn.setEnabled(True)
        self.buildClassifierBtn.setEnabled(True)
        self.zoomEyeBtn.blockSignals(True)
        self.zoomEyeBtn.setChecked(False)
        self.zoomEyeBtn.blockSignals(False)
        self.zoom_to_eye_enabled = False
        self.zoom_roi_left = None
        self.zoom_roi_right = None
        self.manual_zoom_queue = []
        self.manual_zoom_side = None
        self.leftVideoLabel.set_selection_enabled(False)
        self.rightVideoLabel.set_selection_enabled(False)
        self.last_display_meta = {
            'left': {'shape': None, 'offset': (0, 0)},
            'right': {'shape': None, 'offset': (0, 0)},
        }
        self.frameJumpEdit.setText("0")

        # Enable processing controls
        self.medianWinEdit.setEnabled(True)
        self.applyMedianBtn.setEnabled(True)
        self.maxGapEdit.setEnabled(True)
        self.fillGapsBtn.setEnabled(True)
        self.leftBlackSlider.setEnabled(True)
        self.leftWhiteSlider.setEnabled(True)
        self.rightBlackSlider.setEnabled(True)
        self.rightWhiteSlider.setEnabled(True)

        # Initialize display saturation from random frames:
        # black = darkest pixel, white = 60th percentile.
        self._auto_initialize_video_levels()
        self._on_video_levels_changed()

        self.loaded = True
        
        # Display the first frame and plot the pupil properties.
        self.updateFrame()
        self.plotPupilProperties()

    def _ensure_qc_field(self, eyedat):
        n = len(eyedat.get('x', []))
        if 'QC' not in eyedat or eyedat['QC'] is None or len(eyedat['QC']) != n:
            eyedat['QC'] = np.zeros(n, dtype=int)

    def _set_levels_from_frame(self, gray_frame, side):
        if gray_frame is None or gray_frame.size == 0:
            return
        p_black = int(np.clip(np.nanmin(gray_frame), 0, 254))
        p_white = int(np.clip(np.nanpercentile(gray_frame, 60), 1, 255))
        if p_white <= p_black:
            p_white = min(255, p_black + 1)

        if side == 'left':
            self.leftBlackSlider.blockSignals(True)
            self.leftWhiteSlider.blockSignals(True)
            self.leftBlackSlider.setValue(p_black)
            self.leftWhiteSlider.setValue(p_white)
            self.leftBlackSlider.blockSignals(False)
            self.leftWhiteSlider.blockSignals(False)
        else:
            self.rightBlackSlider.blockSignals(True)
            self.rightWhiteSlider.blockSignals(True)
            self.rightBlackSlider.setValue(p_black)
            self.rightWhiteSlider.setValue(p_white)
            self.rightBlackSlider.blockSignals(False)
            self.rightWhiteSlider.blockSignals(False)

    def _random_gray_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release()
            return None
        idx = int(np.random.default_rng().integers(0, n))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _auto_initialize_video_levels(self):
        left_gray = self._random_gray_frame(self.video_path_left)
        right_gray = self._random_gray_frame(self.video_path_right)
        if left_gray is not None:
            self._set_levels_from_frame(left_gray, 'left')
        if right_gray is not None:
            self._set_levels_from_frame(right_gray, 'right')

    def _on_video_levels_changed(self):
        left_b = int(self.leftBlackSlider.value())
        left_w = int(self.leftWhiteSlider.value())
        if left_w <= left_b:
            left_w = min(255, left_b + 1)
            self.leftWhiteSlider.blockSignals(True)
            self.leftWhiteSlider.setValue(left_w)
            self.leftWhiteSlider.blockSignals(False)
        self.leftBlackLabel.setText(f"Black: {left_b}")
        self.leftWhiteLabel.setText(f"White: {left_w}")

        right_b = int(self.rightBlackSlider.value())
        right_w = int(self.rightWhiteSlider.value())
        if right_w <= right_b:
            right_w = min(255, right_b + 1)
            self.rightWhiteSlider.blockSignals(True)
            self.rightWhiteSlider.setValue(right_w)
            self.rightWhiteSlider.blockSignals(False)
        self.rightBlackLabel.setText(f"Black: {right_b}")
        self.rightWhiteLabel.setText(f"White: {right_w}")

        if self.loaded:
            self.updateFrame()

    def _draw_xy_series(
        self,
        frame,
        x_vals,
        y_vals,
        color,
        point_radius=2,
        connect=False,
        thickness=1,
        closed=False,
        draw_points=True,
        offset=(0, 0),
    ):
        """Draw per-frame x/y points, optionally connected as a polyline."""
        x_vals = np.asarray(x_vals, dtype=float).reshape(-1)
        y_vals = np.asarray(y_vals, dtype=float).reshape(-1)
        finite = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not np.any(finite):
            return frame

        x0, y0 = offset
        pts = np.stack([x_vals[finite] - x0, y_vals[finite] - y0], axis=1).astype(np.int32)
        if connect and len(pts) >= 2:
            frame = cv2.polylines(frame, [pts], isClosed=closed, color=color, thickness=thickness)
        if draw_points:
            for px, py in pts:
                frame = cv2.circle(frame, (int(px), int(py)), point_radius, color, -1)
        return frame

    def overlay_plot(self, frame, position, eyeDat, offset=(0, 0)):
        """
        Draw fit overlays on top of the current frame.
        """
        x0, y0 = offset
        is_zoomed = self.zoom_to_eye_enabled
        circle_thickness = 2 if is_zoomed else 3
        eye_line_thickness = 1 if is_zoomed else 2
        pupil_point_radius = 2 if is_zoomed else 4
        eye_anchor_point_radius = 2 if is_zoomed else 4

        # Pupil fitted circle as a continuous line.
        if np.isfinite(eyeDat['x'][position]) and np.isfinite(eyeDat['y'][position]) and np.isfinite(eyeDat['radius'][position]):
            center = (int(eyeDat['x'][position] - x0), int(eyeDat['y'][position] - y0))
            radius = int(eyeDat['radius'][position])
            frame = cv2.circle(frame, center, radius, (0, 0, 255), circle_thickness)

        # Eye outline as a continuous closed line.
        if 'eye_lid_x' in eyeDat and 'eye_lid_y' in eyeDat:
            eyelid_x = np.asarray(eyeDat['eye_lid_x'])
            eyelid_y = np.asarray(eyeDat['eye_lid_y'])
            if eyelid_x.ndim >= 2 and eyelid_y.ndim >= 2 and position < eyelid_x.shape[0] and position < eyelid_y.shape[0]:
                frame = self._draw_xy_series(
                    frame,
                    eyelid_x[position],
                    eyelid_y[position],
                    color=(0, 255, 0),
                    connect=True,
                    thickness=eye_line_thickness,
                    closed=True,
                    draw_points=False,
                    offset=offset,
                )

        # Raw eye anchor points (4 points used for eyelid fit) as dots.
        if 'eyeX' in eyeDat and 'eyeY' in eyeDat:
            eye_x = np.asarray(eyeDat['eyeX'])
            eye_y = np.asarray(eyeDat['eyeY'])
            if eye_x.ndim >= 2 and eye_y.ndim >= 2 and position < eye_x.shape[0] and position < eye_y.shape[0]:
                frame = self._draw_xy_series(
                    frame,
                    eye_x[position],
                    eye_y[position],
                    color=(255, 128, 0),
                    point_radius=eye_anchor_point_radius,
                    connect=False,
                    draw_points=True,
                    offset=offset,
                )

        # Pupil points shown as dots over the fitted circle.
        if 'pupilX' in eyeDat and 'pupilY' in eyeDat:
            pupil_x = np.asarray(eyeDat['pupilX'])
            pupil_y = np.asarray(eyeDat['pupilY'])
            if pupil_x.ndim >= 2 and pupil_y.ndim >= 2 and position < pupil_x.shape[0] and position < pupil_y.shape[0]:
                frame = self._draw_xy_series(
                    frame,
                    pupil_x[position],
                    pupil_y[position],
                    color=(255, 255, 0),
                    point_radius=pupil_point_radius,
                    connect=False,
                    draw_points=True,
                    offset=offset,
                )
        return frame

    def _get_video_shape(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        return frame.shape[:2]

    def _compute_eye_zoom_roi(self, eyedat, frame_shape):
        def coerce_point_matrix(arr):
            """
            Convert per-frame point storage to a 2D float matrix [n_frames, n_points].
            Supports dense numeric arrays and object arrays of per-frame vectors.
            """
            a = np.asarray(arr, dtype=object)
            if a.ndim == 2 and a.dtype != object:
                return a.astype(float, copy=False)
            if a.ndim == 1:
                rows = []
                max_len = 0
                for item in a:
                    row = np.asarray(item, dtype=float).reshape(-1)
                    rows.append(row)
                    if row.size > max_len:
                        max_len = row.size
                if max_len == 0 or len(rows) == 0:
                    return None
                out = np.full((len(rows), max_len), np.nan, dtype=float)
                for i, row in enumerate(rows):
                    out[i, :row.size] = row
                return out
            try:
                b = np.asarray(arr, dtype=float)
                if b.ndim == 2:
                    return b
            except Exception:
                pass
            return None

        # Prefer continuous eyelid fit; fall back to 4 anchor points if needed.
        if 'eye_lid_x' in eyedat and 'eye_lid_y' in eyedat:
            x = coerce_point_matrix(eyedat['eye_lid_x'])
            y = coerce_point_matrix(eyedat['eye_lid_y'])
            if x is None or y is None or x.shape != y.shape or x.shape[0] == 0:
                x = None
                y = None
        else:
            x = None
            y = None

        if x is None or y is None:
            if 'eyeX' not in eyedat or 'eyeY' not in eyedat:
                return None
            x = coerce_point_matrix(eyedat['eyeX'])
            y = coerce_point_matrix(eyedat['eyeY'])
            if x is None or y is None or x.shape != y.shape or x.shape[0] == 0:
                return None

        with np.errstate(all='ignore'):
            left = np.nanmin(x, axis=1)
            right = np.nanmax(x, axis=1)
            top = np.nanmin(y, axis=1)
            bottom = np.nanmax(y, axis=1)
        valid = np.isfinite(left) & np.isfinite(right) & np.isfinite(top) & np.isfinite(bottom)
        if not np.any(valid):
            return None

        left_med = float(np.nanmedian(left[valid]))
        right_med = float(np.nanmedian(right[valid]))
        top_med = float(np.nanmedian(top[valid]))
        bottom_med = float(np.nanmedian(bottom[valid]))
        width = right_med - left_med
        height = bottom_med - top_med
        if width <= 1 or height <= 1:
            return None

        # Expand the median eye bounds by 20% per side (40% total) to add border.
        cx = 0.5 * (left_med + right_med)
        cy = 0.5 * (top_med + bottom_med)
        width *= 1.4
        height *= 1.4

        frame_h, frame_w = frame_shape
        x0 = max(0, int(np.floor(cx - 0.5 * width)))
        x1 = min(frame_w, int(np.ceil(cx + 0.5 * width)))
        y0 = max(0, int(np.floor(cy - 0.5 * height)))
        y1 = min(frame_h, int(np.ceil(cy + 0.5 * height)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return (x0, x1, y0, y1)

    def _prepare_zoom_settings(self):
        left_shape = self._get_video_shape(self.video_path_left)
        right_shape = self._get_video_shape(self.video_path_right)
        if left_shape is None or right_shape is None:
            return False

        self.zoom_roi_left = self._compute_eye_zoom_roi(self.left_eyedat, left_shape)
        self.zoom_roi_right = self._compute_eye_zoom_roi(self.right_eyedat, right_shape)

        # Allow zoom mode when at least one side has a valid ROI.
        return (self.zoom_roi_left is not None) or (self.zoom_roi_right is not None)

    def toggleZoomToEye(self, enabled):
        if not self.loaded:
            self.zoom_to_eye_enabled = False
            self.zoomEyeBtn.blockSignals(True)
            self.zoomEyeBtn.setChecked(False)
            self.zoomEyeBtn.blockSignals(False)
            return

        if enabled:
            self.zoom_to_eye_enabled = True
            self._prepare_zoom_settings()
            self.manual_zoom_queue = []
            if self.zoom_roi_left is None:
                self.manual_zoom_queue.append('left')
            if self.zoom_roi_right is None:
                self.manual_zoom_queue.append('right')

            if self.manual_zoom_queue:
                self._prompt_next_manual_zoom_eye()
            else:
                self._set_manual_zoom_side(None)
        else:
            self.zoom_to_eye_enabled = False
            self.manual_zoom_queue = []
            self._set_manual_zoom_side(None)
        self.updateFrame()

    def _set_manual_zoom_side(self, side):
        self.manual_zoom_side = side
        self.leftVideoLabel.set_selection_enabled(side == 'left')
        self.rightVideoLabel.set_selection_enabled(side == 'right')

    def _prompt_next_manual_zoom_eye(self):
        while self.manual_zoom_queue:
            side = self.manual_zoom_queue.pop(0)
            side_name = "LEFT" if side == 'left' else "RIGHT"
            choice = QMessageBox.question(
                self,
                "Auto Zoom Unavailable",
                (
                    f"{side_name} eye auto-zoom window could not be computed.\n"
                    "Do you want to draw the zoom window manually for this eye?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if choice == QMessageBox.Yes:
                self._set_manual_zoom_side(side)
                QMessageBox.information(
                    self,
                    "Manual Zoom Selection",
                    f"Drag a rectangle on the {side_name} eye video.",
                )
                self.updateFrame()
                return

        # Completed manual prompts for all failed eyes.
        self._set_manual_zoom_side(None)
        if self.zoom_roi_left is None and self.zoom_roi_right is None:
            self.zoom_to_eye_enabled = False
            self.zoomEyeBtn.blockSignals(True)
            self.zoomEyeBtn.setChecked(False)
            self.zoomEyeBtn.blockSignals(False)
            QMessageBox.warning(
                self,
                "Zoom Disabled",
                "Zoom was disabled because no valid auto or manual zoom windows were set.",
            )
        self.updateFrame()

    def _label_rect_to_fullframe_roi(self, side, rect):
        meta = self.last_display_meta.get(side, {})
        shape = meta.get('shape', None)
        offset = meta.get('offset', (0, 0))
        if shape is None:
            return None

        frame_h, frame_w = shape
        label = self.leftVideoLabel if side == 'left' else self.rightVideoLabel
        label_w = max(1, label.width())
        label_h = max(1, label.height())

        scale = min(label_w / float(frame_w), label_h / float(frame_h))
        draw_w = max(1, int(round(frame_w * scale)))
        draw_h = max(1, int(round(frame_h * scale)))
        x_pad = (label_w - draw_w) / 2.0
        y_pad = (label_h - draw_h) / 2.0

        x0 = max(rect.left(), x_pad)
        x1 = min(rect.right(), x_pad + draw_w)
        y0 = max(rect.top(), y_pad)
        y1 = min(rect.bottom(), y_pad + draw_h)
        if x1 <= x0 or y1 <= y0:
            return None

        fx0 = int(np.floor((x0 - x_pad) / scale))
        fx1 = int(np.ceil((x1 - x_pad) / scale))
        fy0 = int(np.floor((y0 - y_pad) / scale))
        fy1 = int(np.ceil((y1 - y_pad) / scale))
        fx0 = max(0, min(frame_w - 1, fx0))
        fx1 = max(1, min(frame_w, fx1))
        fy0 = max(0, min(frame_h - 1, fy0))
        fy1 = max(1, min(frame_h, fy1))
        if fx1 - fx0 < 2 or fy1 - fy0 < 2:
            return None

        off_x, off_y = offset
        return (fx0 + off_x, fx1 + off_x, fy0 + off_y, fy1 + off_y)

    def _on_manual_roi_selected(self, side, rect):
        if not self.zoom_to_eye_enabled or self.manual_zoom_side != side:
            return

        roi = self._label_rect_to_fullframe_roi(side, rect)
        if roi is None:
            QMessageBox.warning(self, "Selection Error", "Please drag a larger rectangle inside the video image.")
            return

        if side == 'left':
            self.zoom_roi_left = roi
        else:
            self.zoom_roi_right = roi

        self._set_manual_zoom_side(None)
        self._prompt_next_manual_zoom_eye()

    def openClassifierBuilder(self):
        if self.classifier_builder_window is None:
            self.classifier_builder_window = ClassifierBuilderWindow(self)
        self.classifier_builder_window.show()
        self.classifier_builder_window.raise_()
        self.classifier_builder_window.activateWindow()

    def get_raw_video_frame(self, side_key, frame_index=None):
        if not self.loaded:
            return None
        if frame_index is None:
            frame_index = int(self.slider.value())
        video_path = self.video_path_left if side_key == "left" else self.video_path_right
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _get_display_levels(self, side):
        if side == 'left':
            low = float(self.leftBlackSlider.value())
            high = float(self.leftWhiteSlider.value())
        else:
            low = float(self.rightBlackSlider.value())
            high = float(self.rightWhiteSlider.value())
        if high <= low:
            high = low + 1.0
        return low, high

    def _contrast_scale(self, gray_frame, low, high):
        if (not np.isfinite(low)) or (not np.isfinite(high)) or high <= low:
            return np.zeros_like(gray_frame, dtype=np.uint8)
        scaled = (gray_frame - low) / (high - low)
        scaled = np.clip(scaled, 0.0, 1.0) * 255.0
        return scaled.astype(np.uint8)

    def playVideoFrame(self, frame_position, video_path, eyedat, side="Left"):
        """
        Opens the video file at video_path, grabs the frame at frame_position,
        applies the overlay, and returns the frame.
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = cap.read()
        cap.release()
        if ret:
            side_key = side.lower()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            offset = (0, 0)

            if self.zoom_to_eye_enabled:
                if side.lower() == "left":
                    roi = self.zoom_roi_left
                else:
                    roi = self.zoom_roi_right

                if roi is not None:
                    x0, x1, y0, y1 = roi
                    gray = gray[y0:y1, x0:x1]
                    offset = (x0, y0)
            low, high = self._get_display_levels(side_key)
            gray = self._contrast_scale(gray, low, high)

            frame = np.stack((gray,) * 3, axis=-1).astype(np.uint8)
            frame = self.overlay_plot(frame, frame_position, eyedat, offset=offset)
            if side_key in self.last_display_meta:
                self.last_display_meta[side_key] = {
                    'shape': frame.shape[:2],
                    'offset': offset,
                }
            return frame
        return None

    def updateFrame(self):
        """
        Called when the slider value changes. Updates both the video displays and the vertical line.
        Also updates the frame number box to show the current frame.
        """
        if not self.loaded:
            return
        
        position = self.slider.value()
        # Update left video frame.
        frame_left = self.playVideoFrame(position, self.video_path_left, self.left_eyedat, side="Left")
        if frame_left is not None:
            image_left = QtGui.QImage(frame_left.data, frame_left.shape[1], frame_left.shape[0],
                                      frame_left.strides[0], QtGui.QImage.Format_RGB888)
            pixmap_left = QtGui.QPixmap.fromImage(image_left)
            self.leftVideoLabel.setPixmap(pixmap_left.scaled(self.leftVideoLabel.size(), Qt.KeepAspectRatio))
        
        # Update right video frame.
        frame_right = self.playVideoFrame(position, self.video_path_right, self.right_eyedat, side="Right")
        if frame_right is not None:
            image_right = QtGui.QImage(frame_right.data, frame_right.shape[1], frame_right.shape[0],
                                       frame_right.strides[0], QtGui.QImage.Format_RGB888)
            pixmap_right = QtGui.QPixmap.fromImage(image_right)
            self.rightVideoLabel.setPixmap(pixmap_right.scaled(self.rightVideoLabel.size(), Qt.KeepAspectRatio))
        
        # Update vertical sliding lines on the plots.
        if self.vlines:
            for vline in self.vlines:
                vline.set_xdata(position)
            self.canvas.draw_idle()

        # Update the frame number box
        if self.frameJumpEdit.isEnabled():
            self.frameJumpEdit.setText(str(position))

    def startPlayback(self):
        if not self.loaded:
            return
        self.playing = True
        self.timer.start(33)  # roughly 30 fps

    def stopPlayback(self):
        self.playing = False
        self.timer.stop()

    def playFrame(self):
        if self.slider.value() < self.total_frames - 1:
            self.slider.setValue(self.slider.value() + 1)
        else:
            self.stopPlayback()

    # ----- QC editing methods -----
    def setStartLeft(self):
        if not self.loaded:
            return
        self.current_eye = 'left'
        self.current_start_idx = int(self.slider.value())

    def setStartRight(self):
        if not self.loaded:
            return
        self.current_eye = 'right'
        self.current_start_idx = int(self.slider.value())

    def setEndRange(self):
        if not self.loaded:
            return
        if self.current_eye is None or self.current_start_idx is None:
            QMessageBox.warning(self, "Selection Error", "Set a start point first.")
            return
        end_idx = int(self.slider.value())
        start_idx = int(self.current_start_idx)
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        self._apply_invalid_range(self.current_eye, start_idx, end_idx)

        # reset selection
        self.current_eye = None
        self.current_start_idx = None

        # refresh views but keep current zoom
        self.updateFrame()
        self.plotPupilProperties(preserve_view=True)

    def _apply_invalid_range(self, eye, start_idx, end_idx):
        eyedat = self.left_eyedat if eye == 'left' else self.right_eyedat
        n = len(eyedat['x'])
        start_idx = max(0, min(start_idx, n - 1))
        end_idx = max(0, min(end_idx, n - 1))

        self._ensure_qc_field(eyedat)

        # Set QC=7 and values to NaN in the chosen range
        eyedat['QC'][start_idx:end_idx + 1] = 7
        for key in ['x', 'y', 'radius', 'velocity']:
            if key in eyedat and len(eyedat[key]) == n:
                arr = np.asarray(eyedat[key], dtype=float).copy()
                arr[start_idx:end_idx + 1] = np.nan
                eyedat[key] = arr

    def saveChanges(self):
        if not self.loaded:
            return
        try:
            with open(os.path.join(self.exp_dir_processed_recordings, 'dlcEyeLeft.pickle'), "wb") as file:
                pickle.dump(self.left_eyedat, file)
            with open(os.path.join(self.exp_dir_processed_recordings, 'dlcEyeRight.pickle'), "wb") as file:
                pickle.dump(self.right_eyedat, file)
            QMessageBox.information(self, "Saved", "Changes saved to pickle files.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", "Error saving changes: " + str(e))

    def _add_invalid_overlays(self, ax, qc_array, ymin, ymax):
        """Overlay black spans where QC==7."""
        if qc_array is None:
            return
        mask = (qc_array == 7)
        if not np.any(mask):
            return
        in_seg = False
        start = 0
        for i, m in enumerate(mask):
            if m and not in_seg:
                in_seg = True
                start = i
            elif not m and in_seg:
                in_seg = False
                ax.axvspan(start, i - 1, color='k', alpha=0.25)
        if in_seg:
            ax.axvspan(start, len(mask) - 1, color='k', alpha=0.25)

    # ----- Navigation helpers -----
    def jumpToViewCenter(self):
        """Jump slider/video to the center of the current x-limits (bottom plot)."""
        if not self.loaded or not self.figure.axes:
            return
        bottom_ax = self.figure.axes[-1]
        xmin, xmax = bottom_ax.get_xlim()
        center = int(round((xmin + xmax) / 2.0))
        center = max(0, min(center, self.total_frames - 1))
        self.slider.setValue(center)

    def jumpToTypedFrame(self):
        """When Enter is pressed in the frame box, jump to that frame."""
        if not self.loaded:
            return
        text = self.frameJumpEdit.text().strip()
        try:
            frame = int(float(text))  # allow "123.0"
            frame = max(0, min(frame, self.total_frames - 1))
            self.slider.setValue(frame)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Enter a valid frame number.")

    # ----- Filtering / Interpolation -----
    def applyMedianFilter(self):
        """
        Apply moving median filter to x, y, radius, diameter (if present), and velocity for both eyes.
        Shows a QProgressDialog. Keeps current zoom after updating plots.
        """
        if not self.loaded:
            return
        # window length
        try:
            k = int(float(self.medianWinEdit.text().strip()))
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Enter a valid median window length.")
            return
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1  # make it odd

        keys = [key for key in ['x', 'y', 'radius', 'diameter', 'velocity']
                if (key in self.left_eyedat) or (key in self.right_eyedat)]
        if not keys:
            QMessageBox.information(self, "Info", "No applicable series found.")
            return

        total = 0
        for eye in (self.left_eyedat, self.right_eyedat):
            for key in keys:
                if key in eye:
                    total += len(np.asarray(eye[key], dtype=float))

        progress = QProgressDialog("Filtering data...", None, 0, total, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress_value = 0
        progress.setValue(progress_value)
        QtWidgets.QApplication.processEvents()

        def step(n=1):
            nonlocal progress_value
            progress_value += n
            progress.setValue(progress_value)
            QtWidgets.QApplication.processEvents()

        for eye_name, eye in (('Left', self.left_eyedat), ('Right', self.right_eyedat)):
            for key in keys:
                if key in eye:
                    arr = np.asarray(eye[key], dtype=float)
                    progress.setLabelText(f"Filtering {eye_name} eye: {key}")
                    QtWidgets.QApplication.processEvents()
                    filtered = self._median_filter_nan(arr, k, progress_callback=step)
                    eye[key] = filtered

        progress.close()

        # refresh views, keep current zoom
        self.updateFrame()
        self.plotPupilProperties(preserve_view=True)
        QMessageBox.information(self, "Done", "Median filtering complete.")

    def fillNanGaps(self):
        """
        Fill NaN gaps up to a specified max size with linear interpolation for
        x, y, radius, diameter (if present), and velocity for both eyes.
        Shows a QProgressDialog. Keeps current zoom after updating plots.
        """
        if not self.loaded:
            return
        try:
            max_gap = int(float(self.maxGapEdit.text().strip()))
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Enter a valid max gap length.")
            return
        if max_gap <= 0:
            QMessageBox.warning(self, "Input Error", "Max gap must be > 0.")
            return

        keys = [key for key in ['x', 'y', 'radius', 'diameter', 'velocity']
                if (key in self.left_eyedat) or (key in self.right_eyedat)]
        if not keys:
            QMessageBox.information(self, "Info", "No applicable series found.")
            return

        total = 0
        for eye in (self.left_eyedat, self.right_eyedat):
            for key in keys:
                if key in eye:
                    total += len(np.asarray(eye[key], dtype=float))

        progress = QProgressDialog("Interpolating NaN gaps...", None, 0, total, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress_value = 0
        progress.setValue(progress_value)
        QtWidgets.QApplication.processEvents()

        def step(n=1):
            nonlocal progress_value
            progress_value += n
            progress.setValue(progress_value)
            QtWidgets.QApplication.processEvents()

        for eye_name, eye in (('Left', self.left_eyedat), ('Right', self.right_eyedat)):
            for key in keys:
                if key in eye:
                    arr = np.asarray(eye[key], dtype=float)
                    progress.setLabelText(f"Interpolating {eye_name} eye: {key}")
                    QtWidgets.QApplication.processEvents()
                    filled = self._interpolate_nan_gaps(arr, max_gap, progress_callback=step)
                    eye[key] = filled

        progress.close()

        # refresh
        self.updateFrame()
        self.plotPupilProperties(preserve_view=True)
        QMessageBox.information(self, "Done", "NaN gap interpolation complete.")

    def _median_filter_nan(self, arr, k, progress_callback=None):
        """
        Moving median with NaN support (np.nanmedian).
        Window size k should be odd. Edges use smaller windows.
        All-NaN windows produce NaN (handled by np.nanmedian).
        """
        x = np.asarray(arr, dtype=float)
        n = x.size
        out = np.full(n, np.nan, dtype=float)
        half = k // 2
        for i in range(n):
            s = max(0, i - half)
            e = min(n, i + half + 1)
            window = x[s:e]
            val = np.nanmedian(window)
            out[i] = val
            if progress_callback:
                progress_callback(1)
        return out

    def _interpolate_nan_gaps(self, arr, max_gap, progress_callback=None):
        """
        Linearly interpolate NaN runs up to length `max_gap`.
        Only interior gaps with finite endpoints are interpolated.
        """
        y = np.asarray(arr, dtype=float).copy()
        n = y.size
        i = 0
        while i < n:
            if np.isnan(y[i]):
                start = i
                while i < n and np.isnan(y[i]):
                    i += 1
                end = i
                run_len = end - start
                if run_len > 0 and run_len <= max_gap and start > 0 and end < n and (not np.isnan(y[start - 1])) and (not np.isnan(y[end])):
                    y0 = y[start - 1]
                    y1 = y[end]
                    for k in range(1, run_len + 1):
                        y[start + k - 1] = y0 + (y1 - y0) * (k / (run_len + 1.0))
                if progress_callback:
                    progress_callback(run_len if run_len > 0 else 1)
            else:
                i += 1
                if progress_callback:
                    progress_callback(1)
        return y

    # ----- Drag selection on plots -----
    def _setup_span_selectors(self, axs):
        # Clear references to old selectors (GC will remove them)
        self.left_span = None
        self.right_span = None

        def on_select_left(xmin, xmax):
            if not self.loaded:
                return
            self.selection_eye = 'left'
            self._set_selection_visual(axs[0], xmin, xmax)

        def on_select_right(xmin, xmax):
            if not self.loaded:
                return
            self.selection_eye = 'right'
            self._set_selection_visual(axs[1], xmin, xmax)

        self.left_span = SpanSelector(
            axs[0], onselect=on_select_left, direction='horizontal',
            useblit=True, interactive=True, props=dict(alpha=0.15, facecolor='yellow')
        )
        self.right_span = SpanSelector(
            axs[1], onselect=on_select_right, direction='horizontal',
            useblit=True, interactive=True, props=dict(alpha=0.15, facecolor='yellow')
        )

    def _set_selection_visual(self, ax, xmin, xmax):
        # Normalize and clamp to data range
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        start = int(max(0, np.floor(xmin)))
        end = int(np.ceil(xmax))
        if self.loaded:
            end = min(end, self.total_frames - 1)
        # store
        self.selection_range = (start, end)
        # remove previous patch
        if self.selection_patch is not None:
            try:
                self.selection_patch.remove()
            except Exception:
                pass
            self.selection_patch = None
        # draw new patch (yellow selection preview)
        self.selection_patch = ax.axvspan(start, end, color='yellow', alpha=0.3)
        self.canvas.draw_idle()

    def _on_key_press(self, event):
        # Backup handler; main path uses Qt shortcuts (B / Enter / Return)
        if event.key in ('b', 'B', 'enter', 'return'):
            self.apply_current_selection()

    def apply_current_selection(self):
        if not self.loaded or self.selection_eye is None or self.selection_range is None:
            return
        s, e = self.selection_range
        if e < s:
            s, e = e, s
        self._apply_invalid_range(self.selection_eye, s, e)
        # clear selection visuals
        if self.selection_patch is not None:
            try:
                self.selection_patch.remove()
            except Exception:
                pass
            self.selection_patch = None
        self.selection_eye = None
        self.selection_range = None
        # refresh views; keep zoom (gray spans will appear via QC overlay)
        self.updateFrame()
        self.plotPupilProperties(preserve_view=True)

    def plotPupilProperties(self, preserve_view: bool = False):
        """
        Plots:
          1. Left pupil positions (x and y; median subtracted)
          2. Right pupil positions (x and y; median subtracted)
          3. Pupil radius
          4. Pupil velocity

        Keeps existing behavior. If preserve_view is True, current x/y limits are restored.
        """
        # capture current limits BEFORE clearing
        stored_limits = None
        if preserve_view and self.figure.axes:
            stored_limits = [(ax.get_xlim(), ax.get_ylim()) for ax in self.figure.axes[:4]]

        left_dlc = self.left_eyedat
        right_dlc = self.right_eyedat
        
        # Percentiles
        try:
            lower_pct = float(self.lowerPercentileEdit.text())
            upper_pct = float(self.upperPercentileEdit.text())
        except ValueError:
            lower_pct, upper_pct = 0, 99
        
        self.figure.clear()
        axs = self.figure.subplots(4, 1, sharex=True)
        
        # --- Plot 1: Left pupil positions ---
        try:
            left_x = left_dlc['x'] - np.nanmedian(left_dlc['x'])
            left_y = left_dlc['y'] - np.nanmedian(left_dlc['y'])
            ax = axs[0]
            ax.plot(left_x, color='skyblue')
            ax.plot(left_y, color='navy')
            combined = np.concatenate([left_x, left_y])
            lower_lim = np.nanpercentile(combined, lower_pct)
            upper_lim = np.nanpercentile(combined, upper_pct)
            if np.isnan(lower_lim) or np.isnan(upper_lim):
                lower_lim = -1
                upper_lim = 1
            ax.set_ylim(lower_lim, upper_lim)
            ax.set_ylabel('Left Pos')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', labelbottom=False)
            self._add_invalid_overlays(ax, left_dlc.get('QC', None), lower_lim, upper_lim)
        except Exception as e:
            print("Error plotting left pupil data:", e)
            
        # --- Plot 2: Right pupil positions ---
        try:
            right_x = right_dlc['x'] - np.nanmedian(right_dlc['x'])
            right_y = right_dlc['y'] - np.nanmedian(right_dlc['y'])
            ax = axs[1]
            ax.plot(right_x, color='lightcoral')
            ax.plot(right_y, color='maroon')
            combined = np.concatenate([right_x, right_y])
            lower_lim = np.nanpercentile(combined, lower_pct)
            upper_lim = np.nanpercentile(combined, upper_pct)
            if np.isnan(lower_lim) or np.isnan(upper_lim):
                lower_lim = -1
                upper_lim = 1
            ax.set_ylim(lower_lim, upper_lim)
            ax.set_ylabel('Right Pos')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='x', labelbottom=False)
            self._add_invalid_overlays(ax, right_dlc.get('QC', None), lower_lim, upper_lim)
        except Exception as e:
            print("Error plotting right pupil data:", e)
        
        # --- Plot 3: Pupil radius ---
        ax = axs[2]
        ax.plot(left_dlc['radius'], color='blue')
        ax.plot(right_dlc['radius'], color='red')
        combined = np.concatenate([left_dlc['radius'], right_dlc['radius']])
        lower_lim = np.nanpercentile(combined, lower_pct)
        upper_lim = np.nanpercentile(combined, upper_pct)
        if np.isnan(lower_lim) or np.isnan(upper_lim):
            lower_lim = 0
            upper_lim = 1
        ax.set_ylim(lower_lim, upper_lim)
        ax.set_ylabel('Radius')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelbottom=False)
        self._add_invalid_overlays(ax, left_dlc.get('QC', None), lower_lim, upper_lim)
        self._add_invalid_overlays(ax, right_dlc.get('QC', None), lower_lim, upper_lim)
        
        # --- Plot 4: Pupil velocity ---
        ax = axs[3]
        ax.plot(left_dlc['velocity'], color='blue')
        ax.plot(right_dlc['velocity'], color='red')
        combined = np.concatenate([left_dlc['velocity'], right_dlc['velocity']])
        lower_lim = np.nanpercentile(combined, lower_pct)
        upper_lim = np.nanpercentile(combined, upper_pct)
        if np.isnan(lower_lim) or np.isnan(upper_lim):
            lower_lim = 0
            upper_lim = 1
        ax.set_ylim(lower_lim, upper_lim)
        ax.set_ylabel('Velocity')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._add_invalid_overlays(ax, left_dlc.get('QC', None), lower_lim, upper_lim)
        self._add_invalid_overlays(ax, right_dlc.get('QC', None), lower_lim, upper_lim)
        
        # --- Vertical dashed line on each subplot ---
        self.vlines = []
        current_frame = self.slider.value() if self.loaded else 0
        for ax in axs:
            vline = ax.axvline(x=current_frame, color='k', linestyle='--')
            self.vlines.append(vline)

        # Drag-select setup (left/right position plots)
        self._setup_span_selectors(axs)
        
        # restore limits if requested
        if preserve_view and stored_limits:
            for i, ax in enumerate(axs):
                if i < len(stored_limits):
                    try:
                        xlim, ylim = stored_limits[i]
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)
                    except Exception:
                        pass

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoAnalysisApp()
    win.show()
    sys.exit(app.exec_())
