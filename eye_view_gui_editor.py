import sys
import os
import pickle
import shutil
import cv2
import numpy as np

# PyQt5 imports
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSlider, QMessageBox, QProgressDialog, QShortcut
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

        controlLayout.addWidget(self.slider)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.stopButton)
        controlLayout.addWidget(QLabel("Current / Go to:"))
        controlLayout.addWidget(self.frameJumpEdit)
        controlLayout.addWidget(self.centerViewBtn)
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
        self.leftVideoLabel = QLabel("Left Eye Video")
        self.leftVideoLabel.setFixedSize(320, 240)
        self.rightVideoLabel = QLabel("Right Eye Video")
        self.rightVideoLabel.setFixedSize(320, 240)
        videoLayout.addWidget(self.leftVideoLabel)
        videoLayout.addWidget(self.rightVideoLabel)
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
        
        # Get paths using the custom module.
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
        self.frameJumpEdit.setText("0")

        # Enable processing controls
        self.medianWinEdit.setEnabled(True)
        self.applyMedianBtn.setEnabled(True)
        self.maxGapEdit.setEnabled(True)
        self.fillGapsBtn.setEnabled(True)

        self.loaded = True
        
        # Display the first frame and plot the pupil properties.
        self.updateFrame()
        self.plotPupilProperties()

    def _ensure_qc_field(self, eyedat):
        n = len(eyedat.get('x', []))
        if 'QC' not in eyedat or eyedat['QC'] is None or len(eyedat['QC']) != n:
            eyedat['QC'] = np.zeros(n, dtype=int)

    def overlay_plot(self, frame, position, eyeDat):
        """
        Draw the pupil circle unless any needed value is NaN.
        """
        if np.isnan(eyeDat['x'][position]) or np.isnan(eyeDat['y'][position]) or np.isnan(eyeDat['radius'][position]):
            return frame
        color = (0, 0, 255)
        center = (int(eyeDat['x'][position]), int(eyeDat['y'][position]))
        radius = int(eyeDat['radius'][position])
        frame = cv2.circle(frame, center, radius, color, 2)
        return frame

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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame[:, :, 0]
            frame[frame >= np.percentile(frame, 70)] = np.percentile(frame, 70)
            min_val = np.min(frame)
            max_val = np.max(frame)
            frame = (frame - min_val) / (max_val - min_val) * 255
            frame = np.stack((frame,) * 3, axis=-1).astype(np.uint8)
            frame = self.overlay_plot(frame, frame_position, eyedat)
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
