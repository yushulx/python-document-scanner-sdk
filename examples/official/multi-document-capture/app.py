import os
import sys
import math
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas

from PySide6.QtCore import (
    Qt, QTimer, QThreadPool, QRunnable, Signal, QSize, QPoint, QRect, QPropertyAnimation, QObject,
)
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont, QPolygonF, QAction,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QStackedWidget, QScrollArea, QFrame,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox, QDialog,
    QDialogButtonBox, QSlider, QCheckBox, QMenu, QProgressDialog, QSpacerItem,
    QSizePolicy, QProgressBar, QGroupBox, QRadioButton, QTextEdit,
)

from dynamsoft_capture_vision_bundle import EnumErrorCode

from scanner import (
    DCVScanner, Page, QuadPoint, DEFAULT_LICENSE_KEY, DETECT_TEMPLATE, NORMALIZE_TEMPLATE,
    apply_filter, rotate_image_90, create_default_quad_points, is_quad_stable,
    polygon_area, points_to_bounding_box, build_stitched_image, save_image, load_image,
    copy_image,
)


# ── Qt helpers ──


def np_to_qpixmap(image: np.ndarray) -> QPixmap:
    """Convert RGB numpy array to QPixmap."""
    h, w = image.shape[:2]
    bytes_per_line = 3 * w
    q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_image.copy())


def np_to_qimage(image: np.ndarray) -> QImage:
    h, w = image.shape[:2]
    bytes_per_line = 3 * w
    return QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def scale_to_fit(image_size: QSize, widget_size: QSize) -> Tuple[float, float, float]:
    """Return (scale, offset_x, offset_y) to fit image in widget, centered."""
    if widget_size.width() <= 0 or widget_size.height() <= 0 or image_size.width() <= 0 or image_size.height() <= 0:
        return 1.0, 0.0, 0.0
    img_aspect = image_size.width() / image_size.height()
    widget_aspect = widget_size.width() / widget_size.height()
    if widget_aspect > img_aspect:
        scale = widget_size.width() / image_size.width()
        offset_y = (widget_size.height() - image_size.height() * scale) / 2
        offset_x = 0.0
    else:
        scale = widget_size.height() / image_size.height()
        offset_x = (widget_size.width() - image_size.width() * scale) / 2
        offset_y = 0.0
    return scale, offset_x, offset_y


# ── Worker signals and workers ──


class WorkerSignals(QObject):
    result = Signal(object)
    finished = Signal()
    error = Signal(str)


class DetectWorker(QRunnable):
    def __init__(self, scanner: DCVScanner, image: np.ndarray):
        super().__init__()
        self.scanner = scanner
        self.image = image
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            quad = self.scanner.detect_document(self.image)
            self.signals.result.emit(quad)
        except Exception as e:
            print(f"Detection worker error: {e}")
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class NormalizeWorker(QRunnable):
    def __init__(self, scanner: DCVScanner, image: np.ndarray, quad_points: Optional[List[QuadPoint]]):
        super().__init__()
        self.scanner = scanner
        self.image = image
        self.quad_points = quad_points
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            normalized = self.scanner.normalize_document(self.image, self.quad_points)
            self.signals.result.emit(normalized)
        except Exception as e:
            print(f"Normalization worker error: {e}")
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


class ProcessFileWorker(QRunnable):
    def __init__(self, scanner: DCVScanner, file_path: str):
        super().__init__()
        self.scanner = scanner
        self.file_path = file_path
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            image = load_image(self.file_path)
            if image is None:
                self.signals.result.emit((None, None, None))
                return
            quad = self.scanner.detect_document(image)
            normalized = None
            if quad:
                normalized = self.scanner.normalize_document(image, quad)
            self.signals.result.emit((image, quad, normalized))
        except Exception as e:
            print(f"Process file worker error: {e}")
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


# ── Custom widgets ──


class CameraWidget(QLabel):
    """Displays camera frame with optional document quad overlay."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(320, 240)
        self.raw_frame: Optional[np.ndarray] = None
        self.display_pixmap: Optional[QPixmap] = None
        self.display_scale: float = 1.0
        self.overlay_points: Optional[List[QuadPoint]] = None
        self.overlay_color = QColor(106, 196, 187)
        self.auto_capture_indicator = False
        self.max_display_width = 1280

    def set_frame(self, image: np.ndarray):
        self.raw_frame = image
        self._update_display_pixmap()
        self.update()

    def _update_display_pixmap(self):
        if self.raw_frame is None:
            self.display_pixmap = None
            return
        h, w = self.raw_frame.shape[:2]
        if w > self.max_display_width:
            new_w = self.max_display_width
            new_h = max(1, int(h * new_w / w))
            display_img = cv2.resize(self.raw_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            display_img = self.raw_frame
            new_w, new_h = w, h
        self.display_pixmap = np_to_qpixmap(display_img)
        self.display_scale = new_w / w

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.raw_frame is not None:
            self._update_display_pixmap()
        self.update()

    def set_overlay(self, points: Optional[List[QuadPoint]]):
        self.overlay_points = points
        self.update()

    def show_auto_capture(self):
        self.auto_capture_indicator = True
        QTimer.singleShot(1500, self._hide_indicator)
        self.update()

    def _hide_indicator(self):
        self.auto_capture_indicator = False
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.display_pixmap is not None:
            img_size = self.display_pixmap.size()
            widget_size = self.size()
            fit_scale, offset_x, offset_y = scale_to_fit(img_size, widget_size)
            target_w = img_size.width() * fit_scale
            target_h = img_size.height() * fit_scale
            target_rect = QRect(int(offset_x), int(offset_y), int(target_w), int(target_h))
            painter.drawPixmap(target_rect, self.display_pixmap)

            # Draw overlay
            if self.overlay_points and len(self.overlay_points) == 4:
                total_scale = self.display_scale * fit_scale
                polygon = QPolygonF()
                for p in self.overlay_points:
                    x = p.x * total_scale + offset_x
                    y = p.y * total_scale + offset_y
                    polygon.append(QPoint(x, y))
                painter.setBrush(QBrush(QColor(106, 196, 187, 50)))
                pen = QPen(self.overlay_color)
                pen.setWidth(3)
                painter.setPen(pen)
                painter.drawPolygon(polygon)
        else:
            painter.setPen(QPen(Qt.white))
            painter.drawText(self.rect(), Qt.AlignCenter, "Camera not started")

        # Auto capture indicator
        if self.auto_capture_indicator:
            painter.setPen(QPen(Qt.NoPen))
            painter.setBrush(QBrush(QColor(0, 0, 0, 140)))
            rect = self.rect()
            painter.drawRect(rect)
            painter.setPen(QPen(Qt.white))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(rect, Qt.AlignCenter, "Document auto-captured")


class CaptureButton(QPushButton):
    """Shutter button with visual feedback for capturing."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(72, 72)
        self.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
            }
        """)
        self.capturing = False
        self._press_anim = 0.0

    def set_capturing(self, capturing: bool):
        self.capturing = capturing
        self.update()

    def mousePressEvent(self, event):
        self._press_anim = 1.0
        self.update()
        super().mousePressEvent(event)
        QTimer.singleShot(100, lambda: self._release_press_anim())

    def _release_press_anim(self):
        self._press_anim = 0.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = self.rect().center()
        outer_radius = 34
        inner_radius = 26

        # Outer ring
        painter.setPen(QPen(Qt.NoPen))
        painter.setBrush(QBrush(QColor(255, 255, 255, 64)))
        painter.drawEllipse(center, outer_radius, outer_radius)

        # Press shrink
        if self._press_anim > 0:
            inner_radius = int(inner_radius * 0.85)

        # Inner circle
        if self.capturing:
            color = QColor(254, 142, 20)  # Orange
        else:
            color = QColor(255, 255, 255)  # White
        painter.setBrush(QBrush(color))
        painter.drawEllipse(center, inner_radius, inner_radius)


class EditWidget(QLabel):
    """Interactive widget for editing document quad corners."""
    quad_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #161616;")
        self.original_image: Optional[np.ndarray] = None
        self.display_pixmap: Optional[QPixmap] = None
        self.display_scale: float = 1.0
        self.quad_points: List[QuadPoint] = []
        self.dragging_corner = -1
        self.handle_radius = 18
        self.setMouseTracking(True)
        self.max_display_width = 1280

    def set_image(self, image: np.ndarray):
        self.original_image = image
        self._update_display_pixmap()
        self.update()

    def _update_display_pixmap(self):
        if self.original_image is None:
            self.display_pixmap = None
            return
        h, w = self.original_image.shape[:2]
        if w > self.max_display_width:
            new_w = self.max_display_width
            new_h = max(1, int(h * new_w / w))
            display_img = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            display_img = self.original_image
            new_w, new_h = w, h
        self.display_pixmap = np_to_qpixmap(display_img)
        self.display_scale = new_w / w

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_image is not None:
            self._update_display_pixmap()
        self.update()

    def set_quad_points(self, points: List[QuadPoint]):
        self.quad_points = points
        self.update()

    def get_image_rect(self) -> QRect:
        if self.original_image is None or self.display_pixmap is None:
            return QRect()
        img_size = self.display_pixmap.size()
        widget_size = self.size()
        fit_scale, offset_x, offset_y = scale_to_fit(img_size, widget_size)
        return QRect(
            int(offset_x),
            int(offset_y),
            int(img_size.width() * fit_scale),
            int(img_size.height() * fit_scale),
        )

    def _screen_to_display(self, pos: QPoint) -> Tuple[float, float]:
        """Map screen position to display pixmap coordinates."""
        img_rect = self.get_image_rect()
        if img_rect.width() == 0 or img_rect.height() == 0:
            return 0.0, 0.0
        x = (pos.x() - img_rect.x()) / img_rect.width() * self.display_pixmap.width()
        y = (pos.y() - img_rect.y()) / img_rect.height() * self.display_pixmap.height()
        return x, y

    def _display_to_image(self, x: float, y: float) -> Tuple[float, float]:
        """Map display pixmap coordinates to original image coordinates."""
        img_w = self.display_pixmap.width() / self.display_scale
        img_h = self.display_pixmap.height() / self.display_scale
        return x / self.display_scale, y / self.display_scale

    def _find_corner(self, pos: QPoint) -> int:
        if self.original_image is None or not self.quad_points or self.display_pixmap is None:
            return -1
        img_rect = self.get_image_rect()
        if img_rect.width() == 0 or img_rect.height() == 0:
            return -1
        disp_x, disp_y = self._screen_to_display(pos)
        for i, p in enumerate(self.quad_points):
            px = p.x * self.display_scale
            py = p.y * self.display_scale
            if math.hypot(px - disp_x, py - disp_y) < 30:
                return i
        return -1

    def _to_image_coords(self, pos: QPoint) -> Tuple[float, float]:
        disp_x, disp_y = self._screen_to_display(pos)
        return self._display_to_image(disp_x, disp_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.original_image is not None:
            corner = self._find_corner(event.position().toPoint())
            if corner >= 0:
                self.dragging_corner = corner
                self.update()

    def mouseMoveEvent(self, event):
        if self.dragging_corner >= 0 and self.original_image is not None:
            x, y = self._to_image_coords(event.position().toPoint())
            img_w = self.original_image.shape[1]
            img_h = self.original_image.shape[0]
            x = max(0, min(img_w, x))
            y = max(0, min(img_h, y))
            self.quad_points[self.dragging_corner] = QuadPoint(x, y)
            self.quad_changed.emit()
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_corner = -1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.display_pixmap is not None:
            img_rect = self.get_image_rect()
            painter.drawPixmap(img_rect, self.display_pixmap)

            if len(self.quad_points) == 4:
                polygon = QPolygonF()
                for p in self.quad_points:
                    x = img_rect.x() + (p.x * self.display_scale) * (img_rect.width() / self.display_pixmap.width())
                    y = img_rect.y() + (p.y * self.display_scale) * (img_rect.height() / self.display_pixmap.height())
                    polygon.append(QPoint(x, y))
                painter.setBrush(QBrush(QColor(106, 196, 187, 50)))
                pen = QPen(QColor(106, 196, 187))
                pen.setWidth(3)
                painter.setPen(pen)
                painter.drawPolygon(polygon)

                for i, p in enumerate(self.quad_points):
                    x = img_rect.x() + (p.x * self.display_scale) * (img_rect.width() / self.display_pixmap.width())
                    y = img_rect.y() + (p.y * self.display_scale) * (img_rect.height() / self.display_pixmap.height())
                    color = QColor(254, 142, 20) if i == self.dragging_corner else QColor(106, 196, 187)
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(Qt.white, 3))
                    painter.drawEllipse(QPoint(x, y), self.handle_radius, self.handle_radius)
                    # Draw index number
                    painter.setPen(QPen(Qt.black, 1))
                    painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
                    painter.drawText(QRect(int(x) - 10, int(y) - 10, 20, 20), Qt.AlignCenter, str(i + 1))
        else:
            painter.setPen(QPen(Qt.white))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image to edit")


class ThumbnailButton(QFrame):
    """Widget showing a page thumbnail with a visible delete button."""
    clicked_index = Signal(int)
    remove_index = Signal(int)

    def __init__(self, index: int, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.index = index
        self.setFixedSize(72, 90)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #4a4a4a;
                background: #3f3f3f;
                border-radius: 4px;
            }
            QFrame:hover {
                border: 1px solid #6ac4bb;
            }
        """)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(0, 0, 72, 90)
        self.image_label.setAlignment(Qt.AlignCenter)
        pixmap = np_to_qpixmap(image).scaled(70, 88, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        self.delete_btn = QPushButton("×", self)
        self.delete_btn.setGeometry(48, 0, 24, 24)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 0, 0, 0.6);
                color: white;
                border: none;
                border-radius: 0px;
                font-size: 16px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background: rgba(254, 70, 20, 0.85);
            }
        """)
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.clicked.connect(lambda: self.remove_index.emit(self.index))

        self.image_label.mousePressEvent = self._on_image_click

    def _on_image_click(self, event):
        self.clicked_index.emit(self.index)


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stabilization Settings")
        self.setMinimumWidth(360)
        self.setStyleSheet("""
            QDialog { background: #2b2b2b; color: white; }
            QLabel { color: white; }
            QSlider::groove:horizontal { height: 6px; background: #4a4a4a; border-radius: 3px; }
            QSlider::handle:horizontal { background: #6ac4bb; width: 16px; margin: -5px 0; border-radius: 8px; }
            QCheckBox { color: white; }
        """)
        self.settings = dict(settings)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        self.auto_checkbox = QCheckBox("Auto Capture")
        self.auto_checkbox.setChecked(self.settings.get("enabled", True))
        layout.addWidget(self.auto_checkbox)

        self.iou_slider = self._create_slider(50, 100, int(self.settings.get("iou_threshold", 0.85) * 100), 1)
        self.iou_label = QLabel(f"IoU Threshold: {self.iou_slider.value() / 100:.2f}")
        layout.addWidget(self.iou_label)
        layout.addWidget(self.iou_slider)

        self.area_slider = self._create_slider(1, 50, int(self.settings.get("area_delta_threshold", 0.15) * 100), 1)
        self.area_label = QLabel(f"Area Delta Threshold: {self.area_slider.value() / 100:.2f}")
        layout.addWidget(self.area_label)
        layout.addWidget(self.area_slider)

        self.stable_slider = self._create_slider(1, 10, self.settings.get("stable_frame_count", 3), 1)
        self.stable_label = QLabel(f"Stable Frame Count: {self.stable_slider.value()}")
        layout.addWidget(self.stable_label)
        layout.addWidget(self.stable_slider)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.auto_checkbox.stateChanged.connect(self._update)
        self.iou_slider.valueChanged.connect(self._update)
        self.area_slider.valueChanged.connect(self._update)
        self.stable_slider.valueChanged.connect(self._update)

    def _create_slider(self, min_v, max_v, value, step):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(value)
        slider.setSingleStep(step)
        return slider

    def _update(self):
        self.settings["enabled"] = self.auto_checkbox.isChecked()
        self.settings["iou_threshold"] = self.iou_slider.value() / 100
        self.settings["area_delta_threshold"] = self.area_slider.value() / 100
        self.settings["stable_frame_count"] = self.stable_slider.value()
        self.iou_label.setText(f"IoU Threshold: {self.settings['iou_threshold']:.2f}")
        self.area_label.setText(f"Area Delta Threshold: {self.settings['area_delta_threshold']:.2f}")
        self.stable_label.setText(f"Stable Frame Count: {self.settings['stable_frame_count']}")

    def get_settings(self):
        return self.settings


class SortDialog(QDialog):
    def __init__(self, pages: List[Page], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reorder Pages")
        self.setMinimumSize(360, 400)
        self.setStyleSheet("""
            QDialog { background: #2b2b2b; color: white; }
            QListWidget { background: #343434; border: none; }
            QListWidget::item { background: #2b2b2b; border-radius: 6px; margin: 4px; padding: 4px; }
        """)
        self.order = list(range(len(pages)))
        self.pages = pages
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        hint = QLabel("Drag to reorder pages")
        hint.setStyleSheet("color: #999999; padding: 8px;")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QListWidget.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.setSpacing(4)
        self._populate()
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self):
        self.list_widget.clear()
        for idx in self.order:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, idx)
            item.setText(f"Page {idx + 1}")
            item.setSizeHint(QSize(80, 60))
            # Thumbnail icon
            thumb = apply_filter(self.pages[idx].base_image, self.pages[idx].filter_mode)
            pixmap = np_to_qpixmap(thumb).scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item.setIcon(pixmap)
            self.list_widget.addItem(item)

    def _on_rows_moved(self, parent, start, end, destination, row):
        # Reconstruct order based on current list
        new_order = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            new_order.append(item.data(Qt.UserRole))
        self.order = new_order

    def get_order(self):
        return self.order


# ── Main application window ──


class DocumentScannerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamsoft Document Scanner")
        self.setMinimumSize(900, 700)
        self.setStyleSheet("""
            QMainWindow { background: #2b2b2b; }
            QWidget { color: white; font-family: "Segoe UI", "Helvetica Neue", Arial; }
            QPushButton { background: #343434; border: 1px solid #4a4a4a; border-radius: 6px; padding: 8px 16px; }
            QPushButton:hover { background: #4a4a4a; }
            QPushButton:disabled { opacity: 0.4; }
            QPushButton#primary { background: #fe8e14; border: none; }
            QPushButton#primary:hover { background: #ff9f33; }
            QLineEdit { background: #252525; border: 1px solid #5a5a5a; border-radius: 6px; padding: 8px; color: white; }
            QLabel { color: white; }
            QScrollArea { border: none; }
        """)

        self.scanner = DCVScanner()
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)

        self.pages: List[Page] = []
        self.current_page_index = 0
        self.retake_index = -1

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self._on_camera_frame)
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self._on_detection_tick)

        self.is_scanning = False
        self.is_capture_in_progress = False
        self.is_processing_frame = False
        self.cool_down = False
        self.manual_capture_pending = False
        self.capture_timeout: Optional[QTimer] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_detected_quad: Optional[List[QuadPoint]] = None
        self.last_quad: Optional[List[QuadPoint]] = None
        self.stable_counter = 0
        self._gallery_import_active = False

        self.quad_stabilizer = {
            "enabled": True,
            "iou_threshold": 0.85,
            "area_delta_threshold": 0.15,
            "stable_frame_count": 3,
        }

        self._build_ui()
        self._show_license_screen()

    def _build_ui(self):
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.license_screen = self._build_license_screen()
        self.stack.addWidget(self.license_screen)

        self.scanner_screen = self._build_scanner_screen()
        self.stack.addWidget(self.scanner_screen)

        self.result_screen = self._build_result_screen()
        self.stack.addWidget(self.result_screen)

        self.edit_screen = self._build_edit_screen()
        self.stack.addWidget(self.edit_screen)

        self.statusBar().showMessage("Ready")

    def _build_license_screen(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)

        box = QWidget()
        box.setStyleSheet("background: #343434; border: 1px solid #4a4a4a; border-radius: 14px; padding: 22px;")
        box_layout = QVBoxLayout(box)
        box_layout.setSpacing(12)

        title = QLabel("Dynamsoft Document Scanner")
        title.setStyleSheet("font-size: 26px; font-weight: bold;")
        box_layout.addWidget(title)

        subtitle = QLabel("Enter your license key to start scanning documents.")
        subtitle.setStyleSheet("color: #d0d0d0; font-size: 14px;")
        subtitle.setWordWrap(True)
        box_layout.addWidget(subtitle)

        label = QLabel("License Key")
        label.setStyleSheet("font-size: 13px; color: #d8d8d8;")
        box_layout.addWidget(label)

        self.license_input = QLineEdit()
        self.license_input.setPlaceholderText("Paste license key (leave empty to use default)")
        self.license_input.setText(DEFAULT_LICENSE_KEY)
        box_layout.addWidget(self.license_input)

        self.activate_btn = QPushButton("Start Scanner")
        self.activate_btn.setObjectName("primary")
        self.activate_btn.setStyleSheet("background: #fe8e14; border: none; padding: 10px; font-weight: bold;")
        self.activate_btn.clicked.connect(self._on_activate)
        box_layout.addWidget(self.activate_btn)

        self.init_progress = QProgressBar()
        self.init_progress.setRange(0, 0)
        self.init_progress.setVisible(False)
        box_layout.addWidget(self.init_progress)

        self.init_status = QLabel("")
        self.init_status.setStyleSheet("color: #999999;")
        self.init_status.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(self.init_status)

        layout.addWidget(box)
        return widget

    def _build_scanner_screen(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Camera view
        self.camera_widget = CameraWidget()
        layout.addWidget(self.camera_widget, 1)

        # Thumbnail bar
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setFixedHeight(100)
        self.thumbnail_scroll.setStyleSheet("background: rgba(0,0,0,0.55);")
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QHBoxLayout(self.thumbnail_container)
        self.thumbnail_layout.setSpacing(8)
        self.thumbnail_layout.setContentsMargins(8, 8, 8, 8)
        self.thumbnail_layout.addStretch()
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        self.thumbnail_scroll.setVisible(False)
        layout.addWidget(self.thumbnail_scroll)

        # Bottom controls
        bottom = QWidget()
        bottom.setFixedHeight(100)
        bottom.setStyleSheet("background: rgba(0,0,0,0.55);")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(24, 16, 24, 16)

        self.gallery_btn = QPushButton("🖼")
        self.gallery_btn.setFixedSize(48, 48)
        self.gallery_btn.setStyleSheet("background: transparent; border: none; font-size: 24px;")
        self.gallery_btn.clicked.connect(self._on_open_file)
        bottom_layout.addWidget(self.gallery_btn)

        bottom_layout.addStretch()

        self.capture_btn = CaptureButton()
        self.capture_btn.clicked.connect(self._on_manual_capture)
        bottom_layout.addWidget(self.capture_btn)

        bottom_layout.addStretch()

        self.next_btn = QPushButton("➜")
        self.next_btn.setFixedSize(48, 48)
        self.next_btn.setStyleSheet("background: transparent; border: none; font-size: 24px;")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._on_show_result)
        bottom_layout.addWidget(self.next_btn)

        layout.addWidget(bottom)

        # Settings button (top right overlay)
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(40, 40)
        self.settings_btn.setStyleSheet("background: rgba(52,52,52,0.75); border: none; border-radius: 8px; font-size: 22px;")
        self.settings_btn.clicked.connect(self._on_open_settings)
        self.settings_btn.setParent(self.camera_widget)
        self.settings_btn.move(12, 12)
        self.settings_btn.setVisible(True)
        self.settings_btn.raise_()

        # Retake indicator (top center overlay)
        self.retake_indicator = QLabel("Retake mode")
        self.retake_indicator.setStyleSheet("""
            background: rgba(254, 142, 20, 0.85);
            color: white;
            padding: 6px 14px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 600;
        """)
        self.retake_indicator.setParent(self.camera_widget)
        self.retake_indicator.move(60, 12)
        self.retake_indicator.setVisible(False)
        self.retake_indicator.raise_()

        return widget

    def _build_result_screen(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_bar = QWidget()
        top_bar.setFixedHeight(56)
        top_bar.setStyleSheet("background: #343434;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setSpacing(0)
        top_layout.setContentsMargins(0, 0, 0, 0)

        actions = [
            ("Continue", self._on_continue),
            ("Retake", self._on_retake),
            ("Edit", self._on_edit),
            ("Rotate", self._on_rotate),
            ("Sort", self._on_sort),
            ("Save ▼", self._on_save_menu),
        ]
        for label, handler in actions:
            btn = QPushButton(label)
            btn.setStyleSheet("background: transparent; border: none; font-size: 12px; font-weight: 600; padding: 10px 8px;")
            btn.clicked.connect(handler)
            top_layout.addWidget(btn)
            if label == "Edit":
                self.edit_btn = btn
            elif label == "Save ▼":
                self.save_btn = btn

        layout.addWidget(top_bar)

        self.page_indicator = QLabel("1 / 1")
        self.page_indicator.setFixedHeight(32)
        self.page_indicator.setAlignment(Qt.AlignCenter)
        self.page_indicator.setStyleSheet("color: #999999; font-size: 14px;")
        layout.addWidget(self.page_indicator)

        result_view = QWidget()
        result_view_layout = QHBoxLayout(result_view)
        result_view_layout.setContentsMargins(6, 6, 6, 6)
        result_view_layout.setSpacing(6)

        self.prev_page_btn = QPushButton("‹")
        self.prev_page_btn.setFixedSize(40, 80)
        self.prev_page_btn.setStyleSheet("""
            QPushButton {
                background: #5a5a5a;
                border: 2px solid #808080;
                border-radius: 8px;
                color: white;
                font-size: 32px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #707070;
            }
            QPushButton:disabled {
                background: #3c3c3c;
                border: 2px solid #4a4a4a;
                color: #808080;
            }
        """)
        self.prev_page_btn.clicked.connect(self._on_prev_page)
        result_view_layout.addWidget(self.prev_page_btn)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background: #161616;")
        result_view_layout.addWidget(self.result_label, 1)

        self.next_page_btn = QPushButton("›")
        self.next_page_btn.setFixedSize(40, 80)
        self.next_page_btn.setStyleSheet("""
            QPushButton {
                background: #5a5a5a;
                border: 2px solid #808080;
                border-radius: 8px;
                color: white;
                font-size: 32px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #707070;
            }
            QPushButton:disabled {
                background: #3c3c3c;
                border: 2px solid #4a4a4a;
                color: #808080;
            }
        """)
        self.next_page_btn.clicked.connect(self._on_next_page)
        result_view_layout.addWidget(self.next_page_btn)

        layout.addWidget(result_view, 1)

        filter_bar = QWidget()
        filter_bar.setFixedHeight(64)
        filter_bar.setStyleSheet("background: #343434;")
        filter_layout = QHBoxLayout(filter_bar)
        filter_layout.setSpacing(6)
        filter_layout.setContentsMargins(6, 6, 6, 10)

        self.filter_color = QPushButton("Color")
        self.filter_gray = QPushButton("Grayscale")
        self.filter_binary = QPushButton("Binary")
        for btn in (self.filter_color, self.filter_gray, self.filter_binary):
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #999999;
                    border: none;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: 600;
                }
                QPushButton:checked {
                    color: #fe8e14;
                    background: rgba(254, 142, 20, 0.12);
                }
            """)
            filter_layout.addWidget(btn)

        self.filter_color.clicked.connect(lambda: self._on_filter("color"))
        self.filter_gray.clicked.connect(lambda: self._on_filter("grayscale"))
        self.filter_binary.clicked.connect(lambda: self._on_filter("binary"))

        layout.addWidget(filter_bar)
        return widget

    def _build_edit_screen(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("Edit Quad")
        header.setFixedHeight(56)
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("background: #343434; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        self.edit_widget = EditWidget()
        layout.addWidget(self.edit_widget, 1)

        footer = QWidget()
        footer.setFixedHeight(56)
        footer.setStyleSheet("background: #343434;")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setSpacing(0)
        footer_layout.setContentsMargins(0, 0, 0, 0)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background: transparent; color: #999999; border: none; font-size: 16px; font-weight: 600;")
        cancel_btn.clicked.connect(self._on_cancel_edit)
        footer_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("background: transparent; color: #fe8e14; border: none; font-size: 16px; font-weight: 600;")
        apply_btn.clicked.connect(self._on_apply_edit)
        footer_layout.addWidget(apply_btn)

        layout.addWidget(footer)
        return widget

    def _show_license_screen(self):
        self.stack.setCurrentWidget(self.license_screen)

    def _show_scanner_screen(self):
        self.stack.setCurrentWidget(self.scanner_screen)
        self.next_btn.setEnabled(self.retake_index < 0 and len(self.pages) > 0)
        self.gallery_btn.setEnabled(True)
        self.thumbnail_scroll.setVisible(self.retake_index < 0 and len(self.pages) > 0)
        if self.cap and self.cap.isOpened():
            self.camera_timer.start(33)
        self.settings_btn.setParent(self.camera_widget)
        self.settings_btn.move(12, 12)
        self.settings_btn.raise_()
        self.settings_btn.setVisible(True)
        self.retake_indicator.setParent(self.camera_widget)
        self.retake_indicator.setVisible(self.retake_index >= 0)
        self.retake_indicator.raise_()

    def _show_result_screen(self):
        self.stack.setCurrentWidget(self.result_screen)
        self._stop_scanning()
        self.camera_timer.stop()
        self._render_result()

    def _show_edit_screen(self):
        if not self.pages:
            return
        page = self.pages[self.current_page_index]
        if page.original_image is None:
            self._show_toast("No source image available for editing.")
            return
        if page.quad_points and len(page.quad_points) == 4:
            points = [QuadPoint(p.x, p.y) for p in page.quad_points]
        else:
            h, w = page.original_image.shape[:2]
            points = create_default_quad_points(w, h)
        self.edit_widget.set_image(page.original_image)
        self.edit_widget.set_quad_points(points)
        self._stop_scanning()
        self.camera_timer.stop()
        self.stack.setCurrentWidget(self.edit_screen)

    def _on_activate(self):
        key = self.license_input.text().strip() or DEFAULT_LICENSE_KEY
        self.activate_btn.setEnabled(False)
        self.init_progress.setVisible(True)
        self.init_status.setText("Activating license...")

        QTimer.singleShot(100, lambda: self._init_sdk(key))

    def _init_sdk(self, key: str):
        try:
            self.scanner = DCVScanner(license_key=key)
            ec, msg = self.scanner.init()
            if ec != EnumErrorCode.EC_OK:
                raise RuntimeError(f"License initialization failed: {msg}")
            self._init_camera()
            self.init_status.setText("Opening camera...")
            self._show_scanner_screen()
            self._start_scanning()
        except Exception as e:
            self._show_toast(f"Initialization failed: {e}")
            self.activate_btn.setEnabled(True)
            self.init_progress.setVisible(False)
            self.init_status.setText(str(e))

    def _init_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            # Try any backend
            self.cap = cv2.VideoCapture(0)
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camera_timer.start(33)  # ~30 fps preview

    def _on_camera_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.latest_frame = rgb
        self.camera_widget.set_frame(rgb)

    def _on_detection_tick(self):
        if not self.is_scanning or self.is_processing_frame or self.is_capture_in_progress:
            return
        if self.latest_frame is None:
            return
        self.is_processing_frame = True
        # Downscale for detection speed
        h, w = self.latest_frame.shape[:2]
        scale = min(1.0, 640 / w)
        if scale < 1.0:
            small = cv2.resize(self.latest_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = self.latest_frame.copy()

        worker = DetectWorker(self.scanner, small)
        worker.signals.result.connect(lambda quad: self._on_detection_result(quad, scale))
        worker.signals.error.connect(lambda msg: self._on_detection_result(None, scale))
        self.thread_pool.start(worker)

    def _on_detection_result(self, quad: Optional[List[QuadPoint]], scale: float):
        self.is_processing_frame = False
        if quad:
            # Scale points back to original frame size
            if scale < 1.0:
                full_quad = [QuadPoint(p.x / scale, p.y / scale) for p in quad]
            else:
                full_quad = quad
            self.latest_detected_quad = full_quad
            self.camera_widget.set_overlay(full_quad)

            if self.manual_capture_pending:
                self.manual_capture_pending = False
                if self.capture_timeout is not None:
                    try:
                        self.capture_timeout.stop()
                    except Exception:
                        pass
                    self.capture_timeout = None
                self._reset_stabilizer()
                self._perform_capture(False, full_quad)
                return

            if self.last_quad is None:
                self.last_quad = full_quad
                self.stable_counter = 1
            elif is_quad_stable(full_quad, self.last_quad, self.quad_stabilizer["iou_threshold"], self.quad_stabilizer["area_delta_threshold"]):
                self.stable_counter += 1
                self.last_quad = full_quad
            else:
                self.stable_counter = 0
                self.last_quad = full_quad

            if self.quad_stabilizer["enabled"] and self.stable_counter >= self.quad_stabilizer["stable_frame_count"]:
                self._reset_stabilizer()
                self._perform_capture(True, full_quad)
        else:
            self.camera_widget.set_overlay(None)
            self._reset_stabilizer()

    def _reset_stabilizer(self):
        self.stable_counter = 0
        self.last_quad = None
        self.latest_detected_quad = None

    def _on_manual_capture(self):
        if self.cool_down or self.is_capture_in_progress:
            return
        self.manual_capture_pending = True
        self.capture_btn.set_capturing(True)
        if self.capture_timeout is None:
            self.capture_timeout = QTimer(self)
            self.capture_timeout.setSingleShot(True)
            self.capture_timeout.timeout.connect(self._manual_capture_timeout)
        self.capture_timeout.start(500)

    def _manual_capture_timeout(self):
        if self.manual_capture_pending:
            self.manual_capture_pending = False
            self._perform_capture(False, None)
        else:
            # Capture was already triggered by detection; make sure button resets
            if not self.is_capture_in_progress:
                self.capture_btn.set_capturing(False)
        self.capture_timeout = None

    def _perform_capture(self, auto_captured: bool, quad_points: Optional[List[QuadPoint]]):
        if self.is_capture_in_progress or self.cool_down or self.latest_frame is None:
            self.capture_btn.set_capturing(False)
            return
        self.is_capture_in_progress = True
        self.capture_btn.set_capturing(True)
        frame = self.latest_frame.copy()
        original = frame.copy()

        finished = False
        safety_timeout = QTimer(self)
        safety_timeout.setSingleShot(True)

        def _do_finish():
            nonlocal finished
            if finished:
                return
            finished = True
            safety_timeout.stop()
            self._finish_capture()

        def on_normalized(normalized):
            if finished:
                return
            if normalized is not None:
                self._add_page(normalized, original, quad_points)
                if auto_captured:
                    self.camera_widget.show_auto_capture()
            else:
                if not auto_captured:
                    self._show_toast("No document detected. Using original image.")
                self._add_page(original, original, quad_points)
            _do_finish()

        def on_error(msg: str):
            if finished:
                return
            print(f"Capture normalization failed: {msg}")
            self._show_toast("Capture failed. Please try again.")
            _do_finish()

        def on_timeout():
            if finished:
                return
            print("Capture normalization timed out")
            self._show_toast("Capture timed out. Please try again.")
            _do_finish()

        safety_timeout.timeout.connect(on_timeout)
        safety_timeout.start(5000)

        worker = NormalizeWorker(self.scanner, original, quad_points if (quad_points and len(quad_points) == 4) else None)
        worker.signals.result.connect(on_normalized)
        worker.signals.error.connect(on_error)
        self.thread_pool.start(worker)

    def _finish_capture(self):
        self.is_capture_in_progress = False
        self.capture_btn.set_capturing(False)
        if not self._gallery_import_active:
            self.cool_down = True
            QTimer.singleShot(1500, lambda: setattr(self, "cool_down", False))

    def _add_page(self, base_image: np.ndarray, original_image: Optional[np.ndarray], quad_points: Optional[List[QuadPoint]]):
        page = Page(
            base_image=copy_image(base_image),
            original_image=copy_image(original_image) if original_image is not None else None,
            quad_points=[QuadPoint(p.x, p.y) for p in quad_points] if quad_points else None,
            filter_mode="color",
        )
        if self.retake_index >= 0:
            self.pages[self.retake_index] = page
            self.current_page_index = self.retake_index
            self.retake_index = -1
            self._show_result_screen()
        else:
            self.pages.append(page)
            self._update_thumbnail_bar()
            self.next_btn.setEnabled(True)
            self.thumbnail_scroll.setVisible(True)

    def _update_thumbnail_bar(self):
        # Remove existing thumbnail buttons
        while self.thumbnail_layout.count() > 1:
            item = self.thumbnail_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, page in enumerate(self.pages):
            thumb = apply_filter(page.base_image, page.filter_mode)
            btn = ThumbnailButton(i, thumb)
            btn.clicked_index.connect(self._on_thumbnail_click)
            btn.remove_index.connect(self._on_thumbnail_remove)
            self.thumbnail_layout.insertWidget(i, btn)

    def _on_thumbnail_click(self, index: int):
        self.current_page_index = index
        self._show_result_screen()

    def _on_thumbnail_remove(self, index: int):
        if 0 <= index < len(self.pages):
            del self.pages[index]
            if self.current_page_index >= len(self.pages):
                self.current_page_index = max(0, len(self.pages) - 1)
            self._update_thumbnail_bar()
            self.next_btn.setEnabled(len(self.pages) > 0)
            self.thumbnail_scroll.setVisible(len(self.pages) > 0)

    def _on_show_result(self):
        if not self.pages:
            return
        self.current_page_index = len(self.pages) - 1
        self._stop_scanning()
        self._show_result_screen()

    def _on_continue(self):
        self._show_scanner_screen()
        self._start_scanning()

    def _on_retake(self):
        if not self.pages:
            return
        self.retake_index = self.current_page_index
        self._show_scanner_screen()
        self._start_scanning()
        self.retake_indicator.setText(f"Retake: replace page {self.retake_index + 1}")
        self._show_toast(f"Retake mode: capture or load an image to replace page {self.retake_index + 1}.")

    def _on_edit(self):
        self._show_edit_screen()

    def _on_rotate(self):
        if not self.pages:
            return
        page = self.pages[self.current_page_index]
        page.base_image = rotate_image_90(page.base_image)
        self._render_result()
        self._update_thumbnail_bar()

    def _on_sort(self):
        if len(self.pages) < 2:
            self._show_toast("Need at least 2 pages to reorder.")
            return
        dialog = SortDialog(self.pages, self)
        if dialog.exec() == QDialog.Accepted:
            order = dialog.get_order()
            self.pages = [self.pages[i] for i in order]
            self.current_page_index = 0
            self._render_result()
            self._update_thumbnail_bar()

    def _on_save_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("background: #2a2a2a; border: 1px solid #545454; border-radius: 10px; padding: 6px;")
        export_pdf = QAction("Export as PDF", self)
        export_images = QAction("Export as Images", self)
        export_long = QAction("Export as Long Image", self)
        export_pdf.triggered.connect(self._on_export_pdf)
        export_images.triggered.connect(self._on_export_images)
        export_long.triggered.connect(self._on_export_long_image)
        menu.addAction(export_pdf)
        menu.addAction(export_images)
        menu.addAction(export_long)
        menu.exec(self.save_btn.mapToGlobal(self.save_btn.rect().bottomRight()))

    def _on_export_pdf(self):
        if not self.pages:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "documents.pdf", "PDF Files (*.pdf)")
        if not path:
            return
        try:
            pdf = pdf_canvas.Canvas(path, pagesize=A4)
            page_width, page_height = A4
            for i, page in enumerate(self.pages):
                if i > 0:
                    pdf.showPage()
                img = apply_filter(page.base_image, page.filter_mode)
                h, w = img.shape[:2]
                ratio = min(page_width / w, page_height / h)
                draw_w = w * ratio
                draw_h = h * ratio
                x = (page_width - draw_w) / 2
                y = (page_height - draw_h) / 2
                # Save temp image
                temp_path = f"_temp_pdf_{i}.jpg"
                save_image(img, temp_path)
                pdf.drawImage(temp_path, x, y, width=draw_w, height=draw_h)
            pdf.save()
            # Cleanup temp files
            for i in range(len(self.pages)):
                temp = f"_temp_pdf_{i}.jpg"
                if os.path.exists(temp):
                    os.remove(temp)
            self._show_toast("PDF exported.")
        except Exception as e:
            self._show_toast(f"PDF export failed: {e}")

    def _on_export_images(self):
        if not self.pages:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, page in enumerate(self.pages):
            img = apply_filter(page.base_image, page.filter_mode)
            path = os.path.join(folder, f"document_{stamp}_{i + 1}.png")
            save_image(img, path)
        self._show_toast("Images exported.")

    def _on_export_long_image(self):
        if not self.pages:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Long Image", "document_stitched.png", "PNG Images (*.png)")
        if not path:
            return
        try:
            self._show_toast("Stitching images...", 3000)
            filtered = [apply_filter(p.base_image, p.filter_mode) for p in self.pages]
            stitched, matched = build_stitched_image(filtered)
            if stitched is None:
                return
            save_image(stitched, path)
            msg = f"Long image exported ({matched} matched overlaps)."
            self._show_toast(msg)
        except Exception as e:
            self._show_toast(f"Long image export failed: {e}")

    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not path:
            return
        self._gallery_import_active = True
        self._stop_scanning()
        self._show_toast("Processing image...")

        def on_done(image, quad, normalized):
            if image is None:
                self._show_toast("Failed to load image.")
                if self.stack.currentWidget() == self.scanner_screen:
                    self._show_scanner_screen()
                    self._start_scanning()
                    self._on_detection_tick()
                self._gallery_import_active = False
                return
            if normalized is not None:
                self._add_page(normalized, image, quad)
                self._show_toast("Document detected and captured.")
            else:
                self._show_toast("No document detected. Original image saved.")
                self._add_page(image, image, None)
            # Resume real-time document detection if still on the scanner screen
            if self.stack.currentWidget() == self.scanner_screen:
                self._show_scanner_screen()
                self._start_scanning()
                self._on_detection_tick()
            self._gallery_import_active = False
            self.cool_down = False

        worker = ProcessFileWorker(self.scanner, path)
        worker.signals.result.connect(lambda args: on_done(*args))
        self.thread_pool.start(worker)

    def _on_open_settings(self):
        dialog = SettingsDialog(self.quad_stabilizer, self)
        if dialog.exec() == QDialog.Accepted:
            self.quad_stabilizer = dialog.get_settings()

    def _on_filter(self, mode: str):
        if not self.pages:
            return
        self.pages[self.current_page_index].filter_mode = mode
        self._render_result()
        self._update_thumbnail_bar()

    def _on_prev_page(self):
        if self.current_page_index > 0:
            self.current_page_index -= 1
            self._render_result()

    def _on_next_page(self):
        if self.current_page_index < len(self.pages) - 1:
            self.current_page_index += 1
            self._render_result()

    def _on_cancel_edit(self):
        self._show_result_screen()

    def _on_apply_edit(self):
        if not self.pages:
            return
        page = self.pages[self.current_page_index]
        new_points = self.edit_widget.quad_points
        original = page.original_image

        def on_done(normalized):
            if normalized is not None:
                page.base_image = normalized
                page.quad_points = [QuadPoint(p.x, p.y) for p in new_points]
                self._show_result_screen()
                self._render_result()
                self._update_thumbnail_bar()
            else:
                self._show_toast("Normalization failed. Adjust corners and try again.")

        worker = NormalizeWorker(self.scanner, original, new_points)
        worker.signals.result.connect(on_done)
        self.thread_pool.start(worker)

    def _render_result(self):
        if not self.pages:
            return
        self.current_page_index = max(0, min(self.current_page_index, len(self.pages) - 1))
        page = self.pages[self.current_page_index]
        img = apply_filter(page.base_image, page.filter_mode)
        pixmap = np_to_qpixmap(img)
        self.result_label.setPixmap(pixmap.scaled(self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.page_indicator.setText(f"{self.current_page_index + 1} / {len(self.pages)}")
        self.prev_page_btn.setEnabled(self.current_page_index > 0)
        self.next_page_btn.setEnabled(self.current_page_index < len(self.pages) - 1)
        self.edit_btn.setEnabled(page.original_image is not None)

        self.filter_color.setChecked(page.filter_mode == "color")
        self.filter_gray.setChecked(page.filter_mode == "grayscale")
        self.filter_binary.setChecked(page.filter_mode == "binary")

    def _start_scanning(self):
        if self.is_scanning:
            return
        self.is_scanning = True
        self.detection_timer.start(500)  # Detect twice per second to keep UI responsive

    def _stop_scanning(self):
        self.is_scanning = False
        self.manual_capture_pending = False
        self.is_processing_frame = False
        self.detection_timer.stop()
        self.camera_widget.set_overlay(None)
        self._reset_stabilizer()

    def _show_toast(self, message: str, duration: int = 2000):
        self.statusBar().showMessage(message, duration)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.stack.currentWidget() == self.result_screen:
            self._render_result()
        if self.camera_widget:
            self.settings_btn.setParent(self.camera_widget)
            self.settings_btn.move(12, 12)
            self.settings_btn.raise_()
            self.settings_btn.setVisible(self.stack.currentWidget() == self.scanner_screen)

    def closeEvent(self, event):
        self._stop_scanning()
        self.camera_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.thread_pool.waitForDone(2000)
        event.accept()


# ── Entry point ──


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = DocumentScannerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
