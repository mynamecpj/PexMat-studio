import copy
import os
import sys
import shutil
import hashlib
import subprocess
import platform
import time
import traceback
import collections
import uuid
import gc
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import av
from PySide6.QtMultimedia import QMediaPlayer

try:
    import torch
except ImportError:
    torch = None

from PySide6.QtCore import (
    Qt, Signal, QPointF, QPoint, QRect, QRectF, QEvent, QTimer, QSizeF, QSize, Slot,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, Property,
    QSequentialAnimationGroup, QUrl, QThread, QMimeData, QByteArray, QObject, QMetaObject,
    QEventLoop
)
from PySide6.QtGui import (
    QPixmap, QCursor, QResizeEvent, QPainter, QPaintEvent, QColor, QPen,
    QPainterPath, QBrush, QMouseEvent, QFontMetrics, QWheelEvent,
    QDragEnterEvent, QDropEvent, QTransform, QDrag, QImage
)
import PySide6.QtGui as qg
from PySide6.QtWidgets import (
    QApplication, QLabel, QSizePolicy, QWidget, QScrollBar, QStyle, QVBoxLayout,
    QHBoxLayout, QToolButton, QButtonGroup, QSpinBox, QAbstractSpinBox,
    QMessageBox, QFrame, QGraphicsDropShadowEffect, QPushButton,
    QStackedWidget, QListWidget, QScrollArea, QGroupBox, QRadioButton, QSlider,
    QCheckBox, QGraphicsOpacityEffect, QAbstractItemView, QFileDialog,
    QListWidgetItem, QGridLayout, QStyleOptionViewItem, QColorDialog, QSpacerItem,
    QProgressDialog, QStackedLayout, QDoubleSpinBox, QGraphicsBlurEffect
)

from ui.components.dialogs import ProgressDialog
from core.utils import convert_cv_to_pixmap, imread_unicode
from config.settings import (
    MIN_ZOOM, MAX_ZOOM, ZOOM_FACTOR, SUPPORTED_VIDEO_FORMATS,
    DEFAULT_MASK_ALPHA_VIDEO, VIDEO_DEFAULT_FPS, VIDEO_TARGET_COLORS,
    VIDEO_CLICK_PREDICT_MASK_ALPHA, VIDEO_OBJ_COLORS, VIDEO_POINT_RADIUS_IMG,
    VIDEO_PLAYBACK_INTERVAL_MS, C_ICON_COLOR_TOOL, TEMP_BASE_DIR, _TR
)

# ==========================================
# Modern video editor dark theme constants
# ==========================================
VID_DARK_BG = "#181818"
VID_PANEL_BG = "#262626"
VID_BORDER = "#333333"
VID_TEXT_PRIMARY = "#E0E0E0"
VID_TEXT_SECONDARY = "#888888"
VID_ACCENT = "#1A73E8"

VID_MODERN_BTN_STYLE = f"""
    QToolButton, QPushButton {{
        background-color: transparent;
        color: {VID_TEXT_PRIMARY};
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 13px;
        border: 1px solid transparent;
    }}
    QToolButton:hover, QPushButton:hover {{
        background-color: #383838;
    }}
    QToolButton:pressed, QPushButton:pressed {{
        background-color: #404040;
    }}
    QToolButton:checked {{
        color: {VID_ACCENT};
        background-color: rgba(26, 115, 232, 0.15);
    }}
"""

VID_ACTION_BTN_STYLE = f"""
    QPushButton#ActionBtn {{
        background-color: transparent;
        color: {VID_TEXT_PRIMARY};
        border-radius: 4px;
        font-weight: bold;
        padding: 6px 12px;
        font-size: 13px;
        border: 1px solid {VID_BORDER};
    }}
    QPushButton#ActionBtn:hover {{
        background-color: #333333;
    }}
"""

VID_PRIMARY_ACTION_BTN_STYLE = f"""
    QPushButton#PrimaryActionBtn {{
        background-color: {VID_ACCENT};
        color: #FFFFFF;
        border-radius: 4px;
        font-weight: bold;
        padding: 6px 16px;
        font-size: 13px;
        border: none;
    }}
    QPushButton#PrimaryActionBtn:hover {{
        background-color: #1B66C9;
    }}
"""

VID_SLIDER_STYLE = f"""
    QSlider {{ min-height: 24px; background: transparent; }}
    QSlider::groove:horizontal {{
        border: none;
        height: 4px;
        background: #333333;
        border-radius: 2px;
        margin: 0px 0px;
    }}
    QSlider::sub-page:horizontal {{
        background: {VID_ACCENT};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: #FFFFFF;
        border: 2px solid {VID_ACCENT};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {VID_ACCENT};
        border: 2px solid #FFFFFF;
        width: 16px;
        height: 16px;
        margin: -6px -1px;
        border-radius: 8px;
    }}
"""

VID_SEGMENT_RADIO_STYLE = f"""
    QRadioButton {{
        background-color: #242424;
        color: #888888;
        border-radius: 6px;
        padding: 6px 0px;
        font-weight: bold;
        font-size: 12px;
    }}
    QRadioButton::indicator {{
        width: 0px; height: 0px; border: none; background: transparent;
    }}
    QRadioButton:hover {{
        background-color: #2A2A2A;
        color: #BBBBBB;
    }}
    QRadioButton:checked {{
        background-color: #383838;
        color: #FFFFFF;
    }}
"""

VID_TIMELINE_SLIDER_STYLE = f"""
    QSlider {{ min-height: 24px; background: transparent; }}
    QSlider::groove:horizontal {{
        border: none;
        height: 4px;
        background: #404040;
        border-radius: 2px;
        margin: 0px 0px;
    }}
    QSlider::sub-page:horizontal {{
        background: {VID_ACCENT};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {VID_ACCENT};
        border: none;
        width: 12px;
        height: 12px;
        margin: -4px 0;
        border-radius: 6px;
    }}
    QSlider::handle:horizontal:hover {{
        background: #4C8BF5;
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
"""


class VideoSimpleTimeline(QSlider):
    frame_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("VidTimeline")
        self.setStyleSheet(VID_TIMELINE_SLIDER_STYLE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.valueChanged.connect(self._on_value_changed)
        self.sliderReleased.connect(self._on_slider_released)
        self._is_updating = False

    def set_params(self, total_frames, *args, **kwargs):
        # 【核心修复】：使用 blockSignals(True) 强制阻断 Qt 内部在重设 Range 及 Value 时
        # 产生的一切延迟/异步 valueChanged 信号，防止其穿透并引起视口回弹至第 0 帧。
        self.blockSignals(True)
        self._is_updating = True
        self.setMinimum(0)
        self.setMaximum(max(0, total_frames - 1))
        self.setValue(0)
        self._is_updating = False
        self.blockSignals(False)

    def set_current_frame(self, frame_idx):
        self.blockSignals(True)
        self._is_updating = True
        self.setValue(frame_idx)
        self._is_updating = False
        self.blockSignals(False)

    def set_info_text(self, time_str, frame_str):
        pass

    def _on_value_changed(self, val):
        if not self._is_updating:
            self.frame_selected.emit(val)

    def _on_slider_released(self):
        if not self._is_updating:
            self.frame_selected.emit(self.value())


class AspectRatioContainer(QWidget):
    def __init__(self, child_widget, parent=None):
        super().__init__(parent)
        self.child_widget = child_widget
        self.child_widget.setParent(self)
        self.aspect_ratio = 16.0 / 9.0
        self.setStyleSheet("background: transparent;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.height()

        target_w = w
        target_h = int(w / self.aspect_ratio)

        if target_h > h:
            target_h = h
            target_w = int(h * self.aspect_ratio)

        x = (w - target_w) // 2
        y = (h - target_h) // 2
        self.child_widget.setGeometry(x, y, target_w, target_h)


class LibraryItemWidget(QWidget):
    def __init__(self, file_path, thumbnail_pixmap, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setFixedSize(170, 104)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: transparent;")

        self.container = QWidget(self)
        self.container.setGeometry(5, 10, 160, 90)
        self.container.setStyleSheet("background-color: transparent;")

        self.img_label = QLabel(self.container)
        self.img_label.setGeometry(0, 0, 160, 90)
        self.img_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        if thumbnail_pixmap and not thumbnail_pixmap.isNull():
            icon_pix = QPixmap(160, 90)
            icon_pix.fill(Qt.GlobalColor.transparent)
            painter = QPainter(icon_pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            path = QPainterPath()
            path.addRoundedRect(0, 0, 160, 90, 6, 6)
            painter.setClipPath(path)
            scaled = thumbnail_pixmap.scaled(160, 90, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                             Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(0, 0, scaled)
            painter.end()
            self.img_label.setPixmap(icon_pix)
        else:
            self.img_label.setStyleSheet("border-radius: 6px; background-color: #202020;")

        self.shadow_effect = QGraphicsDropShadowEffect(self.container)
        self.shadow_effect.setColor(QColor(0, 0, 0, 180))
        self.shadow_effect.setBlurRadius(6)
        self.shadow_effect.setOffset(0, 2)
        self.container.setGraphicsEffect(self.shadow_effect)

        self.anim_group = QParallelAnimationGroup(self)
        self.pos_anim = QPropertyAnimation(self.container, b"geometry")
        self.pos_anim.setDuration(160)
        self.pos_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.shadow_blur_anim = QPropertyAnimation(self.shadow_effect, b"blurRadius")
        self.shadow_blur_anim.setDuration(160)
        self.shadow_blur_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.shadow_offset_anim = QPropertyAnimation(self.shadow_effect, b"offset")
        self.shadow_offset_anim.setDuration(160)
        self.shadow_offset_anim.setEasingCurve(QEasingCurve.Type.OutQuad)

        self.anim_group.addAnimation(self.pos_anim)
        self.anim_group.addAnimation(self.shadow_blur_anim)
        self.anim_group.addAnimation(self.shadow_offset_anim)

    def enterEvent(self, event):
        self.anim_group.stop()
        self.pos_anim.setStartValue(self.container.geometry())
        self.pos_anim.setEndValue(QRect(5, 4, 160, 90))
        self.shadow_blur_anim.setStartValue(self.shadow_effect.blurRadius())
        self.shadow_blur_anim.setEndValue(15)
        self.shadow_offset_anim.setStartValue(self.shadow_effect.offset())
        self.shadow_offset_anim.setEndValue(QPointF(0, 6))
        self.anim_group.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.anim_group.stop()
        self.pos_anim.setStartValue(self.container.geometry())
        self.pos_anim.setEndValue(QRect(5, 10, 160, 90))
        self.shadow_blur_anim.setStartValue(self.shadow_effect.blurRadius())
        self.shadow_blur_anim.setEndValue(6)
        self.shadow_offset_anim.setStartValue(self.shadow_effect.offset())
        self.shadow_offset_anim.setEndValue(QPointF(0, 2))
        self.anim_group.start()
        super().leaveEvent(event)


class StoryboardItemWidget(QWidget):
    def __init__(self, file_path, duration_sec, thumbnail_pixmap, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setFixedSize(240, 170)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: transparent;")

        self.container = QWidget(self)
        self.container.setGeometry(0, 17, 240, 135)
        self.container.setStyleSheet("background-color: transparent;")

        self.bg_label = QLabel(self.container)
        self.bg_label.setGeometry(0, 0, 240, 135)
        self.bg_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.bg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if thumbnail_pixmap and not thumbnail_pixmap.isNull():
            scaled_pix = thumbnail_pixmap.scaled(240, 135, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                 Qt.TransformationMode.SmoothTransformation)
            self.bg_label.setPixmap(scaled_pix)

        self.bg_label.setStyleSheet(
            "border-radius: 6px; background-color: #1A1A1A; border-bottom: 4px solid transparent;")

        self.shadow_effect = QGraphicsDropShadowEffect(self.container)
        self.shadow_effect.setColor(QColor(0, 0, 0, 160))
        self.shadow_effect.setBlurRadius(8)
        self.shadow_effect.setOffset(0, 3)
        self.container.setGraphicsEffect(self.shadow_effect)

        self.overlay = QWidget(self.container)
        self.overlay.setGeometry(0, 0, 240, 135)
        self.overlay.setStyleSheet("background-color: transparent; border: none;")

        overlay_layout = QVBoxLayout(self.overlay)
        overlay_layout.setContentsMargins(8, 8, 8, 8)

        # Top layout: floating mute button on the left, checkbox on the right
        top_layout = QHBoxLayout()

        # Upgraded floating mute button - Reconfigured with premium sizing and micro-interactions (flat modern styling)
        self.btn_mute = QToolButton()
        self.btn_mute.setFixedSize(28, 28)
        self.btn_mute.setIconSize(QSize(20, 20))
        self.btn_mute.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mute.setStyleSheet("""
            QToolButton { 
                background-color: transparent; 
                border: none; 
                padding: 0px;
            }
            QToolButton:hover { 
                background-color: rgba(255, 255, 255, 0.08); 
                border-radius: 6px;
            }
            QToolButton:pressed {
                background-color: rgba(255, 255, 255, 0.15);
            }
        """)
        self.btn_mute.clicked.connect(self._toggle_mute_state)

        # Initialize icon with the upgraded premium default color scheme
        self._update_mute_icon(False)

        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet("""
            QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 2px solid #FFF; background: rgba(0,0,0,0.5); }
            QCheckBox::indicator:checked { background: #1A73E8; border: none; }
        """)
        self.checkbox.hide()

        top_layout.addWidget(self.btn_mute)
        top_layout.addStretch()
        top_layout.addWidget(self.checkbox)
        overlay_layout.addLayout(top_layout)
        overlay_layout.addStretch()

        bottom_layout = QHBoxLayout()
        self.time_label = QLabel(f"{duration_sec:.1f}s")
        self.time_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.time_label.setStyleSheet(
            "color: white; font-weight: bold; background-color: rgba(0,0,0,0.6); padding: 2px 6px; border-radius: 4px; border: none;")
        bottom_layout.addWidget(self.time_label)
        bottom_layout.addStretch()
        overlay_layout.addLayout(bottom_layout)

        self.anim_group = QParallelAnimationGroup(self)
        self.pos_anim = QPropertyAnimation(self.container, b"geometry")
        self.pos_anim.setDuration(200)
        self.pos_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.shadow_blur_anim = QPropertyAnimation(self.shadow_effect, b"blurRadius")
        self.shadow_blur_anim.setDuration(200)
        self.shadow_blur_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.shadow_offset_anim = QPropertyAnimation(self.shadow_effect, b"offset")
        self.shadow_offset_anim.setDuration(200)
        self.shadow_offset_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.anim_group.addAnimation(self.pos_anim)
        self.anim_group.addAnimation(self.shadow_blur_anim)
        self.anim_group.addAnimation(self.shadow_offset_anim)

    def _update_mute_icon(self, is_muted: bool):
        main_win = QApplication.instance().activeWindow()
        if hasattr(main_win, '_create_svg_icon'):
            icon_name = "volume-mute-fill.svg" if is_muted else "volume-up-fill.svg"
            # Upgrade to premium color aesthetics: sophisticated pastel-coral for mute and slate off-white for active sound
            color = "#F87171" if is_muted else "#F3F4F6"
            self.btn_mute.setIcon(main_win._create_svg_icon(icon_name, size=20, color=color))

    def _toggle_mute_state(self):
        """【MPV版】故事板喇叭按钮：控制当前片段的所有声音（原声 + BGM）"""
        from PySide6.QtWidgets import QApplication
        main_win = QApplication.instance().activeWindow()

        if hasattr(main_win, 'virtual_timeline') and getattr(main_win, 'storyboard_list', None):
            list_widget = main_win.storyboard_list
            idx = -1
            for i in range(list_widget.count()):
                if list_widget.itemWidget(list_widget.item(i)) == self:
                    idx = i
                    break

            if idx != -1:
                clip = main_win.virtual_timeline[idx]

                # 故事板控制的是绝对的全局静音 (mute_all)
                is_currently_muted = clip.get('mute_all', False)
                new_state = not is_currently_muted

                clip['mute_all'] = new_state
                self._update_mute_icon(new_state)

                # 立刻对 MPV 引擎应用全面静音
                if getattr(main_win, '_current_playing_clip_idx', -1) == idx:
                    main_win._sync_audio_engine_to_current_frame(idx)

                # 同步更新其他处于打开状态的面板UI
                if getattr(main_win, '_current_crop_clip_idx', -1) == idx:
                    if hasattr(main_win, '_update_audio_console_ui'):
                        main_win._update_audio_console_ui()

                if getattr(main_win, '_matting_clip_idx', -1) == idx:
                    if hasattr(main_win, 'matting_mute_checkbox'):
                        main_win.matting_mute_checkbox.blockSignals(True)
                        main_win.matting_mute_checkbox.setChecked(new_state)
                        main_win.matting_mute_checkbox.blockSignals(False)

    def sync_ui_with_data(self, is_muted: bool):
        """External call to force sync UI state"""
        self._update_mute_icon(is_muted)

    def set_active(self, is_active: bool):
        if is_active:
            self.bg_label.setStyleSheet(
                "border-radius: 6px; background-color: #1A1A1A; border-bottom: 4px solid #1A73E8;")
        else:
            self.bg_label.setStyleSheet(
                "border-radius: 6px; background-color: #1A1A1A; border-bottom: 4px solid transparent;")

    def enterEvent(self, event):
        self.checkbox.show()
        self.anim_group.stop()
        self.pos_anim.setStartValue(self.container.geometry())
        self.pos_anim.setEndValue(QRect(0, 8, 240, 135))
        self.shadow_blur_anim.setStartValue(self.shadow_effect.blurRadius())
        self.shadow_blur_anim.setEndValue(20)
        self.shadow_offset_anim.setStartValue(self.shadow_effect.offset())
        self.shadow_offset_anim.setEndValue(QPointF(0, 10))
        self.anim_group.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self.checkbox.isChecked():
            self.checkbox.hide()
        self.anim_group.stop()
        self.pos_anim.setStartValue(self.container.geometry())
        self.pos_anim.setEndValue(QRect(0, 17, 240, 135))
        self.shadow_blur_anim.setStartValue(self.shadow_effect.blurRadius())
        self.shadow_blur_anim.setEndValue(8)
        self.shadow_offset_anim.setStartValue(self.shadow_effect.offset())
        self.shadow_offset_anim.setEndValue(QPointF(0, 3))
        self.anim_group.start()
        super().leaveEvent(event)

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()


class AssetLibraryListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(160, 90))
        self.setSpacing(10)
        self.setDragEnabled(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setAcceptDrops(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)

        self.setStyleSheet("""
            QListWidget { background-color: transparent; border: none; outline: 0; }
            QListWidget:focus { outline: 0; border: none; }
            QListWidget::item { background-color: transparent; padding: 0px; margin: 0px; border: none; outline: 0; }
            QListWidget::item:focus { outline: 0; border: none; background-color: transparent; }
            QListWidget::item:hover { background-color: transparent; border: none; outline: 0; }
            QListWidget::item:selected { background-color: transparent; border: none; outline: 0; }
        """)
        self._drag_start_pos = QPoint()

    def viewOptions(self) -> QStyleOptionViewItem:
        options = super().viewOptions()
        options.state &= ~QStyle.StateFlag.State_HasFocus
        return options

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """[Refactor]: Automatically capture HD first frame when dragging from project library and perform 3D sticker-like rotation. Animation is extremely smooth across systems."""
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(event)
            return

        if (event.pos() - self._drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            super().mouseMoveEvent(event)
            return

        item = self.itemAt(self._drag_start_pos)
        if not item:
            super().mouseMoveEvent(event)
            return

        file_path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path:
            super().mouseMoveEvent(event)
            return

        widget = self.itemWidget(item)
        original_pixmap = None

        if widget and hasattr(widget, 'img_label'):
            original_pixmap = widget.img_label.pixmap()

        # Safe fallback: if cover image is not cached, read the first frame of the video in the background as a high-precision drag image
        if not original_pixmap or original_pixmap.isNull():
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    original_pixmap = convert_cv_to_pixmap(frame)
                cap.release()

        # If still empty, create a dark high-quality placeholder background
        if not original_pixmap or original_pixmap.isNull():
            original_pixmap = QPixmap(160, 90)
            original_pixmap.fill(QColor("#202020"))

        w, h = original_pixmap.width(), original_pixmap.height()
        if w <= 0: w = 160
        if h <= 0: h = 90

        final_pixmap = QPixmap(int(w * 1.1 + 10), int(h * 1.1 + 10))
        final_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(final_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        center_x, center_y = final_pixmap.width() / 2, final_pixmap.height() / 2
        painter.translate(center_x, center_y)
        painter.rotate(3.0)  # Slight angle rotation to enhance drag damping feeling
        painter.scale(1.05, 1.04)

        # Clip edges
        clip_path = QPainterPath()
        clip_path.addRoundedRect(int(-w / 2), int(-h / 2), w, h, 6, 6)
        painter.setClipPath(clip_path)

        painter.drawPixmap(int(-w / 2), int(-h / 2), original_pixmap)

        painter.setClipping(False)
        painter.setPen(QPen(QColor(255, 255, 255, 120), 2))
        painter.drawRoundedRect(int(-w / 2), int(-h / 2), w - 1, h - 1, 6, 6)
        painter.end()

        # Precise calculation for coordinate alignment
        click_offset = self._drag_start_pos - self.visualItemRect(item).topLeft()
        scale_x = final_pixmap.width() / float(w)
        scale_y = final_pixmap.height() / float(h)
        hot_x = int(click_offset.x() * scale_x)
        hot_y = int(click_offset.y() * scale_y)

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(file_path)])
        mime_data.setText(file_path)
        drag.setMimeData(mime_data)

        drag.setPixmap(final_pixmap)
        drag.setHotSpot(QPoint(hot_x, hot_y))
        drag.exec(Qt.DropAction.CopyAction)


class VideoTrimSlider(QWidget):
    range_changed = Signal(int, int)
    preview_frame = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(40)
        self.setMinimumWidth(200)
        self.setMouseTracking(True)
        self.min_val = 0
        self.max_val = 100
        self.start_val = 0
        self.end_val = 100
        self.handle_w = 16
        self.active_handle = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pending_preview_val = None

    def set_range(self, min_v, max_v):
        self.min_val = min_v
        self.max_val = max(min_v + 1, max_v)
        self.update()

    def set_values(self, start_v, end_v):
        self.start_val = max(self.min_val, start_v)
        self.end_val = min(self.max_val, end_v)
        self.update()

    def _x_to_val(self, x):
        w = self.width() - self.handle_w * 2
        if w <= 0: return self.min_val
        ratio = (x - self.handle_w) / float(w)
        val = self.min_val + ratio * (self.max_val - self.min_val)
        return int(max(self.min_val, min(self.max_val, val)))

    def _val_to_x(self, val):
        w = self.width() - self.handle_w * 2
        if w <= 0: return self.handle_w
        if self.max_val == self.min_val: return self.handle_w
        ratio = (val - self.min_val) / float(self.max_val - self.min_val)
        return self.handle_w + int(ratio * w)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        track_y = h // 2 - 3
        track_h = 6

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#333333"))
        painter.drawRoundedRect(self.handle_w, track_y, w - self.handle_w * 2, track_h, 3, 3)

        x_start = self._val_to_x(self.start_val)
        x_end = self._val_to_x(self.end_val)

        if x_end > x_start:
            painter.setBrush(QColor(VID_ACCENT))
            painter.drawRoundedRect(x_start, track_y, x_end - x_start, track_h, 3, 3)

        hw = self.handle_w
        hh = 24
        hy = h // 2 - hh // 2

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#4C8BF5"))
        left_rect = QRect(int(x_start - hw), hy, hw, hh)
        painter.drawRoundedRect(left_rect, 4, 4)

        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawLine(int(x_start - hw / 2), hy + 6, int(x_start - hw / 2), hy + hh - 6)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#4C8BF5"))
        right_rect = QRect(int(x_end), hy, hw, hh)
        painter.drawRoundedRect(right_rect, 4, 4)

        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawLine(int(x_end + hw / 2), hy + 6, int(x_end + hw / 2), hy + hh - 6)

    def mousePressEvent(self, event):
        main_win = QApplication.instance().activeWindow()
        if hasattr(main_win, 'is_playing') and getattr(main_win, 'is_playing', False):
            if hasattr(main_win, 'pause_video'):
                main_win.pause_video()

        x = event.position().x()
        x_start = self._val_to_x(self.start_val)
        x_end = self._val_to_x(self.end_val)

        left_center = x_start - self.handle_w / 2
        right_center = x_end + self.handle_w / 2

        if abs(x - left_center) <= 20:
            self.active_handle = 'start'
        elif abs(x - right_center) <= 20:
            self.active_handle = 'end'
        else:
            self.active_handle = None

        if self.active_handle and hasattr(main_win, '_on_timeline_scrub_started'):
            main_win._on_timeline_scrub_started()

    def mouseMoveEvent(self, event):
        if not self.active_handle: return
        val = self._x_to_val(event.position().x())

        if self.active_handle == 'start':
            self.start_val = min(val, self.end_val - 1)
            target_val = self.start_val
        else:
            self.end_val = max(val, self.start_val + 1)
            target_val = self.end_val

        self.update()
        self.range_changed.emit(self.start_val, self.end_val)

        # =========================================================================
        # [Core Throttle Debounce]: Use a single-shot high-frequency timer to control the upper limit of decoding frequency to about 40FPS golden smooth scale
        # Completely solves the problem of main thread queue task backlog caused by fast sliding hand speed
        # =========================================================================
        if not hasattr(self, '_throttle_timer'):
            self._throttle_timer = QTimer(self)
            self._throttle_timer.setSingleShot(True)
            self._throttle_timer.setInterval(25)  # 25ms delay threshold
            self._throttle_timer.timeout.connect(self._on_throttle_timeout)

        self._pending_preview_val = target_val
        if not self._throttle_timer.isActive():
            self._throttle_timer.start()

    def _on_throttle_timeout(self):
        if getattr(self, '_pending_preview_val', None) is not None:
            self.preview_frame.emit(self._pending_preview_val)
            self._pending_preview_val = None

    def mouseReleaseEvent(self, event):
        self.active_handle = None
        main_win = QApplication.instance().activeWindow()
        if hasattr(main_win, '_on_timeline_scrub_finished'):
            main_win._on_timeline_scrub_finished()


class StoryboardTrackWidget(QListWidget):
    item_dropped = Signal(str, int)
    track_reordered = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.setIconSize(QSize(240, 170))
        self.setSpacing(16)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)

        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setDropIndicatorShown(False)

        self.setStyleSheet("""
            QListWidget { background-color: transparent; border: none; outline: none; }
            QListWidget:focus { border: none; outline: none; }
            QListWidget::item { padding: 0px; margin: 0px; background-color: transparent; border: none; outline: none; }
            QListWidget::item:focus { border: none; outline: none; background-color: transparent; }
            QListWidget::item:hover { border: none; outline: none; background-color: transparent; }
            QListWidget::item:selected { border: none; outline: none; background-color: transparent; }

            QScrollBar:horizontal {
                border: none;
                background: transparent;
                height: 8px;
                margin: 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #333333;
                min-width: 40px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #1A73E8;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { border: none; background: none; width: 0px; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
        """)

        self._drag_start_pos = QPoint()
        self._source_row = -1
        self._current_target_row = -1
        self._drag_source = None
        self._anim_overlay = None
        self._proxies = []
        self._orig_x = []

        self._stride = 256
        self._ghost_proxy = None
        self._ghost_anim = None

    def viewOptions(self) -> QStyleOptionViewItem:
        options = super().viewOptions()
        options.state &= ~QStyle.StateFlag.State_HasFocus
        return options

    def _setup_ghost_proxy(self, proxy, size):
        pix = QPixmap(size)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(QColor(26, 115, 232, 120), 2, Qt.PenStyle.DashLine))
        p.setBrush(QColor(26, 115, 232, 20))
        p.drawRoundedRect(1, 17, size.width() - 2, 135, 6, 6)
        p.end()
        proxy.setPixmap(pix)

    def _setup_overlay(self, is_external=False):
        if self._anim_overlay:
            self._teardown_overlay()

        self._drag_source = 'external' if is_external else 'internal'
        self._anim_overlay = QWidget(self.viewport())
        self._anim_overlay.setGeometry(self.viewport().rect())
        self._anim_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._anim_overlay.show()

        self._proxies = []
        self._orig_x = []

        for i in range(self.count()):
            curr_item = self.item(i)
            rect = self.visualItemRect(curr_item)
            self._orig_x.append(rect.x())

            curr_widget = self.itemWidget(curr_item)
            proxy = QLabel(self._anim_overlay)

            if not is_external and i == self._source_row:
                self._setup_ghost_proxy(proxy, rect.size())
                self._ghost_proxy = proxy
            else:
                if curr_widget:
                    proxy.setPixmap(curr_widget.grab())

            proxy.setGeometry(rect)
            proxy.show()
            if curr_widget:
                curr_widget.setHidden(True)

            anim = QPropertyAnimation(proxy, b"pos")
            anim.setDuration(300)
            anim.setEasingCurve(QEasingCurve.Type.OutExpo)
            self._proxies.append({'proxy': proxy, 'anim': anim})

        if is_external:
            self._ghost_proxy = QLabel(self._anim_overlay)
            self._setup_ghost_proxy(self._ghost_proxy, QSize(240, 170))
            self._ghost_proxy.hide()
            self._ghost_anim = QPropertyAnimation(self._ghost_proxy, b"pos")
            self._ghost_anim.setDuration(300)
            self._ghost_anim.setEasingCurve(QEasingCurve.Type.OutExpo)

    def _teardown_overlay(self):
        if self._anim_overlay:
            self._anim_overlay.deleteLater()
            self._anim_overlay = None

        for i in range(self.count()):
            curr_widget = self.itemWidget(self.item(i))
            if curr_widget:
                curr_widget.setHidden(False)

        self._ghost_proxy = None
        self._ghost_anim = None
        self._drag_source = None
        self._source_row = -1
        self._current_target_row = -1

    def _update_positions(self, target_row):
        if self._current_target_row == target_row:
            return
        self._current_target_row = target_row

        base_y = self.visualItemRect(self.item(0)).y() if self.count() > 0 else 0

        for i, p_data in enumerate(self._proxies):
            anim = p_data['anim']
            proxy = p_data['proxy']

            if self._drag_source == 'internal':
                if i == self._source_row:
                    target_x = self._orig_x[target_row]
                else:
                    visual_idx = i
                    if i > self._source_row and i <= target_row:
                        visual_idx = i - 1
                    elif i < self._source_row and i >= target_row:
                        visual_idx = i + 1
                    target_x = self._orig_x[visual_idx]
            else:
                target_x = self._orig_x[i]
                if i >= target_row:
                    target_x += self._stride

            if anim.endValue() is None or anim.endValue().x() != target_x:
                anim.stop()
                anim.setStartValue(proxy.pos())
                anim.setEndValue(QPoint(int(target_x), proxy.y()))
                anim.start()

        if self._drag_source == 'external' and self._ghost_proxy:
            ghost_x = self._orig_x[target_row] if target_row < len(self._orig_x) else (
                self._orig_x[-1] + self._stride if self._orig_x else 16)
            if not self._ghost_proxy.isVisible():
                self._ghost_proxy.setGeometry(int(ghost_x), base_y, 240, 170)
                self._ghost_proxy.show()
            else:
                if self._ghost_anim.endValue() is None or self._ghost_anim.endValue().x() != ghost_x:
                    self._ghost_anim.stop()
                    self._ghost_anim.setStartValue(self._ghost_proxy.pos())
                    self._ghost_anim.setEndValue(QPoint(int(ghost_x), base_y))
                    self._ghost_anim.start()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """[Refactor]: Pre-render ultra-exquisite drag image with 3D tilt, hover and physical rounded corners, completely eliminating QTimer"""
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if (event.pos() - self._drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return

        item = self.itemAt(self._drag_start_pos)
        if not item: return

        self._source_row = self.row(item)
        v_data = item.data(Qt.ItemDataRole.UserRole + 1)
        if not v_data or 'pixmap' not in v_data: return

        raw_pixmap = v_data['pixmap']
        if raw_pixmap is None or raw_pixmap.isNull():
            raw_pixmap = QPixmap(240, 135)
            raw_pixmap.fill(QColor(32, 32, 32, 255))

        # Proportional scaling
        original_pixmap = raw_pixmap.scaled(240, 135, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)

        base_w, base_h = 240, 135
        scale_factor = 1.05
        tilt_angle = 3.0  # Precise tilt of 3 degrees to present a realistic grab effect

        canvas_w = int(base_w * scale_factor * 1.1 + 10)
        canvas_h = int(base_h * scale_factor * 1.1 + 10)

        final_drag_pixmap = QPixmap(canvas_w, canvas_h)
        final_drag_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(final_drag_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Perform 3D geometric hover offset transformation
        cx, cy = canvas_w / 2.0, canvas_h / 2.0
        painter.translate(cx, cy)
        painter.rotate(tilt_angle)
        painter.scale(scale_factor, scale_factor)

        # Clip high-precision rounded corners
        clip_path = QPainterPath()
        clip_path.addRoundedRect(int(-base_w / 2), int(-base_h / 2), base_w, base_h, 8, 8)
        painter.setClipPath(clip_path)

        offset_x = int(-original_pixmap.width() / 2)
        offset_y = int(-original_pixmap.height() / 2)
        painter.drawPixmap(offset_x, offset_y, original_pixmap)

        # Draw translucent slightly bright textured stroke
        painter.setClipping(False)
        painter.setPen(QPen(QColor(255, 255, 255, 120), 2))
        painter.drawRoundedRect(int(-base_w / 2), int(-base_h / 2), base_w - 1, base_h - 1, 8, 8)
        painter.end()

        # Precisely calculate mouse contact hotspot
        click_offset = self._drag_start_pos - self.visualItemRect(item).topLeft()
        scale_x_final = final_drag_pixmap.width() / float(base_w)
        scale_y_final = final_drag_pixmap.height() / float(base_h)
        hot_x = int(click_offset.x() * scale_x_final)
        hot_y = int(click_offset.y() * scale_y_final)

        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setData("application/x-storyboard-item", QByteArray(b"moving"))
        drag.setMimeData(mime_data)

        drag.setPixmap(final_drag_pixmap)
        drag.setHotSpot(QPoint(hot_x, hot_y))

        self._setup_overlay(is_external=False)

        # Smoothly execute operation
        drag.exec(Qt.DropAction.MoveAction)

        if self._drag_source == 'internal':
            self._teardown_overlay()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasFormat("application/x-storyboard-item"):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        elif event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            if self._drag_source != 'external':
                self._setup_overlay(is_external=True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        if self._drag_source == 'external':
            self._teardown_overlay()

    def dragMoveEvent(self, event):
        pos_x = event.position().x()
        target_row = 0

        if self._orig_x:
            target_row = len(self._orig_x)
            for i, x in enumerate(self._orig_x):
                if pos_x < x + (self._stride / 2):
                    target_row = i
                    break
            if self._drag_source == 'internal':
                target_row = min(target_row, len(self._orig_x) - 1)

        if event.mimeData().hasFormat("application/x-storyboard-item"):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
            self._update_positions(target_row)
        elif event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            self._update_positions(target_row)
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        final_row = self._current_target_row

        if event.mimeData().hasFormat("application/x-storyboard-item"):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
            src = self._source_row
            self._teardown_overlay()
            if src != -1 and final_row != -1 and src != final_row:
                self.track_reordered.emit(src, final_row)

        elif event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            self._teardown_overlay()

            path = None
            if event.mimeData().hasUrls():
                for url in event.mimeData().urls():
                    if url.isLocalFile():
                        path = url.toLocalFile()
                        break
            elif event.mimeData().hasText():
                path = event.mimeData().text()
                if not os.path.exists(path):
                    path = None

            if path:
                self.item_dropped.emit(path, final_row)

class BakeSingleClipWorker(QObject):
    """
    单视频片段专属局部烘焙工作线程类。
    仅对指定索引区间的帧进行 Alpha 图层混合与背景重构，
    生成该片段独占的、带有背景合并结果的高速 JPG 帧序列文件夹，避免全局冗余计算。
    """
    progress = Signal(int, str)
    finished = Signal(bool, str)

    def __init__(self, raw_frame_dir, processed_masks, start_frame, end_frame, target_w, target_h,
                 bg_color, custom_bg_path, bg_is_transparent, output_dir):
        super().__init__(None)
        self.raw_frame_dir = raw_frame_dir
        self.processed_masks = processed_masks
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.target_w = target_w
        self.target_h = target_h
        self.bg_color = bg_color
        self.custom_bg_path = custom_bg_path
        self.bg_is_transparent = bg_is_transparent
        self.output_dir = output_dir
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    @Slot()
    def run(self):
        try:
            import os
            import cv2
            import numpy as np
            import shutil
            import gc

            os.makedirs(self.output_dir, exist_ok=True)
            total_frames = self.end_frame - self.start_frame + 1

            # 预载背景图片
            bg_img_canvas = None
            if not self.bg_is_transparent and self.custom_bg_path and os.path.exists(self.custom_bg_path):
                from core.utils import imread_unicode
                bg_img = imread_unicode(self.custom_bg_path, cv2.IMREAD_COLOR)
                if bg_img is not None:
                    bg_img_canvas = cv2.resize(bg_img, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

            for local_idx in range(total_frames):
                if self._is_cancelled:
                    self.finished.emit(False, "Cancelled")
                    return

                global_idx = self.start_frame + local_idx
                out_path = os.path.join(self.output_dir, f"{local_idx:05d}.jpg")
                raw_frame_path = os.path.join(self.raw_frame_dir, f"{global_idx:05d}.jpg")
                if not os.path.exists(raw_frame_path):
                    raw_frame_path = os.path.join(self.raw_frame_dir, f"{global_idx}.jpg")

                frame_masks = self.processed_masks.get(global_idx, {})
                has_mask = False
                for mask_raw in frame_masks.values():
                    if mask_raw is not None and np.any(mask_raw):
                        has_mask = True
                        break

                # 如果没有蒙版且使用的是默认纯绿背景，直接进行极速链接/复制，节省大量 I/O
                is_default_bg = (not self.bg_is_transparent and self.custom_bg_path is None and self.bg_color == QColor(0, 255, 0))
                if not has_mask and is_default_bg:
                    try:
                        if os.path.exists(out_path):
                            os.remove(out_path)
                        os.link(raw_frame_path, out_path)
                    except Exception:
                        shutil.copy2(raw_frame_path, out_path)
                    continue

                # 读取原始提取帧并实施多目标 Alpha 混合
                from core.utils import imread_unicode, imwrite_unicode
                frame_cv = imread_unicode(raw_frame_path, cv2.IMREAD_COLOR)
                if frame_cv is None:
                    frame_cv = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)

                if frame_cv.shape[:2] != (self.target_h, self.target_w):
                    frame_cv = cv2.resize(frame_cv, (self.target_w, self.target_h), interpolation=cv2.INTER_LANCZOS4)

                h, w = frame_cv.shape[:2]
                combined_alpha = np.zeros((h, w), dtype=np.float32)

                for mask_raw in frame_masks.values():
                    if mask_raw is not None:
                        if mask_raw.dtype == bool:
                            mask_float = mask_raw.astype(np.float32)
                        else:
                            mask_float = np.clip(mask_raw.astype(np.float32), 0.0, 1.0)

                        if mask_float.shape[:2] != (h, w):
                            mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)

                        combined_alpha = np.maximum(combined_alpha, mask_float)

                # 重构渲染背景
                if self.bg_is_transparent:
                    checker_size = 20
                    y_indices = (np.arange(h) // checker_size) % 2
                    x_indices = (np.arange(w) // checker_size) % 2
                    grid_mask = (y_indices[:, None] == x_indices[None, :])
                    bg_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    bg_canvas[grid_mask] = [40, 40, 40]
                    bg_canvas[~grid_mask] = [60, 60, 60]
                elif bg_img_canvas is not None:
                    bg_canvas = bg_img_canvas.copy()
                else:
                    bg_bgr = (self.bg_color.blue(), self.bg_color.green(), self.bg_color.red())
                    bg_canvas = np.full((h, w, 3), bg_bgr, dtype=np.uint8)

                alpha_3d = combined_alpha[:, :, np.newaxis]
                blended_frame = frame_cv.astype(np.float32) * alpha_3d + bg_canvas.astype(np.float32) * (1.0 - alpha_3d)
                frame_cv = np.clip(blended_frame, 0.0, 255.0).astype(np.uint8)

                imwrite_unicode(out_path, frame_cv)

                if local_idx % 20 == 0 or local_idx == total_frames - 1:
                    pct = int((local_idx / total_frames) * 100)
                    self.progress.emit(pct, f"正在进行 Alpha 发丝融合... ({local_idx + 1}/{total_frames})")

                if local_idx % 50 == 0:
                    gc.collect()

            self.finished.emit(True, self.output_dir)
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(False, str(e))


class GlobalTimelineExtractor(QThread):
    """
    [Pure PyAV level] High-performance multi-core concurrent video frame extraction engine.
    No dependency on external system FFmpeg. Main thread is responsible for lightning-level packet decoding,
    and asynchronously delegates image scaling, black border filling and disk saving to multi-thread pool for parallel processing, fully utilizing CPU to solve time-consuming issues.
    """
    progress = Signal(int, str)

    def __init__(self, timeline, out_dir, target_width, target_height):
        super().__init__()
        self.timeline = []
        for vid in timeline:
            self.timeline.append({
                'path': vid.get('path', ''),
                'in_point': vid.get('in_point', 0),
                'out_point': vid.get('out_point', vid.get('frames', 1)),
                'frames': vid.get('frames', 0),
                'fps': vid.get('fps', 30.0)
            })
        self.out_dir = out_dir
        self.target_width = target_width
        self.target_height = target_height
        self.success = False
        self.error_msg = ""
        # Intelligently create a thread pool matching the number of CPU cores to achieve multi-thread concurrent image encoding
        self.pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    def _get_clip_hash(self, clip) -> str:
        raw_identity = f"{clip['path']}_{clip['in_point']}_{clip['out_point']}_{clip['fps']}_{self.target_width}_{self.target_height}"
        return hashlib.md5(raw_identity.encode('utf-8')).hexdigest()

    def _safe_link_or_copy(self, src: str, dst: str):
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)

    def _save_frame_async(self, frame_bgr, output_path):
        """Thread pool concurrent task: execute adaptive scaling of picture proportion and black border filling, then write to disk rapidly"""
        try:
            h, w = frame_bgr.shape[:2]
            scale = min(self.target_width / w, self.target_height / h)
            new_w, new_h = int(w * scale), int(h * scale)

            resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Use black background board for Pillarbox/Letterbox filling to avoid image quality distortion or deformation
            canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            x_off = (self.target_width - new_w) // 2
            y_off = (self.target_height - new_h) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

            cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception as e:
            print(f"Async save frame error: {e}")

    def _extract_single_clip_to_cache_pyav_fast(self, clip, cache_dir) -> bool:
        """Pure PyAV lightning-level extraction technology (coupled with multi-core asynchronous pipeline, completely breaks the single-thread disk I/O bottleneck)"""
        os.makedirs(cache_dir, exist_ok=True)
        try:
            container = av.open(clip['path'])
            stream = container.streams.video[0]

            fps = stream.average_rate
            if not fps:
                fps = stream.r_frame_rate
            fps = float(fps) if fps else clip['fps']

            time_base = float(stream.time_base)
            frame_duration_pts = 1.0 / (fps * time_base)

            in_point = clip['in_point']
            frames_to_extract = clip['frames']

            # High-precision fast keyframe localization
            seek_frame = max(0, in_point - 10)
            seek_pts = int(seek_frame * frame_duration_pts)

            try:
                container.seek(seek_pts, stream=stream)
            except Exception:
                container.seek(0, stream=stream)

            local_extracted = 0
            futures = []

            for frame in container.decode(video=0):
                if self.isInterruptionRequested():
                    container.close()
                    return False

                current_idx = int(round(frame.pts * time_base * fps))
                if current_idx >= in_point:
                    # Rapidly convert AVFrame to raw numpy array (main thread only executes this fast operation)
                    bgr_array = frame.to_ndarray(format='bgr24')
                    output_path = os.path.join(cache_dir, f"{(local_extracted + 1):05d}.jpg")

                    # Instantly delegate image encoding, scaling and hard disk writing tasks to multi-thread pool
                    f = self.pool.submit(self._save_frame_async, bgr_array, output_path)
                    futures.append(f)

                    local_extracted += 1
                    if local_extracted >= frames_to_extract:
                        break

            container.close()

            # Blocking synchronous wait for all queued frames in thread pool to finish writing to disk
            for f in futures:
                f.result()

            # Missing frame safe replenishment mechanism
            if local_extracted < frames_to_extract:
                last_valid_path = os.path.join(cache_dir, f"{local_extracted:05d}.jpg")
                for missing in range(frames_to_extract - local_extracted):
                    target_path = os.path.join(cache_dir, f"{(local_extracted + 1):05d}.jpg")
                    if os.path.exists(last_valid_path):
                        shutil.copy2(last_valid_path, target_path)
                    else:
                        cv2.imwrite(target_path, np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8))
                    local_extracted += 1

            return True
        except Exception as e:
            self.error_msg = str(e)
            return False

    def run(self):
        try:
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir, exist_ok=True)

            app_data_path = os.environ.get('APPDATA') or os.path.expanduser('~')
            base_cache_root = os.path.join(app_data_path, "ImageVideoToolbox", "clip_cache")
            os.makedirs(base_cache_root, exist_ok=True)

            global_frame_index = 0
            total_timeline_frames = sum(v['frames'] for v in self.timeline)

            if total_timeline_frames <= 0:
                self.success = True
                return

            for clip_idx, clip in enumerate(self.timeline):
                if self.isInterruptionRequested():
                    raise InterruptedError("操作被用户中止")

                clip_hash = self._get_clip_hash(clip)
                clip_cache_dir = os.path.join(base_cache_root, clip_hash)

                cache_valid = False
                if os.path.exists(clip_cache_dir):
                    cached_files = [f for f in os.listdir(clip_cache_dir) if f.endswith('.jpg')]
                    if len(cached_files) >= clip['frames']:
                        cache_valid = True

                if not cache_valid:
                    self.progress.emit(
                        int((global_frame_index / total_timeline_frames) * 90),
                        f"正在高速并行解复用视频片段 {clip_idx + 1}/{len(self.timeline)}..."
                    )

                    extract_ok = self._extract_single_clip_to_cache_pyav_fast(clip, clip_cache_dir)
                    if not extract_ok:
                        raise RuntimeError(f"片段提取失败: {self.error_msg}")

                # Zero-copy high-speed link to main stream
                for local_frame_idx in range(clip['frames']):
                    if self.isInterruptionRequested():
                        raise InterruptedError("操作被用户中止")

                    src_file = os.path.join(clip_cache_dir, f"{(local_frame_idx + 1):05d}.jpg")
                    dst_file = os.path.join(self.out_dir, f"{global_frame_index:05d}.jpg")

                    self._safe_link_or_copy(src_file, dst_file)
                    global_frame_index += 1

                current_percent = int((global_frame_index / total_timeline_frames) * 100)
                self.progress.emit(min(99, current_percent), "")

            self.success = True
            self.progress.emit(100, "")

        except InterruptedError as ie:
            self.success = False
            self.error_msg = str(ie)
        except Exception as e:
            self.success = False
            self.error_msg = str(e)
            traceback.print_exc()
        finally:
            # Stop and clear thread pool, release VRAM and memory
            self.pool.shutdown(wait=False)


class FullScreenImageOverlay(QWidget):
    """
    全屏柔性非线性图片预览遮罩。
    模拟移动端系统（如 iOS）的柔和弹性阻尼过渡，并伴随磨砂玻璃背景虚化。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_win = parent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # 1. 视口截图标签
        self.bg_label = QLabel(self)
        self.bg_label.setScaledContents(True)

        # 2. 柔和的黑色遮罩，提升对比度
        self.tint_overlay = QWidget(self)
        self.tint_overlay.setStyleSheet("background-color: rgba(10, 10, 10, 0.62);")

        # 3. 背景高斯模糊滤镜
        self.blur_effect = QGraphicsBlurEffect(self)
        self.blur_effect.setBlurRadius(0)
        self.bg_label.setGraphicsEffect(self.blur_effect)

        # 4. 前景放大图展示
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background: transparent; 
                border: 1px solid rgba(255, 255, 255, 0.12); 
                border-radius: 16px;
            }
        """)

        # 5. 透明度淡入淡出滤镜
        self.opacity_effect = QGraphicsOpacityEffect(self.image_label)
        self.image_label.setGraphicsEffect(self.opacity_effect)

        # 6. 并行动画组
        self.anim_group = QParallelAnimationGroup(self)

        # 模糊动画：使用 OutQuint 缓慢减速
        self.blur_anim = QPropertyAnimation(self.blur_effect, b"blurRadius")
        self.blur_anim.setDuration(550)  # 550ms 提供从容的呼吸感
        self.blur_anim.setEasingCurve(QEasingCurve.Type.OutQuint)

        # 缩放几何动画：配置极轻微的物理弹性（非线性柔性核心）
        self.geom_anim = QPropertyAnimation(self.image_label, b"geometry")
        self.geom_anim.setDuration(550)

        # 定制阻尼回弹曲线：大幅减小过冲量 (0.4)，使其保持柔和
        soft_elastic_curve = QEasingCurve(QEasingCurve.Type.OutBack)
        soft_elastic_curve.setAmplitude(0.4)
        self.geom_anim.setEasingCurve(soft_elastic_curve)

        # 透明度渐变
        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_anim.setDuration(450)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.OutQuad)

        self.anim_group.addAnimation(self.blur_anim)
        self.anim_group.addAnimation(self.geom_anim)
        self.anim_group.addAnimation(self.fade_anim)

        self.hide()

    def show_image(self, image_path: str, origin_widget: QWidget = None):
        """
        优雅淡入并漂浮放大图片。
        """
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return

        # 获取父视口截图并更新遮罩范围
        if self.parent_win:
            parent_pix = self.parent_win.grab()
            self.bg_label.setPixmap(parent_pix)
            self.setGeometry(self.parent_win.rect())

        self.bg_label.setGeometry(self.rect())
        self.tint_overlay.setGeometry(self.rect())

        # 计算动画起点与中央落点
        if origin_widget:
            global_rect = origin_widget.rect()
            global_top_left = origin_widget.mapToGlobal(global_rect.topLeft())
            local_top_left = self.mapFromGlobal(global_top_left)
            start_rect = QRect(local_top_left, origin_widget.size())
        else:
            start_rect = QRect(self.width() // 2, self.height() // 2, 0, 0)

        target_rect = self._calculate_target_rect(pixmap.size())

        scaled_pix = pixmap.scaled(
            target_rect.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pix)
        self.image_label.setGeometry(start_rect)

        # 启动入场动画
        self.anim_group.stop()

        # 恢复入场所需的柔和弹性参数
        self.blur_anim.setDuration(550)
        self.blur_anim.setStartValue(0.0)
        self.blur_anim.setEndValue(25.0)

        self.geom_anim.setDuration(550)
        soft_elastic_curve = QEasingCurve(QEasingCurve.Type.OutBack)
        soft_elastic_curve.setAmplitude(0.4)
        self.geom_anim.setEasingCurve(soft_elastic_curve)
        self.geom_anim.setStartValue(start_rect)
        self.geom_anim.setEndValue(target_rect)

        self.fade_anim.setDuration(420)
        self.fade_anim.setStartValue(0.1)
        self.fade_anim.setEndValue(1.0)

        self.show()
        self.raise_()
        self.anim_group.start()

    def hide_image(self):
        """
        顺滑、干净地向中心退场，不带生硬回弹。
        """
        self.anim_group.stop()

        # 退场采用极度平滑的五次方衰减曲线
        exit_curve = QEasingCurve(QEasingCurve.Type.OutQuint)

        self.blur_anim.setDuration(480)
        self.blur_anim.setEasingCurve(exit_curve)
        self.blur_anim.setStartValue(self.blur_effect.blurRadius())
        self.blur_anim.setEndValue(0.0)

        self.fade_anim.setDuration(400)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.fade_anim.setStartValue(self.opacity_effect.opacity())
        self.fade_anim.setEndValue(0.0)

        self.geom_anim.setDuration(480)
        self.geom_anim.setEasingCurve(exit_curve)

        current_geom = self.image_label.geometry()
        target_shrink = QRect(
            current_geom.x() + current_geom.width() // 2,
            current_geom.y() + current_geom.height() // 2,
            0, 0
        )
        self.geom_anim.setStartValue(current_geom)
        self.geom_anim.setEndValue(target_shrink)

        try:
            self.anim_group.finished.disconnect()
        except (TypeError, RuntimeError):
            pass

        self.anim_group.finished.connect(self.hide, Qt.ConnectionType.SingleShotConnection)
        self.anim_group.start()

    def _calculate_target_rect(self, img_size: QSize) -> QRect:
        # 给图片边界预留一定富余量，避免全屏贴边，使其观感更松弛
        w_limit = self.width() - 120
        h_limit = self.height() - 120

        img_w = img_size.width()
        img_h = img_size.height()

        scale = min(w_limit / img_w, h_limit / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        x = (self.width() - new_w) // 2
        y = (self.height() - new_h) // 2
        return QRect(x, y, new_w, new_h)

    def mousePressEvent(self, event):
        self.hide_image()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.bg_label.setGeometry(self.rect())
        self.tint_overlay.setGeometry(self.rect())


class AsyncScrubReader(QObject):
    """
    专属异步片段寻址解码器。
    独立子线程中运行，支持单元素覆写队列与高频拖拽防抖。
    """
    frame_decoded = Signal(object, int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self._current_request_idx = -1
        self._is_busy = False
        self._container = None
        self._stream = None

    @Slot(int)
    def request_frame(self, frame_index):
        """非阻塞式帧解码请求"""
        self._current_request_idx = frame_index
        if not self._is_busy:
            QMetaObject.invokeMethod(self, "_decode_next", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _decode_next(self):
        if self._current_request_idx == -1:
            return

        self._is_busy = True
        target_idx = self._current_request_idx
        self._current_request_idx = -1

        try:
            if self._container is None:
                self._container = av.open(self.video_path)
                self._stream = self._container.streams.video[0]

            container = self._container
            stream = self._stream

            fps = stream.average_rate
            if not fps:
                fps = stream.r_frame_rate
            fps = float(fps) if fps else 30.0

            time_base = float(stream.time_base)
            frame_duration_pts = 1.0 / (fps * time_base)

            seek_frame = max(0, target_idx - 5)
            seek_pts = int(seek_frame * frame_duration_pts)

            try:
                container.seek(seek_pts, stream=stream)
            except Exception:
                container.seek(0, stream=stream)

            frame_arr = None
            for av_frame in container.decode(video=0):
                current_idx = int(round(av_frame.pts * time_base * fps))
                if current_idx == target_idx:
                    frame_arr = av_frame.to_ndarray(format='bgr24')
                    break
                if current_idx > target_idx:
                    break

            if frame_arr is None:
                container.seek(0, stream=stream)
                for av_frame in container.decode(video=0):
                    current_idx = int(round(av_frame.pts * time_base * fps))
                    if current_idx == target_idx:
                        frame_arr = av_frame.to_ndarray(format='bgr24')
                        break
                    if current_idx > target_idx:
                        break

            if frame_arr is not None:
                self.frame_decoded.emit(frame_arr, target_idx)

        except Exception as e:
            print(f"Async decode inner error: {e}")
        finally:
            self._is_busy = False
            if self._current_request_idx != -1:
                QMetaObject.invokeMethod(self, "_decode_next", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def close(self):
        """
        线程安全地释放视频流。
        该方法必须在 AsyncScrubReader 所在的子线程中被调用。
        """
        if self._container:
            try:
                self._container.close()
            except Exception as e:
                print(f"[DEBUG] Error closing av container inside thread: {e}")
            self._container = None
            self._stream = None

class VideoDisplayLabel(QLabel):
        interaction_point_added = Signal(QPointF, int)
        video_points_changed = Signal()
        video_box_drawn = Signal(QRectF)

        def __init__(self, parent_window=None):
            super().__init__(parent_window)
            self.parent_window = parent_window
            self.setObjectName("VideoDisplayLabel")
            self.setProperty("acceptingDrops", False)
            self.setAcceptDrops(True)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.setMinimumSize(480, 360)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            self.setText("拖放视频/GIF文件至此，\n或使用“加载”按钮，\n支持缩放/平移。")
            self.setStyleSheet("color: #9CA3AF;")
            self.setMouseTracking(True)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._is_dragging_over = False

            self.current_pixmap: QPixmap | None = None
            self.current_frame_index: int = -1
            self.allow_interaction: bool = False
            self.interaction_points_for_display: dict[int, list[tuple[float, float, int]]] = {}
            self.interaction_boxes_for_display: dict[int, list[float]] = {}
            self.segmentation_masks_for_display: dict[int, np.ndarray | None] = {}
            self.temp_annotation_frame_mask: np.ndarray | None = None
            self.temp_annotation_target_id: int = -1
            self.temp_annotation_mask_frame_idx: int = -1

            self._is_drawing_box = False
            self._box_start_pos = QPoint()
            self._current_box_rect = QRectF()

            self.active_object_id = -1
            self.interaction_mode = 'sam'

            self.zoom_scale: float = 1.0
            self.view_offset_img: QPointF = QPointF(0.0, 0.0)
            self._is_panning: bool = False
            self._last_pan_pos_widget: QPoint = QPoint()
            self._is_user_zoomed = False

            self._is_in_compare_mode = False
            self._split_ratio = 0.5
            self._is_dragging_splitter = False
            self._splitter_line_rect = QRect()
            self.pixmap_for_compare_original: QPixmap | None = None
            self.pixmap_for_compare_matted: QPixmap | None = None

            self._default_cursor = self.cursor()
            self._cross_cursor = QCursor(Qt.CursorShape.CrossCursor)
            self._forbidden_cursor = QCursor(Qt.CursorShape.ForbiddenCursor)
            self._pan_open_cursor = QCursor(Qt.CursorShape.OpenHandCursor)
            self._pan_closed_cursor = QCursor(Qt.CursorShape.ClosedHandCursor)
            self._size_hor_cursor = QCursor(Qt.CursorShape.SizeHorCursor)

        def set_active_object(self, obj_id: int):
            self.active_object_id = obj_id
            self.update()

        def set_interaction_mode(self, mode: str):
            self.interaction_mode = mode
            self.update_cursor()
            self.update()

        def set_brush_size(self, size: int):
            self.brush_size = size
            self.update()

        def _map_rect_to_image(self, ui_rect: QRectF) -> QRectF | None:
            if not self.current_pixmap: return None
            top_left_img = self.map_widget_to_image_coords(ui_rect.topLeft().toPoint())
            bottom_right_img = self.map_widget_to_image_coords(ui_rect.bottomRight().toPoint())
            if top_left_img and bottom_right_img:
                img_w, img_h = self.current_pixmap.width(), self.current_pixmap.height()
                x1 = max(0.0, min(top_left_img.x(), float(img_w)))
                y1 = max(0.0, min(top_left_img.y(), float(img_h)))
                x2 = max(0.0, min(bottom_right_img.x(), float(img_w)))
                y2 = max(0.0, min(bottom_right_img.y(), float(img_h)))
                return QRectF(x1, y1, x2 - x1, y2 - y1)
            return None

        def begin_mask_shift(self):
            active_id = getattr(self.parent_window, 'current_target_id', -1)
            frame_idx = self.current_frame_index
            if active_id == -1 or frame_idx == -1: return

            # 【修复】：多级优先搜索当前正在生效的蒙版，防止字典指向落空导致滑动无反应
            mask = None
            if self.temp_annotation_target_id == active_id and self.temp_annotation_mask_frame_idx == frame_idx and self.temp_annotation_frame_mask is not None:
                mask = self.temp_annotation_frame_mask
            elif hasattr(self, 'temp_multi_masks') and frame_idx in self.temp_multi_masks and active_id in self.temp_multi_masks[frame_idx]:
                mask = self.temp_multi_masks[frame_idx][active_id]
            elif frame_idx in self.parent_window.processed_masks and active_id in self.parent_window.processed_masks[frame_idx]:
                mask = self.parent_window.processed_masks[frame_idx][active_id]
            elif active_id in self.segmentation_masks_for_display and self.segmentation_masks_for_display[active_id] is not None:
                mask = self.segmentation_masks_for_display[active_id]

            if mask is not None:
                # 记录滑动开始前最原始的纯净蒙版状态，作为扩张/收缩的基准
                self._base_mask_for_shift = mask.copy()

        def apply_mask_shift(self, amount: int):
            active_id = getattr(self.parent_window, 'current_target_id', -1)
            frame_idx = self.current_frame_index
            if active_id == -1 or not hasattr(self, '_base_mask_for_shift'): return

            # 每次拖拽滑动，都基于记录的原始基准蒙版进行计算，确保左右滑动丝滑且不累积误差
            mask = self._base_mask_for_shift
            if amount == 0:
                result_mask = mask.copy()
            else:
                mask_u8 = mask.astype(np.uint8) * 255
                kernel_size = abs(amount) * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                if amount > 0:
                    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
                else:
                    mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
                result_mask = mask_u8 > 127

            # 【核心修复】：将缩放结果直接贯穿写回全局/局部所有的追踪字典，防止在重绘读取时读到旧数据
            if frame_idx not in self.parent_window.processed_masks:
                self.parent_window.processed_masks[frame_idx] = {}
            self.parent_window.processed_masks[frame_idx][active_id] = result_mask

            if not hasattr(self, 'temp_multi_masks'):
                self.temp_multi_masks = {}
            if frame_idx not in self.temp_multi_masks:
                self.temp_multi_masks[frame_idx] = {}
            self.temp_multi_masks[frame_idx][active_id] = result_mask

            if self.temp_annotation_target_id == active_id and self.temp_annotation_mask_frame_idx == frame_idx:
                self.temp_annotation_frame_mask = result_mask

            self.segmentation_masks_for_display[active_id] = result_mask

            # 强制命令主窗口重新提取蒙版与背景进行混合渲染，实现滑块实时视觉反馈
            if hasattr(self.parent_window, '_display_frame_wrapper'):
                self.parent_window._display_frame_wrapper(frame_idx)

        def end_mask_shift(self):
            if hasattr(self, '_base_mask_for_shift'):
                del self._base_mask_for_shift
            # 【脏标记】：通知主窗口当前蒙版已被修改，退出沙盒时需执行重新烘焙
            if hasattr(self.parent_window, '_is_matting_dirty'):
                self.parent_window._is_matting_dirty = True

        def shift_active_mask(self, amount: int):
            active_id = getattr(self.parent_window, 'current_target_id', -1)
            frame_idx = self.current_frame_index
            if active_id == -1 or amount == 0:
                QMessageBox.information(self, "提示", "请先在右侧对象列表中选中一个具体的目标。")
                return

            mask = None
            if self.temp_annotation_target_id == active_id and self.temp_annotation_mask_frame_idx == frame_idx and self.temp_annotation_frame_mask is not None:
                mask = self.temp_annotation_frame_mask
            elif hasattr(self, 'temp_multi_masks') and frame_idx in self.temp_multi_masks and active_id in \
                    self.temp_multi_masks[frame_idx]:
                mask = self.temp_multi_masks[frame_idx][active_id]
            elif frame_idx in self.parent_window.processed_masks and active_id in self.parent_window.processed_masks[
                frame_idx]:
                mask = self.parent_window.processed_masks[frame_idx][active_id]
            elif active_id in self.segmentation_masks_for_display and self.segmentation_masks_for_display[
                active_id] is not None:
                mask = self.segmentation_masks_for_display[active_id]

            if mask is None or not np.any(mask):
                QMessageBox.information(self, "提示", "当前帧尚未生成该目标的蒙版。")
                return
            if hasattr(self.parent_window, '_save_video_state'): self.parent_window._save_video_state()

            mask_u8 = mask.astype(np.uint8) * 255
            kernel_size = abs(amount) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            if amount > 0:
                mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
            else:
                mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)

            result_mask = mask_u8 > 127

            if frame_idx not in self.parent_window.processed_masks:
                self.parent_window.processed_masks[frame_idx] = {}
            self.parent_window.processed_masks[frame_idx][active_id] = result_mask

            if not hasattr(self, 'temp_multi_masks'):
                self.temp_multi_masks = {}
            if frame_idx not in self.temp_multi_masks:
                self.temp_multi_masks[frame_idx] = {}
            self.temp_multi_masks[frame_idx][active_id] = result_mask

            if self.temp_annotation_target_id == active_id and self.temp_annotation_mask_frame_idx == frame_idx:
                self.temp_annotation_frame_mask = result_mask

            self.segmentation_masks_for_display[active_id] = result_mask

            if hasattr(self.parent_window, '_display_frame_wrapper'):
                self.parent_window._display_frame_wrapper(frame_idx)

        def _draw_on_mask(self, img_pos: QPointF, is_press: bool, is_adding: bool = True):
            # 【脏标记】：通知主窗口当前蒙版已被画笔修改
            if hasattr(self.parent_window, '_is_matting_dirty'):
                self.parent_window._is_matting_dirty = True

            active_id = getattr(self.parent_window, 'current_target_id', -1)
            frame_idx = self.current_frame_index
            if active_id == -1: return

            mask = None
            if self.temp_annotation_target_id == active_id and self.temp_annotation_mask_frame_idx == frame_idx and self.temp_annotation_frame_mask is not None:
                mask = self.temp_annotation_frame_mask
            elif hasattr(self, 'temp_multi_masks') and frame_idx in self.temp_multi_masks and active_id in \
                    self.temp_multi_masks[frame_idx]:
                mask = self.temp_multi_masks[frame_idx][active_id]
            elif frame_idx in self.parent_window.processed_masks and active_id in self.parent_window.processed_masks[
                frame_idx]:
                mask = self.parent_window.processed_masks[frame_idx][active_id]
            elif active_id in self.segmentation_masks_for_display and self.segmentation_masks_for_display[
                active_id] is not None:
                mask = self.segmentation_masks_for_display[active_id]

            if mask is None:
                if self.current_pixmap:
                    mask = np.zeros((self.current_pixmap.height(), self.current_pixmap.width()), dtype=bool)
                else:
                    return

            mask_u8 = mask.astype(np.uint8) * 255
            x, y = int(img_pos.x()), int(img_pos.y())
            color = 255 if is_adding else 0

            brush_sz = getattr(self, 'brush_size', 15)
            if hasattr(self.parent_window, 'vid_brush_slider'):
                brush_sz = self.parent_window.vid_brush_slider.value()
            elif hasattr(self.parent_window, 'brush_slider'):
                brush_sz = self.parent_window.brush_slider.value()

            if is_press or getattr(self, '_last_paint_pos', None) is None:
                cv2.circle(mask_u8, (x, y), brush_sz, color, -1)
            else:
                lx, ly = int(self._last_paint_pos.x()), int(self._last_paint_pos.y())
                cv2.line(mask_u8, (lx, ly), (x, y), color, brush_sz * 2)
                cv2.circle(mask_u8, (x, y), brush_sz, color, -1)

            self._last_paint_pos = img_pos
            result_mask = mask_u8 > 127

            # 【核心同步】：画笔绘制后一样贯穿覆盖所有的全局层
            if frame_idx not in self.parent_window.processed_masks:
                self.parent_window.processed_masks[frame_idx] = {}
            self.parent_window.processed_masks[frame_idx][active_id] = result_mask

            if not hasattr(self, 'temp_multi_masks'):
                self.temp_multi_masks = {}
            if frame_idx not in self.temp_multi_masks:
                self.temp_multi_masks[frame_idx] = {}
            self.temp_multi_masks[frame_idx][active_id] = result_mask

            if self.temp_annotation_target_id == active_id and self.temp_annotation_mask_frame_idx == frame_idx:
                self.temp_annotation_frame_mask = result_mask

            self.segmentation_masks_for_display[active_id] = result_mask

            if hasattr(self.parent_window, '_display_frame_wrapper'):
                self.parent_window._display_frame_wrapper(frame_idx)

        def set_frame(self, cv_image: np.ndarray | None, frame_index: int,
                      interaction_pts_for_display: dict | None = None,
                      seg_masks_for_display: dict | None = None,
                      temp_annotation_mask_tuple: tuple[np.ndarray, int, int] | None = None,
                      interaction_boxes_for_display: dict | None = None):
            """
            【完全重写】：零拷贝渲染架构，彻底剔除 convert_cv_to_pixmap 造成的深拷贝性能损耗。
            保留所有传入的字典参数验证与分配逻辑，确保界面其他组件正常运作。
            """
            new_image_set = False
            if cv_image is not None:
                # -------------------------------------------------------------
                # 零拷贝优化：直接将 OpenCV 的 numpy 内存映射到 Qt 的 QImage
                # -------------------------------------------------------------
                height, width = cv_image.shape[:2]

                # 确保格式符合 Qt 支持的 RGB/RGBA
                if len(cv_image.shape) == 2:
                    display_cv = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
                    img_format = QImage.Format.Format_RGB888
                elif cv_image.shape[2] == 4:
                    display_cv = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
                    img_format = QImage.Format.Format_RGBA8888
                else:
                    display_cv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    img_format = QImage.Format.Format_RGB888

                # 必须确保存储连续性
                if not display_cv.flags['C_CONTIGUOUS']:
                    display_cv = np.ascontiguousarray(display_cv)

                bytes_per_line = display_cv.strides[0]

                # 【核心防御】：必须将 numpy 数组挂载到 self，防止被 Python 的垃圾回收器(GC)回收，
                # 导致 QImage 读取到野指针而产生严重的段错误(SegFault)闪退！
                self._current_cv_buffer = display_cv

                # 使用指针直接生成 QImage，跳过深拷贝
                qimg = QImage(self._current_cv_buffer.data, width, height, bytes_per_line, img_format)

                # 转换为 QPixmap
                pixmap = QPixmap.fromImage(qimg)

                if pixmap and not pixmap.isNull():
                    if self.current_pixmap is None or self.current_pixmap.size() != pixmap.size():
                        new_image_set = True
                    self.current_pixmap = pixmap
                    self.setText("")
                else:
                    self.current_pixmap = None
                    self.setText(f"帧 {frame_index + 1}: 图像转换错误")
            else:
                self.current_pixmap = None
                self.setText(f"帧 {frame_index + 1}: 无图像数据")

            self.current_frame_index = frame_index

            # =============================================================
            # 保留原有的变量赋值逻辑（绝对不能省）
            # =============================================================
            if interaction_pts_for_display is not None:
                self.interaction_points_for_display = interaction_pts_for_display
            else:
                self.interaction_points_for_display = {}

            if seg_masks_for_display is not None:
                self.segmentation_masks_for_display = seg_masks_for_display
            else:
                self.segmentation_masks_for_display = {}

            if interaction_boxes_for_display is not None:
                self.interaction_boxes_for_display = interaction_boxes_for_display
            else:
                self.interaction_boxes_for_display = {}

            if temp_annotation_mask_tuple:
                self.temp_annotation_frame_mask, self.temp_annotation_target_id, self.temp_annotation_mask_frame_idx = temp_annotation_mask_tuple
            elif not temp_annotation_mask_tuple or frame_index != self.temp_annotation_mask_frame_idx:
                self.temp_annotation_frame_mask = None
                self.temp_annotation_target_id = -1
                self.temp_annotation_mask_frame_idx = -1

            if new_image_set:
                self._is_user_zoomed = False
                self.fit_frame_to_view()
            else:
                self.update()

            if not self.current_pixmap:
                self.zoom_scale = 1.0
                self.view_offset_img = QPointF(0.0, 0.0)

        def set_compare_mode(self, enabled: bool):
            if self._is_in_compare_mode != enabled:
                self._is_in_compare_mode = enabled
                if self.parent_window and self.current_frame_index != -1:
                    self.parent_window._display_frame_wrapper(self.current_frame_index)
                else:
                    self.update()
                self.update_cursor()

        def set_compare_pixmaps(self, original_overlay: QPixmap, matted_preview: QPixmap):
            self.pixmap_for_compare_original = original_overlay
            self.pixmap_for_compare_matted = matted_preview
            self.update()

        def resizeEvent(self, event: QResizeEvent):
            super().resizeEvent(event)
            pixmap_ref = self.pixmap_for_compare_original if self._is_in_compare_mode and self.pixmap_for_compare_original else self.current_pixmap
            if not self._is_user_zoomed or pixmap_ref is None:
                self.fit_frame_to_view()
                return

            old_size = event.oldSize()
            new_size = self.size()
            if old_size.width() <= 0 or old_size.height() <= 0 or old_size == new_size: return

            old_center_widget = QPoint(old_size.width() // 2, old_size.height() // 2)
            center_content_point = self.map_widget_to_image_coords(old_center_widget)
            if center_content_point is None:
                self.fit_frame_to_view()
                return

            new_center_widget = QPoint(new_size.width() // 2, new_size.height() // 2)
            new_offset_x = center_content_point.x() - (new_center_widget.x() / self.zoom_scale)
            new_offset_y = center_content_point.y() - (new_center_widget.y() / self.zoom_scale)
            self.view_offset_img = QPointF(new_offset_x, new_offset_y)
            self.update()

        def fit_frame_to_view(self):
            pixmap_ref = self.pixmap_for_compare_original if self._is_in_compare_mode and self.pixmap_for_compare_original else self.current_pixmap
            if pixmap_ref is None or pixmap_ref.isNull() or self.width() <= 0 or self.height() <= 0:
                self.zoom_scale = 1.0
                self.view_offset_img = QPointF(0.0, 0.0)
            else:
                widget_w, widget_h = self.width(), self.height()
                image_w, image_h = pixmap_ref.width(), pixmap_ref.height()
                if image_w <= 0 or image_h <= 0: return
                padding_factor = 0.98
                scale_w = (widget_w * padding_factor) / image_w
                scale_h = (widget_h * padding_factor) / image_h
                self.zoom_scale = min(scale_w, scale_h)
                self.zoom_scale = max(MIN_ZOOM, min(self.zoom_scale, MAX_ZOOM))
                scaled_img_w = image_w * self.zoom_scale
                scaled_img_h = image_h * self.zoom_scale
                view_top_left_x = (widget_w - scaled_img_w) / 2
                view_top_left_y = (widget_h - scaled_img_h) / 2
                self.view_offset_img = QPointF(-view_top_left_x / self.zoom_scale, -view_top_left_y / self.zoom_scale)

            self._is_user_zoomed = False
            self.update()

        def wheelEvent(self, event: QWheelEvent):
            if not getattr(self, 'allow_zoom_pan', False):
                event.ignore()
                return

            pixmap_ref = self.pixmap_for_compare_original if self._is_in_compare_mode and self.pixmap_for_compare_original else self.current_pixmap
            if pixmap_ref is None or pixmap_ref.isNull(): event.ignore(); return

            delta_steps = event.angleDelta().y() / 120.0
            if abs(delta_steps) < 0.1: event.ignore(); return

            widget_pos = event.position().toPoint()
            img_pos_f_before = self.map_widget_to_image_coords(widget_pos)
            if img_pos_f_before is None:
                widget_center = QPoint(self.width() // 2, self.height() // 2)
                img_pos_f_before = self.map_widget_to_image_coords(widget_center)
                if img_pos_f_before is None: event.ignore(); return
                widget_pos = widget_center

            zoom_multiplier = ZOOM_FACTOR ** delta_steps
            old_scale = self.zoom_scale
            new_scale = max(MIN_ZOOM, min(old_scale * zoom_multiplier, MAX_ZOOM))
            if abs(new_scale - old_scale) < 1e-6: event.ignore(); return

            self._is_user_zoomed = True
            self.zoom_scale = new_scale

            new_offset_x = img_pos_f_before.x() - (widget_pos.x() / self.zoom_scale)
            new_offset_y = img_pos_f_before.y() - (widget_pos.y() / self.zoom_scale)
            self.view_offset_img = QPointF(new_offset_x, new_offset_y)

            self.update()
            self.update_cursor()
            event.accept()

        def dragEnterEvent(self, event: QDragEnterEvent):
            mime = event.mimeData()
            if mime.hasUrls():
                if any(u.isLocalFile() and u.toLocalFile().lower().endswith(SUPPORTED_VIDEO_FORMATS) for u in
                       mime.urls()):
                    event.acceptProposedAction()
                    self._is_dragging_over = True
                    self.update()
                    return
            event.ignore()
            self._is_dragging_over = False
            self.update()

        def dragLeaveEvent(self, event: QEvent):
            self._is_dragging_over = False
            self.update()
            event.accept()

        def dropEvent(self, event: QDropEvent):
            self._is_dragging_over = False
            self.update()
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    fpath = url.toLocalFile()
                    if fpath.lower().endswith(SUPPORTED_VIDEO_FORMATS):
                        if hasattr(self.parent_window, '_load_video_for_segmentation'):
                            self.parent_window.switch_page(self.parent_window.VIDEO_SEG_PAGE_INDEX)
                            QApplication.processEvents()
                            self.parent_window._load_video_for_segmentation(fpath)
                        event.acceptProposedAction()
                        return
            event.ignore()

        def set_allow_interaction(self, allow: bool):
            if self.allow_interaction != allow: self.allow_interaction = allow; self.update_cursor()

        def clear_display(self):
            self.current_pixmap = None
            self.current_frame_index = -1
            self.allow_zoom_pan = False
            self.interaction_points_for_display = {}
            self.segmentation_masks_for_display = {}
            self.temp_annotation_frame_mask = None
            self.temp_annotation_target_id = -1
            self.temp_annotation_mask_frame_idx = -1
            self.view_offset_img = QPointF(0.0, 0.0)
            self._is_panning = False
            self._last_pan_pos_widget = QPoint()
            self._is_user_zoomed = False
            self._is_in_compare_mode = False
            self._split_ratio = 0.5
            self.pixmap_for_compare_original = None
            self.pixmap_for_compare_matted = None
            self.fit_frame_to_view()
            self.setText("拖放视频/GIF文件至此，\n或使用“加载”按钮，\n支持缩放/平移。")
            self.update_cursor()

        def map_widget_to_image_coords(self, widget_pos: QPoint) -> QPointF | None:
            if abs(self.zoom_scale) < 1e-6: return None
            return QPointF((widget_pos.x() / self.zoom_scale) + self.view_offset_img.x(),
                           (widget_pos.y() / self.zoom_scale) + self.view_offset_img.y())

        def map_image_to_widget_coords(self, image_pos_f: QPointF) -> QPoint | None:
            return QPoint(int(round((image_pos_f.x() - self.view_offset_img.x()) * self.zoom_scale)),
                          int(round((image_pos_f.y() - self.view_offset_img.y()) * self.zoom_scale)))

        def mousePressEvent(self, event: QMouseEvent):
            if self._is_in_compare_mode and event.button() == Qt.MouseButton.LeftButton:
                if self._splitter_line_rect.contains(event.position().toPoint()):
                    self._is_dragging_splitter = True
                    self.setCursor(self._size_hor_cursor)
                    event.accept()
                    return

            if event.button() == Qt.MouseButton.MiddleButton:
                if not getattr(self, 'allow_zoom_pan', False):
                    event.ignore()
                    return
                pixmap_ref = self.pixmap_for_compare_original if self._is_in_compare_mode else self.current_pixmap
                if pixmap_ref:
                    self._is_panning = True
                    self._last_pan_pos_widget = event.position().toPoint()
                    self.setCursor(self._pan_closed_cursor)
                    event.accept()
                    return

            is_selecting_mode = self._is_in_compare_mode or getattr(self.parent_window, 'current_target_id', -1) == -1

            if is_selecting_mode and event.button() == Qt.MouseButton.LeftButton and not self._is_panning:
                widget_pos = event.position().toPoint()
                image_pos_f = self.map_widget_to_image_coords(widget_pos)

                if image_pos_f:
                    y, x = int(image_pos_f.y()), int(image_pos_f.x())
                    clicked_obj = -1
                    for obj_id, mask in self.segmentation_masks_for_display.items():
                        if mask is not None and 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                            if mask[y, x]:
                                clicked_obj = obj_id
                                break

                    if hasattr(self.parent_window, '_select_video_object_by_id'):
                        if clicked_obj != -1 or self._is_in_compare_mode:
                            self.parent_window._select_video_object_by_id(clicked_obj)
                            event.accept()
                            return

            if not self._is_in_compare_mode and getattr(self, 'allow_interaction',
                                                        False) and self.current_pixmap and not self._is_panning:
                widget_pos = event.position().toPoint()
                image_pos_f = self.map_widget_to_image_coords(widget_pos)

                if image_pos_f:
                    img_w, img_h = self.current_pixmap.width(), self.current_pixmap.height()
                    if 0 <= image_pos_f.x() < img_w and 0 <= image_pos_f.y() < img_h:
                        mode = getattr(self, 'interaction_mode', 'sam')
                        is_paint_mode = mode in ['paint', 'brush', 'brush_add', 'brush_erase']

                        if is_paint_mode and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                            if hasattr(self.parent_window, '_save_video_state'): self.parent_window._save_video_state()

                            if mode == 'brush_erase':
                                is_add = False
                            elif mode == 'brush_add':
                                is_add = True
                            else:
                                is_add = (event.button() == Qt.MouseButton.LeftButton)

                            self._draw_on_mask(image_pos_f, is_press=True, is_adding=is_add)
                            event.accept()
                            return

                        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier and event.button() == Qt.MouseButton.LeftButton:
                            self._is_drawing_box = True
                            self._box_start_pos = widget_pos
                            self._current_box_rect = QRectF()
                            event.accept()
                            return

                        if not is_paint_mode:
                            button_type = 1 if event.button() == Qt.MouseButton.LeftButton else (
                                0 if event.button() == Qt.MouseButton.RightButton else -1)
                            if button_type != -1:
                                self.interaction_point_added.emit(image_pos_f, button_type)
                                try:
                                    self.video_points_changed.emit()
                                except AttributeError:
                                    pass
                                event.accept()
                                return

            super().mousePressEvent(event)

        def mouseMoveEvent(self, event: QMouseEvent):
            widget_pos = event.position().toPoint()
            self._current_mouse_pos_widget = widget_pos

            mode = getattr(self, 'interaction_mode', 'sam')
            is_paint_mode = mode in ['paint', 'brush', 'brush_add', 'brush_erase']

            if not self._is_in_compare_mode and is_paint_mode and (
                    event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)):
                image_pos_f = self.map_widget_to_image_coords(widget_pos)
                if image_pos_f and self.current_pixmap:
                    img_w, img_h = self.current_pixmap.width(), self.current_pixmap.height()
                    if 0 <= image_pos_f.x() < img_w and 0 <= image_pos_f.y() < img_h:
                        if mode == 'brush_erase':
                            is_add = False
                        elif mode == 'brush_add':
                            is_add = True
                        else:
                            is_add = bool(event.buttons() & Qt.MouseButton.LeftButton)
                        self._draw_on_mask(image_pos_f, is_press=False, is_adding=is_add)
                self.update()
                event.accept()
                return

            if is_paint_mode: self.update()

            if getattr(self, '_is_drawing_box', False):
                x = max(0, min(widget_pos.x(), self.width()))
                y = max(0, min(widget_pos.y(), self.height()))
                end_pos = QPoint(int(x), int(y))
                self._current_box_rect = QRectF(self._box_start_pos, end_pos).normalized()
                self.update()
                event.accept()
                return

            if getattr(self, '_is_dragging_splitter', False):
                pixmap_ref = self.pixmap_for_compare_original if self.pixmap_for_compare_original else self.current_pixmap
                if not pixmap_ref: return
                scaled_w = pixmap_ref.width() * self.zoom_scale
                dest_rect = QRectF(self.map_image_to_widget_coords(QPointF(0, 0)),
                                   QSizeF(scaled_w, pixmap_ref.height() * self.zoom_scale))
                if dest_rect.width() > 0:
                    new_ratio = (widget_pos.x() - dest_rect.left()) / dest_rect.width()
                    self._split_ratio = max(0.0, min(1.0, new_ratio))
                    self.update()
                    return

            if getattr(self, '_is_panning', False) and (event.buttons() & Qt.MouseButton.MiddleButton):
                delta = widget_pos - self._last_pan_pos_widget
                if not delta.isNull():
                    self.view_offset_img = self.view_offset_img - QPointF(delta.x() / self.zoom_scale,
                                                                          delta.y() / self.zoom_scale)
                    self._last_pan_pos_widget = widget_pos
                    self.update()
                return

            self.update_cursor()
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event: QMouseEvent):
            mode = getattr(self, 'interaction_mode', 'sam')
            is_paint_mode = mode in ['paint', 'brush', 'brush_add', 'brush_erase']

            if not self._is_in_compare_mode and is_paint_mode and event.button() in (
                    Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                self._last_paint_pos = None
                if hasattr(self.parent_window,
                           'update_video_preview_all_targets'): self.parent_window.update_video_preview_all_targets(
                    self.current_frame_index)
                event.accept()
                return

            if getattr(self, '_is_drawing_box', False) and event.button() == Qt.MouseButton.LeftButton:
                self._is_drawing_box = False
                if self._current_box_rect.width() > 10 and self._current_box_rect.height() > 10:
                    mapped_rect = self._map_rect_to_image(self._current_box_rect)
                    if mapped_rect: self.video_box_drawn.emit(mapped_rect)
                self._current_box_rect = QRectF()
                self.update()
                event.accept()
                return

            if getattr(self, '_is_dragging_splitter', False) and event.button() == Qt.MouseButton.LeftButton:
                self._is_dragging_splitter = False
                self.update_cursor()
                event.accept()
                return

            if getattr(self, '_is_panning', False) and event.button() == Qt.MouseButton.MiddleButton:
                self._is_panning = False
                self.update_cursor()
                event.accept()
                return

            super().mouseReleaseEvent(event)

        def paintEvent(self, event: QPaintEvent):
            painter = QPainter(self)
            widget_rect = self.rect()

            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

                pixmap_ref = self.pixmap_for_compare_original if self._is_in_compare_mode and self.pixmap_for_compare_original else self.current_pixmap

                if not pixmap_ref or pixmap_ref.isNull():
                    painter.setPen(QColor("#9CA3AF"))
                    painter.drawText(widget_rect, Qt.AlignmentFlag.AlignCenter, self.text())
                else:
                    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                    orig_w, orig_h = pixmap_ref.width(), pixmap_ref.height()

                    view_w_img = widget_rect.width() / self.zoom_scale
                    view_h_img = widget_rect.height() / self.zoom_scale
                    src_rect_f = QRectF(self.view_offset_img.x(), self.view_offset_img.y(), view_w_img, view_h_img)
                    src_rect_clipped_f = src_rect_f.intersected(QRectF(0, 0, orig_w, orig_h))
                    if not src_rect_clipped_f.isValid() or src_rect_clipped_f.isEmpty(): return

                    dest_rect_f = QRectF(
                        (src_rect_clipped_f.left() - self.view_offset_img.x()) * self.zoom_scale,
                        (src_rect_clipped_f.top() - self.view_offset_img.y()) * self.zoom_scale,
                        src_rect_clipped_f.width() * self.zoom_scale,
                        src_rect_clipped_f.height() * self.zoom_scale
                    )

                    # 1. 左右对比滑块视图渲染
                    if self._is_in_compare_mode:
                        if self.pixmap_for_compare_original and self.pixmap_for_compare_matted:
                            split_pos_x = dest_rect_f.left() + dest_rect_f.width() * self._split_ratio

                            painter.save()
                            painter.setClipRect(
                                QRectF(dest_rect_f.left(), dest_rect_f.top(), split_pos_x - dest_rect_f.left(),
                                       dest_rect_f.height()))
                            painter.drawPixmap(dest_rect_f, self.pixmap_for_compare_original,
                                               self.pixmap_for_compare_original.rect())
                            painter.restore()

                            painter.save()
                            painter.setClipRect(
                                QRectF(split_pos_x, dest_rect_f.top(), dest_rect_f.right() - split_pos_x + 1,
                                       dest_rect_f.height()))
                            painter.drawPixmap(dest_rect_f, self.pixmap_for_compare_matted,
                                               self.pixmap_for_compare_matted.rect())
                            painter.restore()

                            painter.setPen(QPen(QColor(255, 255, 255, 200), 2.5))
                            painter.drawLine(int(split_pos_x), int(dest_rect_f.top()), int(split_pos_x),
                                             int(dest_rect_f.bottom()))

                            triangle_size = 10
                            triangle_y_center = dest_rect_f.center().y()
                            triangle_path = QPainterPath()
                            triangle_path.moveTo(split_pos_x - triangle_size, triangle_y_center)
                            triangle_path.lineTo(split_pos_x, triangle_y_center - triangle_size)
                            triangle_path.lineTo(split_pos_x + triangle_size, triangle_y_center)
                            triangle_path.closeSubpath()

                            self._splitter_line_rect = triangle_path.boundingRect().toRect().adjusted(-5, -5, 5, 5)
                            painter.setPen(QPen(QColor(0, 0, 0, 50), 1))
                            painter.setBrush(QColor(255, 255, 255, 220))
                            painter.drawPath(triangle_path)

                    # 2. 单图视图渲染：只做最纯净的一部 Pixmap 直接绘制（彩色覆膜和发丝边缘已完美离屏合成完毕，此步完全零开销）
                    else:
                        dest_rect_int = dest_rect_f.toRect()
                        src_rect_clipped_int = src_rect_clipped_f.toRect()

                        painter.drawPixmap(dest_rect_int, self.current_pixmap, src_rect_clipped_int)

                        # 在最外层高速绘制交互标注框 (Boxes) & 坐标点 (Keypoints)
                        active_id = getattr(self, 'active_object_id', -1)
                        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                        for obj_id, box_coords in self.interaction_boxes_for_display.items():
                            if active_id == -1 or active_id == obj_id:
                                color_idx = (obj_id) % len(VIDEO_TARGET_COLORS)
                                obj_color = VIDEO_TARGET_COLORS[color_idx]
                                x1, y1, x2, y2 = box_coords
                                tl_widget = self.map_image_to_widget_coords(QPointF(x1, y1))
                                br_widget = self.map_image_to_widget_coords(QPointF(x2, y2))
                                if tl_widget and br_widget:
                                    box_rect = QRect(tl_widget, br_widget)
                                    painter.setPen(
                                        QPen(obj_color, max(1.5, 2.0 * self.zoom_scale), Qt.PenStyle.SolidLine))
                                    painter.setBrush(Qt.BrushStyle.NoBrush)
                                    painter.drawRect(box_rect)

                        point_radius_widget = max(2.0, min(8.0, VIDEO_POINT_RADIUS_IMG * self.zoom_scale))
                        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                        for obj_id, points_for_obj_with_labels in self.interaction_points_for_display.items():
                            if active_id == -1 or active_id == obj_id:
                                color_idx = (obj_id) % len(VIDEO_TARGET_COLORS)
                                obj_color = VIDEO_TARGET_COLORS[color_idx]
                                for px, py, label in points_for_obj_with_labels:
                                    widget_coord = self.map_image_to_widget_coords(QPointF(px, py))
                                    if widget_coord and dest_rect_int.contains(widget_coord):
                                        point_fill_color = QColor(34, 177, 76, 240) if label == 1 else QColor(237, 28,
                                                                                                              36,
                                                                                                              240)
                                        painter.setPen(QPen(obj_color, max(1.0, 1.5 * self.zoom_scale)))
                                        painter.setBrush(point_fill_color)
                                        painter.drawEllipse(QPointF(widget_coord), point_radius_widget,
                                                            point_radius_widget)

                    if getattr(self, '_is_dragging_over', False):
                        painter.setBrush(QBrush(QColor(79, 70, 229, 40)))
                        drag_pen = QPen(QColor(79, 70, 229, 200), 3)
                        drag_pen.setStyle(Qt.PenStyle.DashLine)
                        painter.setPen(drag_pen)
                        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 10, 10)

                    if not self._is_in_compare_mode and getattr(self, '_is_drawing_box', False) and not getattr(self,
                                                                                                                '_current_box_rect',
                                                                                                                QRectF()).isEmpty():
                        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                        pen = QPen(QColor(0, 150, 255), 2, Qt.PenStyle.DashLine)
                        painter.setPen(pen)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawRect(self._current_box_rect)

                    mode = getattr(self, 'interaction_mode', 'sam')
                    is_paint_mode = mode in ['paint', 'brush', 'brush_add', 'brush_erase']
                    if not self._is_in_compare_mode and is_paint_mode and getattr(self, '_current_mouse_pos_widget',
                                                                                  None):
                        widget_pos = self._current_mouse_pos_widget
                        img_pos = self.map_widget_to_image_coords(widget_pos)
                        if img_pos and self.current_pixmap:
                            if 0 <= img_pos.x() < orig_w and 0 <= img_pos.y() < orig_h:
                                brush_sz = getattr(self, 'brush_size', 15)
                                if hasattr(self, 'vid_brush_slider'):
                                    brush_sz = self.vid_brush_slider.value()
                                elif hasattr(self, 'brush_slider'):
                                    brush_sz = self.brush_slider.value()

                                cursor_radius_widget = brush_sz * self.zoom_scale
                                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

                                if mode == 'brush_add':
                                    painter.setPen(QPen(QColor(34, 177, 76, 200), 2))
                                    painter.setBrush(QColor(34, 177, 76, 60))
                                elif mode == 'brush_erase':
                                    painter.setPen(QPen(QColor(237, 28, 36, 200), 2))
                                    painter.setBrush(QColor(237, 28, 36, 60))
                                else:
                                    painter.setPen(QPen(QColor(150, 150, 150, 200), 2))
                                    painter.setBrush(QColor(150, 150, 150, 40))

                                painter.drawEllipse(widget_pos, cursor_radius_widget, cursor_radius_widget)

            except Exception as e:
                print(f"视频 paintEvent 错误: {e}")
                traceback.print_exc()
            finally:
                if painter.isActive(): painter.end()

        def update_cursor(self):
            if self._is_panning:
                self.setCursor(self._pan_closed_cursor)
                return
            if self._is_dragging_splitter:
                self.setCursor(self._size_hor_cursor)
                return

            widget_pos = self.mapFromGlobal(QCursor.pos())
            if self._is_in_compare_mode:
                if self._splitter_line_rect.contains(widget_pos):
                    self.setCursor(self._size_hor_cursor)
                elif self.pixmap_for_compare_original:
                    self.setCursor(self._pan_open_cursor)
                else:
                    self.setCursor(self._default_cursor)
                return

            can_interact_now = self.allow_interaction and self.current_pixmap is not None
            cursor_over_image = False
            if self.current_pixmap:
                image_pos_f = self.map_widget_to_image_coords(widget_pos)
                if image_pos_f:
                    img_w, img_h = self.current_pixmap.width(), self.current_pixmap.height()
                    cursor_over_image = 0 <= image_pos_f.x() < img_w and 0 <= image_pos_f.y() < img_h

            mode = getattr(self, 'interaction_mode', 'sam')
            is_paint_mode = mode in ['paint', 'brush', 'brush_add', 'brush_erase']
            if can_interact_now and cursor_over_image and is_paint_mode:
                self.setCursor(Qt.CursorShape.BlankCursor)
                return

            new_cursor = self._default_cursor
            if can_interact_now and cursor_over_image:
                new_cursor = self._cross_cursor

            self.setCursor(new_cursor)

        def enterEvent(self, event: QEvent):
            self.update_cursor()
            super().enterEvent(event)

        def leaveEvent(self, event: QEvent):
            self.setCursor(self._default_cursor)
            super().leaveEvent(event)

class VideoViewMixin:
    """Modern professional video editor layout"""

    def setup_video_segment_page(self):
        self.video_segment_page = QWidget()
        self.video_segment_page.setObjectName("VideoEditorRoot")
        self.video_segment_page.setStyleSheet(f"QWidget#VideoEditorRoot {{ background-color: {VID_DARK_BG}; }}")

        root_layout = QVBoxLayout(self.video_segment_page)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Top global toolbar
        self.top_bar_main = QWidget()
        self.top_bar_main.setObjectName("VidTopBar")
        self.top_bar_main.setFixedHeight(54)
        self.top_bar_main.setStyleSheet(
            f"QWidget#VidTopBar {{ background-color: {VID_DARK_BG}; border: none; }}")
        top_layout = QHBoxLayout(self.top_bar_main)
        top_layout.setContentsMargins(16, 0, 16, 0)

        self.home_button_video_page = QToolButton()
        self.home_button_video_page.setText("主页")
        if hasattr(self, '_create_svg_icon'):
            self.home_button_video_page.setIcon(
                self._create_svg_icon("house.svg", size=20, color=QColor(VID_TEXT_PRIMARY)))
        self.home_button_video_page.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.home_button_video_page.setStyleSheet(
            f"QToolButton {{ color: {VID_TEXT_PRIMARY}; font-size: 15px; font-weight: bold; border:none; background: transparent; padding: 4px 8px; border-radius: 4px; }}"
            f"QToolButton:hover {{ background-color: #333333; }}"
        )
        self.home_button_video_page.clicked.connect(lambda: self.switch_page_with_slide(self.WELCOME_PAGE_INDEX))

        top_layout.addWidget(self.home_button_video_page)

        # [Core Modification]: Add a large stretch here to push all subsequent buttons to the far right
        top_layout.addStretch(1)

        self.vid_undo_button = QToolButton()
        if hasattr(self, '_create_svg_icon'): self.vid_undo_button.setIcon(
            self._create_svg_icon("arrow-90deg-left.svg", color=VID_TEXT_PRIMARY))
        self.vid_undo_button.setStyleSheet(VID_MODERN_BTN_STYLE)
        if hasattr(self, 'undo_video_action'): self.vid_undo_button.clicked.connect(self.undo_video_action)

        self.vid_redo_button = QToolButton()
        if hasattr(self, '_create_svg_icon'): self.vid_redo_button.setIcon(
            self._create_svg_icon("arrow-90deg-right.svg", color=VID_TEXT_PRIMARY))
        self.vid_redo_button.setStyleSheet(VID_MODERN_BTN_STYLE)
        if hasattr(self, 'redo_video_action'): self.vid_redo_button.clicked.connect(self.redo_video_action)

        # Add undo and redo buttons to the right
        top_layout.addWidget(self.vid_undo_button)
        top_layout.addWidget(self.vid_redo_button)

        # Add a small spacing (10px) to slightly separate the arrow buttons and the export button
        top_layout.addSpacing(10)

        self.save_video_seg_button = QPushButton("导出")
        self.save_video_seg_button.setObjectName("PrimaryActionBtn")
        self.save_video_seg_button.setStyleSheet(VID_PRIMARY_ACTION_BTN_STYLE)
        if hasattr(self, 'save_video_segmentation_result'): self.save_video_seg_button.clicked.connect(
            self.save_video_segmentation_result)

        top_layout.addWidget(self.save_video_seg_button)

        root_layout.addWidget(self.top_bar_main)

        # 2. StackedWidget workspace
        self.vid_editor_stack = QStackedWidget()
        self.vid_editor_stack.setObjectName("VidEditorStack")
        self.vid_editor_stack.setStyleSheet(f"QStackedWidget#VidEditorStack {{ background-color: {VID_DARK_BG}; }}")
        root_layout.addWidget(self.vid_editor_stack, 1)

        self.video_display_label = VideoDisplayLabel(self)
        self.video_display_label.setObjectName("VidCoreDisplay")
        self.video_display_label.setStyleSheet(
            f"QLabel#VidCoreDisplay {{ background-color: #000000; color: {VID_TEXT_SECONDARY}; border: none; border-radius: 8px; }}")
        if hasattr(self, 'handle_video_interaction_point'): self.video_display_label.interaction_point_added.connect(
            self.handle_video_interaction_point)
        if hasattr(self, 'update_button_states'): self.video_display_label.video_points_changed.connect(
            self.update_button_states)
        if hasattr(self, 'handle_video_interaction_box'): self.video_display_label.video_box_drawn.connect(
            self.handle_video_interaction_box)

        self.video_thumbnail_scrubber = VideoSimpleTimeline()
        self.video_thumbnail_scrubber.frame_selected.connect(self.seek_video_via_scrubber)

        self.ratio_container = AspectRatioContainer(self.video_display_label)

        self._init_vid_main_editor_page()
        self._init_vid_dedicated_matting_page()
        self._init_vid_dedicated_crop_page()

        self.vid_editor_stack.addWidget(self.vid_main_editor_page)
        self.vid_editor_stack.addWidget(self.vid_dedicated_matting_page)
        self.vid_editor_stack.addWidget(self.vid_dedicated_crop_page)

        self._mount_player_to_workspace(self.main_workspace_wrapper)
        if hasattr(self, 'stacked_widget'): self.stacked_widget.addWidget(self.video_segment_page)

        self.virtual_timeline = []
        self._cv_caps = {}

    def _mount_player_to_workspace(self, workspace_widget):
        if workspace_widget.layout() is not None:
            while workspace_widget.layout().count():
                item = workspace_widget.layout().takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
            layout = workspace_widget.layout()
        else:
            layout = QVBoxLayout(workspace_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(16)

        if not hasattr(self, 'video_margin_wrapper'):
            self.video_margin_wrapper = QWidget()
            vl = QVBoxLayout(self.video_margin_wrapper)
            vl.setContentsMargins(0, 0, 0, 0)
            self.ratio_container.setParent(self.video_margin_wrapper)
            vl.addWidget(self.ratio_container)

        if workspace_widget == getattr(self, 'matting_workspace_wrapper', None):
            self.video_margin_wrapper.layout().setContentsMargins(60, 40, 60, 0)
        else:
            self.video_margin_wrapper.layout().setContentsMargins(0, 0, 0, 0)

        self.video_margin_wrapper.setParent(workspace_widget)
        self.video_margin_wrapper.show()
        layout.addWidget(self.video_margin_wrapper, 1)

        if not hasattr(self, 'playback_bar_wrapper'):
            self.playback_bar_wrapper = QWidget()
            self.playback_bar_wrapper.setFixedHeight(62)

            pb_wrapper_layout = QHBoxLayout(self.playback_bar_wrapper)
            pb_wrapper_layout.setContentsMargins(16, 0, 16, 16)

            self.playback_bar = QWidget()
            self.playback_bar.setObjectName("VidPlaybackBar")
            self.playback_bar.setFixedHeight(46)
            self.playback_bar.setStyleSheet("QWidget#VidPlaybackBar { background-color: #262626; border-radius: 6px; }")

            pb_layout = QHBoxLayout(self.playback_bar)
            pb_layout.setContentsMargins(16, 0, 16, 0)
            pb_layout.setSpacing(12)

            self.btn_prev = QToolButton()
            if hasattr(self, '_create_svg_icon'):
                self.btn_prev.setIcon(self._create_svg_icon("caret-left-fill.svg", color=VID_TEXT_PRIMARY))
            self.btn_prev.setStyleSheet(VID_MODERN_BTN_STYLE)
            # [Added]: Bind previous frame function
            self.btn_prev.clicked.connect(lambda: self.step_video_frame(-1))

            self.play_pause_button = QToolButton()
            if hasattr(self, '_create_svg_icon'):
                self.play_pause_button.setIcon(self._create_svg_icon("play-fill.svg", color=VID_TEXT_PRIMARY))
            self.play_pause_button.setStyleSheet(VID_MODERN_BTN_STYLE)
            self.play_pause_button.setCheckable(True)
            if hasattr(self, 'toggle_play_pause'):
                self.play_pause_button.toggled.connect(self.toggle_play_pause)

            self.btn_next = QToolButton()
            if hasattr(self, '_create_svg_icon'):
                self.btn_next.setIcon(self._create_svg_icon("caret-right-fill.svg", color=VID_TEXT_PRIMARY))
            self.btn_next.setStyleSheet(VID_MODERN_BTN_STYLE)
            # [Added]: Bind next frame function
            self.btn_next.clicked.connect(lambda: self.step_video_frame(1))

            # ==========================================================
            # [Added]: Global mute toggle button (Main Player)
            # ==========================================================
            self.btn_global_mute = QToolButton()
            if hasattr(self, '_create_svg_icon'):
                icon_name = "volume-mute-fill.svg" if getattr(self, '_is_global_muted', False) else "volume-up-fill.svg"
                self.btn_global_mute.setIcon(self._create_svg_icon(icon_name, color=VID_TEXT_PRIMARY))
            self.btn_global_mute.setStyleSheet(VID_MODERN_BTN_STYLE)
            self.btn_global_mute.clicked.connect(self._toggle_global_mute)

            self.time_label_curr = QLabel("0:00.00")

            self.time_label_curr.setStyleSheet(f"color: {VID_TEXT_PRIMARY}; font-size: 12px; background:transparent;")

            self.time_label_total = QLabel("0:00.00")
            self.time_label_total.setStyleSheet(f"color: {VID_TEXT_PRIMARY}; font-size: 12px; background:transparent;")

            # -----------------------------------------------------
            # [Removed] Full screen button has been completely removed
            # -----------------------------------------------------

            pb_layout.addWidget(self.btn_prev)
            pb_layout.addWidget(self.play_pause_button)
            pb_layout.addWidget(self.btn_next)
            pb_layout.addWidget(self.btn_global_mute)
            pb_layout.addSpacing(8)
            pb_layout.addWidget(self.time_label_curr)

            self.video_thumbnail_scrubber.setStyleSheet(VID_TIMELINE_SLIDER_STYLE)
            pb_layout.addWidget(self.video_thumbnail_scrubber, 1)

            pb_layout.addWidget(self.time_label_total)
            # Full screen button removed, only total duration will be displayed on the right of the slider

            pb_wrapper_layout.addWidget(self.playback_bar)

        self.playback_bar_wrapper.setParent(workspace_widget)
        self.playback_bar_wrapper.show()
        layout.addWidget(self.playback_bar_wrapper, 0)
        self.playback_bar.setMaximumWidth(16777215)
        self.playback_bar_wrapper.layout().invalidate()

    def _update_storyboard_highlight(self):
        if not hasattr(self, 'storyboard_list') or self.storyboard_list.count() == 0:
            return

        clip_idx, _ = self._get_current_clip_info()
        if clip_idx is not None and 0 <= clip_idx < self.storyboard_list.count():
            for i in range(self.storyboard_list.count()):
                item = self.storyboard_list.item(i)
                widget = self.storyboard_list.itemWidget(item)
                if isinstance(widget, StoryboardItemWidget):
                    widget.set_active(i == clip_idx)

            current_ui_row = self.storyboard_list.currentRow()
            if current_ui_row != clip_idx:
                self.storyboard_list.blockSignals(True)
                self.storyboard_list.setCurrentRow(clip_idx)
                item = self.storyboard_list.item(clip_idx)
                if item:
                    self.storyboard_list.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                self.storyboard_list.blockSignals(False)

    @Slot()
    def select_multiple_videos_for_library(self):
        supported = ' '.join(['*' + ext for ext in SUPPORTED_VIDEO_FORMATS])
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多个视频或GIF", "",
            f"视频/GIF 文件 ({supported});;所有文件 (*)"
        )
        if not file_paths: return

        for path in file_paths:
            self._add_video_to_library_ui(path)

        self._update_asset_library_ui()

    def _add_video_to_library_ui(self, file_path):
        for i in range(self.vid_asset_list.count()):
            if self.vid_asset_list.item(i).data(Qt.ItemDataRole.UserRole) == file_path:
                return

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return
        ret, frame = cap.read()
        cap.release()

        pixmap = convert_cv_to_pixmap(frame) if ret else QPixmap()
        item = QListWidgetItem(self.vid_asset_list)
        cell_widget = LibraryItemWidget(file_path, pixmap)
        item.setSizeHint(QSize(170, 104))
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        self.vid_asset_list.setItemWidget(item, cell_widget)

    def _update_asset_library_ui(self):
        if getattr(self, 'vid_asset_list', None) and self.vid_asset_list.count() > 0:
            self.vid_asset_stack.setCurrentWidget(self.vid_asset_list)
        else:
            self.vid_asset_stack.setCurrentWidget(self.vid_empty_state_widget)

    def _get_timeline_signature(self) -> str:
        """
        Calculate the physical structure MD5 signature of the current merged timeline track.
        Used to determine whether the video segment order, physical crop points, and segment paths have actually changed.
        """
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline:
            return ""

        sig_parts = []
        for clip in self.virtual_timeline:
            # Physical signature is composed of: path + trim in point + trim out point, aligned to global absolute order
            sig_parts.append(f"{clip['path']}_{clip['in_point']}_{clip['out_point']}")

        return hashlib.md5("_".join(sig_parts).encode('utf-8')).hexdigest()

    def _trigger_global_timeline_rebuild(self):
        """
        [Timeline Reconstruction Hub]: Adopts intelligent feature signature verification to determine in seconds whether re-extraction and re-encoding are required.
        Fixed: One-click bypass of reconstruction mountain when undoing points or re-entering the same source, microsecond-level lightning response, eliminating lag and secondary computation.
        """
        if not getattr(self, 'virtual_timeline', []):
            self._invalidate_timeline_cache()
            if hasattr(self, 'video_display_label'):
                self.video_display_label.clear_display()
            return

        # 1. Physical feature fingerprint signature calculation
        current_sig = self._get_timeline_signature()

        # =========================================================================
        # [Core Performance Fix]: Fingerprint matching determination. If the track physical properties are unchanged, hard disk images and AI features are fully valid, one-click second-level pass!
        # =========================================================================
        if (getattr(self, '_timeline_signature', None) == current_sig and
                getattr(self, 'temp_frame_dir', None) is not None and
                os.path.exists(self.temp_frame_dir)):

            self.is_extracting_frames = False

            # Safely hide full-screen advanced loading overlay
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()

            # Determine if there are queued safe page jumps waiting
            pending_mode = getattr(self, '_pending_mode_switch', None)
            if pending_mode == "matting":
                self._pending_mode_switch = None
                self._enter_dedicated_matting_mode()
            elif pending_mode == "crop":
                self._pending_mode_switch = None
                self._enter_dedicated_crop_mode()
            else:
                self._display_frame_wrapper(self.current_frame_index)
            return

        # =========================================================================
        # The following will only run when [Timeline undergoes physical substantial changes] (e.g., adding/deleting segments, rearranging order, trimming changes)
        # =========================================================================
        self._timeline_signature = current_sig

        if getattr(self, '_timeline_extractor', None) is not None:
            try:
                self._timeline_extractor.requestInterruption()
                self._timeline_extractor.wait(150)
                self._timeline_extractor.deleteLater()
            except Exception:
                pass
            self._timeline_extractor = None

        if hasattr(self, '_video_frame_cache'):
            self._video_frame_cache.clear()

        # Only when the timeline is truly rearranged or reorganized is it necessary to clear the original SAM2 feature encoding state
        if getattr(self, 'video_predictor', None) and getattr(self, 'video_inference_state', None):
            try:
                self.video_predictor.reset_state(self.video_inference_state)
            except Exception:
                pass
        self.video_inference_state = None

        base_name = "merged_timeline"
        session_id = str(uuid.uuid4())[:8]
        self.temp_frame_dir = os.path.join(TEMP_BASE_DIR, f"frames_{base_name}_{session_id}")
        os.makedirs(self.temp_frame_dir, exist_ok=True)

        self.is_extracting_frames = True

        # Pull up full-screen modern overlay
        self.show_global_loading_overlay("正在同步合并视频序列...", 0)

        target_w = getattr(self, 'video_width', 1280)
        target_h = getattr(self, 'video_height', 720)
        if target_w <= 0 or target_h <= 0:
            target_w, target_h = 1280, 720

        self._timeline_extractor = GlobalTimelineExtractor(
            self.virtual_timeline, self.temp_frame_dir, target_w, target_h
        )

        self._timeline_extractor.progress.connect(self._on_worker_progress)
        self._timeline_extractor.finished.connect(self._on_global_extraction_finished)

        self._timeline_extractor.start()

    @Slot()
    def _on_audio_mute_toggled(self):
        """【MPV版】剪裁界面：切换片段静音状态（原声+旧BGM）"""
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'):
            return

        clip = self.virtual_timeline[idx]
        is_muted = self.btn_toggle_mute.isChecked()

        # 核心逻辑：剪辑界面的片段静音是静音原声音，若有旧BGM也一并静音
        clip['mute_original'] = is_muted
        clip['mute_bgm'] = is_muted

        # 如果局部恢复声音了，全局的静音锁也要一并解开，防止死锁
        if not is_muted:
            clip['mute_all'] = False

        new_text_cn = "恢复声音" if is_muted else "片段静音"
        self.btn_toggle_mute.setProperty("orig_text", new_text_cn)
        if hasattr(self, '_TR'):
            self.btn_toggle_mute.setText(self._TR(new_text_cn))
        else:
            self.btn_toggle_mute.setText(new_text_cn)

        self.btn_toggle_mute.setStyleSheet("""
                    QPushButton { background-color: #2C2C2E; color: #FFFFFF; border-radius: 6px; padding: 10px 12px; font-weight: bold; font-size: 13px; border: 1px solid #3A3A3C; } 
                    QPushButton:hover { background-color: #3A3A3C; border: 1px solid #555555; }
                    QPushButton:pressed { background-color: #1C1C1E; }
                """)

        self.orig_volume_slider.setEnabled(not is_muted)
        if hasattr(self, 'bgm_volume_slider'):
            self.bgm_volume_slider.setEnabled(not is_muted)

        if self._current_playing_clip_idx == idx:
            self._sync_audio_engine_to_current_frame(idx)

        if hasattr(self, 'storyboard_list'):
            item_widget = self.storyboard_list.itemWidget(self.storyboard_list.item(idx))
            if hasattr(item_widget, 'sync_ui_with_data'):
                # 只有原声和BGM都被静音，或全部静音时，才将下方故事板图标标红
                all_muted = clip.get('mute_all', False) or (
                            clip.get('mute_original', False) and clip.get('mute_bgm', False))
                item_widget.sync_ui_with_data(all_muted)

    @Slot(int)
    def _on_orig_volume_changed(self, value):
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'): return

        vol_float = value / 100.0
        self.virtual_timeline[idx]['original_audio_volume'] = vol_float

        if self._current_playing_clip_idx == idx and hasattr(self, 'mpv_audio'):
            if not getattr(self, '_is_global_muted', False) and not self.virtual_timeline[idx].get('mute_original',
                                                                                                   False):
                self.mpv_audio.volume = value

    def _safely_reset_media_player(self, player):
        """【MPV版】安全停止并释放引擎句柄"""
        if player is not None:
            try:
                # MPV 底层切断播放的命令
                player.command("stop")
            except Exception:
                pass

    def _get_audio_file_duration_pure(self, file_path: str) -> float:
        """
        高精度提取音频物理时长 (采用 Pydub 方案)。
        Pydub 的 len() 方法会返回真实的物理毫秒数，100% 精确，
        彻底杜绝由于 Header 信息损坏造成的“长几秒”或“短几十秒”问题。
        """
        if not file_path or not os.path.exists(file_path):
            return 0.0

        # 主流精准方案：直接使用 Pydub 提取真实时长
        try:
            from pydub import AudioSegment
            # AudioSegment.from_file 会将音频完整解析，len() 获取精确的毫秒数
            audio = AudioSegment.from_file(file_path)
            duration_sec = len(audio) / 1000.0

            if duration_sec > 0.1:
                return duration_sec

        except ImportError:
            print("[DEBUG] Pydub 未安装，将平滑降级到 PyAV 读取。")
        except Exception as e:
            print(f"[DEBUG] Pydub 解析音频失败: {e}，将平滑降级到 PyAV 读取。")

        # 黄金降级兜底方案：使用项目内置的 PyAV 进行物理推算
        try:
            import av
            with av.open(file_path) as container:
                if not container.streams.audio:
                    return 0.0

                audio_stream = container.streams.audio[0]

                # 兜底 A：物理推算：利用比特率 (Bitrate) 和 文件大小 (File Size) 估算
                if audio_stream.bit_rate is not None and audio_stream.bit_rate > 0:
                    file_size = os.path.getsize(file_path)
                    dur = file_size / (audio_stream.bit_rate / 8.0)
                    if dur > 0.1:
                        return dur

                # 兜底 B：使用 PyAV 音频流专属元数据
                if audio_stream.duration is not None and audio_stream.time_base is not None:
                    dur = float(audio_stream.duration * audio_stream.time_base)
                    if dur > 0.1:
                        return dur

        except Exception as e:
            print(f"[DEBUG] PyAV 备用方案解析失败: {e}")

        # 极限情况下的安全默认值
        return 120.0

    @Slot()
    def _on_add_bgm_clicked(self):
        """【MPV版】导入背景音乐，加入新BGM后必须确保其能发声"""
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'):
            return

        file_paths, _ = QFileDialog.getOpenFileNames(
            self, _TR("选择背景音乐"), "", "音频文件 (*.mp3 *.wav *.aac *.m4a)"
        )
        if not file_paths or len(file_paths) == 0:
            return

        selected_bgm_path = file_paths[0]

        self._stop_bgm_preview_monitoring()
        self._is_playing_bgm_preview = False
        self._safely_reset_media_player(getattr(self, 'mpv_bgm', None))

        bgm_duration = self._get_audio_file_duration_pure(selected_bgm_path)
        self._save_video_state()

        clip = self.virtual_timeline[idx]
        clip['custom_audio_path'] = selected_bgm_path

        # 【核心逻辑】：新加入BGM，必须清除它的静音状态，同时解开全局静音锁
        clip['mute_bgm'] = False
        clip['mute_all'] = False

        vol = 1.0
        if hasattr(self, 'bgm_volume_slider'):
            vol = self.bgm_volume_slider.value() / 100.0

        clip['custom_audio_volume'] = vol
        clip['custom_audio_clip_start'] = 0.0
        clip['custom_audio_clip_end'] = bgm_duration
        clip['_cached_bgm_duration'] = bgm_duration
        clip['custom_audio_start_sec'] = 0.0

        self._current_playing_clip_idx = -1

        # 新加 BGM 必然会解开全局静音，同步更新下方图标
        if hasattr(self, 'storyboard_list'):
            item_widget = self.storyboard_list.itemWidget(self.storyboard_list.item(idx))
            if hasattr(item_widget, 'sync_ui_with_data'):
                all_muted = clip.get('mute_all', False) or (
                            clip.get('mute_original', False) and clip.get('mute_bgm', False))
                item_widget.sync_ui_with_data(all_muted)

        self._update_audio_console_ui()
        self._sync_audio_engine_to_current_frame()

    @Slot(int, int)
    def _on_bgm_trim_range_changed(self, start, end):
        """【MPV版】BGM 区间修剪滑块响应"""
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'): return

        clip = self.virtual_timeline[idx]
        clip['custom_audio_clip_start'] = float(start)
        clip['custom_audio_clip_end'] = float(end)

        if hasattr(self, 'bgm_clip_time_label'):
            self.bgm_clip_time_label.setText(f"{self._format_time_simple(start)} - {self._format_time_simple(end)}")

        # 【核心修改】：拖动时让 MPV 寻道对齐
        if hasattr(self, 'mpv_bgm') and getattr(self.mpv_bgm, 'path', None):
            try:
                self.mpv_bgm.seek(float(start), reference='absolute', precision='exact')
            except Exception:
                pass

    def _format_time_simple(self, seconds: float) -> str:
        """
        [New auxiliary function] Used to fine-tune the time format display of the BGM double slider on the panel.
        """
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    @Slot()
    def _on_remove_bgm_clicked(self):
        """【MPV版】安全地移除背景音乐并立即释放句柄"""
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'):
            return

        # 1. 物理切断播放状态，操作前全部暂停
        was_playing = getattr(self, 'is_playing', False)
        if was_playing:
            self.pause_video()

        # 强制暂停 MPV 音乐播放器
        if hasattr(self, 'mpv_bgm'):
            try:
                self.mpv_bgm.pause = True
            except Exception:
                pass

        self._save_video_state()

        # 2. 物理切断并关闭背景预览监听
        self._stop_bgm_preview_monitoring()
        self._is_playing_bgm_preview = False

        # 3. 必须先清除底层数据结构，使残留事件无法重新唤醒播放
        if idx < len(self.virtual_timeline):
            clip = self.virtual_timeline[idx]
            clip['custom_audio_path'] = None
            clip['custom_audio_clip_start'] = 0.0
            clip['custom_audio_clip_end'] = 0.0
            clip['custom_audio_start_sec'] = 0.0
            if '_cached_bgm_duration' in clip:
                clip['_cached_bgm_duration'] = 0.0

        self._current_playing_clip_idx = -1

        # 4. 停止播放器并清空源（调用上一步我们改好的 mpv 安全清理函数）
        self._safely_reset_media_player(getattr(self, 'mpv_bgm', None))

        # 5. 同步 UI 状态与音频引擎
        self._update_audio_console_ui()
        self._sync_audio_engine_to_current_frame()

        if was_playing:
            self.play_video()

    @Slot(int)
    def _on_bgm_volume_changed(self, value):
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'): return

        vol_float = value / 100.0
        self.virtual_timeline[idx]['custom_audio_volume'] = vol_float

        if self._current_playing_clip_idx == idx and hasattr(self, 'mpv_bgm'):
            if not getattr(self, '_is_global_muted', False):
                self.mpv_bgm.volume = value

    @Slot(float)
    def _on_bgm_start_changed(self, value):
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'): return

        self.virtual_timeline[idx]['custom_audio_start_sec'] = value

        # Immediately sync and align playback engine time in real-time
        if self._current_playing_clip_idx == idx:
            self._sync_audio_engine_to_current_frame(idx)
            if getattr(self, 'is_playing', False):
                self.bgm_player.play()

    @Slot()
    def _on_bgm_preview_play_clicked(self):
        """【MPV版】BGM独立预览播放"""
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'):
            return

        if getattr(self, 'is_playing', False):
            self.pause_video()

        # 停掉主音轨
        if hasattr(self, 'mpv_audio'):
            try:
                self.mpv_audio.command("stop")
            except Exception:
                pass

        is_playing_preview = getattr(self, '_is_playing_bgm_preview', False)

        if is_playing_preview:
            if hasattr(self, 'mpv_bgm'):
                self.mpv_bgm.pause = True
            self._stop_bgm_preview_monitoring()
        else:
            clip = self.virtual_timeline[idx]
            bgm_path = clip.get('custom_audio_path')
            if not bgm_path:
                return

            bgm_clip_start_sec = clip.get('custom_audio_clip_start', 0.0)

            # 【核心修改】：使用 MPV 播放和定位 BGM
            self.mpv_bgm['start'] = str(bgm_clip_start_sec)
            self.mpv_bgm.play(bgm_path)
            self.mpv_bgm.volume = int(clip.get('custom_audio_volume', 1.0) * 100)
            self.mpv_bgm.pause = False

            self._start_bgm_preview_monitoring()

        self._update_bgm_preview_button_ui()
        self._update_bgm_remove_button_state()

    def _start_bgm_preview_monitoring(self):
        """
        开启 BGM 独立预览物理边界监听。
        """
        self._is_playing_bgm_preview = True
        if not hasattr(self, '_bgm_preview_timer') or self._bgm_preview_timer is None:
            self._bgm_preview_timer = QTimer(self)
            self._bgm_preview_timer.setInterval(30)

        # 连接前断开旧连接，防止多重绑定
        try:
            self._bgm_preview_timer.timeout.disconnect()
        except Exception:
            pass

        self._bgm_preview_timer.timeout.connect(self._monitor_bgm_preview_playback)
        self._bgm_preview_timer.start()

    def _stop_bgm_preview_monitoring(self):
        """
        停止 BGM 独立预览，并物理切断所有信号槽连接。
        """
        self._is_playing_bgm_preview = False
        if hasattr(self, '_bgm_preview_timer') and self._bgm_preview_timer:
            self._bgm_preview_timer.stop()
            # 【核心修复】：物理断开信号槽连接，确保留在 Qt 事件队列中的微秒级残留超时信号不会触发回调。
            try:
                self._bgm_preview_timer.timeout.disconnect()
            except Exception:
                pass
        self._update_bgm_remove_button_state()

    @Slot()
    def _on_timeline_scrub_started(self):
        """【MPV版】拖拽进度条时：强制挂起引擎"""
        self._is_scrubbing_timeline = True
        if hasattr(self, 'mpv_audio'):
            self.mpv_audio.pause = True
            self.mpv_bgm.pause = True

    @Slot()
    def _on_timeline_scrub_finished(self):
        """【MPV版】松开进度条时：重新对齐底层时间戳"""
        self._is_scrubbing_timeline = False
        self._sync_audio_engine_to_current_frame()

        if getattr(self, 'is_playing', False):
            if hasattr(self, 'mpv_audio') and getattr(self.mpv_audio, 'path', None):
                self.mpv_audio.pause = False

            idx = getattr(self, '_current_playing_clip_idx', -1)
            if idx != -1 and idx < len(getattr(self, 'virtual_timeline', [])):
                clip = self.virtual_timeline[idx]
                if clip.get('custom_audio_path') and getattr(self.mpv_bgm, 'path', None):
                    self.mpv_bgm.pause = False

    def _monitor_bgm_preview_playback(self):
        """【MPV版】BGM 独立预览物理边界循环监听"""
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'):
            self._stop_bgm_preview_monitoring()
            return

        clip = self.virtual_timeline[idx]
        bgm_path = clip.get('custom_audio_path')
        if not bgm_path or not os.path.exists(bgm_path):
            self._stop_bgm_preview_monitoring()
            return

        bgm_clip_start_sec = clip.get('custom_audio_clip_start', 0.0)
        bgm_clip_end_sec = clip.get('custom_audio_clip_end', 0.0)

        # 【核心修改】：读取 MPV 的播放时间
        current_pos_sec = getattr(self.mpv_bgm, 'time_pos', 0.0)
        if current_pos_sec is None:
            current_pos_sec = 0.0

        if current_pos_sec >= bgm_clip_end_sec:
            try:
                self.mpv_bgm.seek(bgm_clip_start_sec, reference='absolute', precision='exact')
                self.mpv_bgm.pause = False
            except Exception:
                pass

    def _update_bgm_preview_button_ui(self):
        """
        [New method] Refresh preview button icon style based on independent playback state.
        """
        if not hasattr(self, 'btn_bgm_play_preview'): return
        is_playing = getattr(self, '_is_playing_bgm_preview', False)
        icon_name = "pause-fill.svg" if is_playing else "play-fill.svg"
        if hasattr(self, '_create_svg_icon'):
            self.btn_bgm_play_preview.setIcon(self._create_svg_icon(icon_name, color=VID_TEXT_PRIMARY))

    def _update_audio_console_ui(self):
        """更新剪裁界面音频控制台面板的UI状态"""
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'):
            return
        clip = self.virtual_timeline[idx]

        # 检查原声是否被静音
        is_orig_muted = clip.get('mute_original', False) or clip.get('mute_all', False)
        orig_vol = int(clip.get('original_audio_volume', 1.0) * 100)

        self.btn_toggle_mute.blockSignals(True)
        self.btn_toggle_mute.setChecked(is_orig_muted)

        new_text_cn = "恢复声音" if is_orig_muted else "片段静音"
        self.btn_toggle_mute.setProperty("orig_text", new_text_cn)
        if hasattr(self, '_TR'):
            self.btn_toggle_mute.setText(self._TR(new_text_cn))
        else:
            self.btn_toggle_mute.setText(new_text_cn)

        self.btn_toggle_mute.blockSignals(False)

        self.orig_volume_slider.blockSignals(True)
        self.orig_volume_slider.setValue(orig_vol)
        self.orig_volume_slider.setEnabled(not is_orig_muted)
        self.orig_volume_slider.blockSignals(False)

        bgm_path = clip.get('custom_audio_path', None)
        if bgm_path and os.path.exists(bgm_path):
            self.bgm_info_label.setText(f"{_TR('音乐')}: {os.path.basename(bgm_path)}")
            self.bgm_info_label.setProperty("orig_text", f"音乐: {os.path.basename(bgm_path)}")
            self.bgm_info_label.setStyleSheet(
                "color: #34D399; font-weight: bold; font-size: 12px; background: transparent; padding: 2px 4px;")
            self.btn_add_bgm.hide()
            self.btn_remove_bgm.show()
            self.bgm_vol_container.show()
            self.bgm_start_container.show()
            self.bgm_clip_start_container.show()

            # 检查 BGM 是否被静音
            is_bgm_muted = clip.get('mute_bgm', False) or clip.get('mute_all', False)
            bgm_vol = int(clip.get('custom_audio_volume', 1.0) * 100)

            self.bgm_volume_slider.blockSignals(True)
            self.bgm_volume_slider.setValue(bgm_vol)
            self.bgm_volume_slider.setEnabled(not is_bgm_muted)
            self.bgm_volume_slider.blockSignals(False)

            bgm_start = clip.get('custom_audio_start_sec', 0.0)
            self.bgm_start_spinbox.blockSignals(True)
            self.bgm_start_spinbox.setValue(bgm_start)
            self.bgm_start_spinbox.blockSignals(False)

            bgm_clip_start = clip.get('custom_audio_clip_start', 0.0)
            bgm_clip_end = clip.get('custom_audio_clip_end', 0.0)

            bgm_duration = clip.get('_cached_bgm_duration', 0.0)
            if bgm_duration <= 0.0:
                bgm_duration = self._get_audio_file_duration_pure(bgm_path)
                clip['_cached_bgm_duration'] = bgm_duration

            if bgm_clip_end <= 0 or bgm_clip_end > bgm_duration:
                bgm_clip_end = bgm_duration
                clip['custom_audio_clip_end'] = bgm_duration

            self.bgm_trim_slider.blockSignals(True)
            self.bgm_trim_slider.set_range(0, int(bgm_duration))
            self.bgm_trim_slider.set_values(int(bgm_clip_start), int(bgm_clip_end))
            self.bgm_trim_slider.blockSignals(False)

            self.bgm_clip_time_label.setText(
                f"{self._format_time_simple(bgm_clip_start)} - {self._format_time_simple(bgm_clip_end)}")
        else:
            self.bgm_info_label.setText(_TR("无背景音乐"))
            self.bgm_info_label.setProperty("orig_text", "无背景音乐")
            self.bgm_info_label.setStyleSheet(
                "color: #888888; font-style: italic; font-size: 12px; background: transparent; padding: 2px 4px;")
            self.btn_add_bgm.show()
            self.btn_remove_bgm.hide()
            self.bgm_vol_container.hide()
            self.bgm_start_container.hide()
            self.bgm_clip_start_container.hide()

        self._update_bgm_preview_button_ui()
        self._update_bgm_remove_button_state()

    @Slot(str, int)
    def _add_video_to_storyboard(self, file_path, target_index=-1):
        """加载视频到故事板"""
        try:
            self._save_video_state()
            self._distribute_global_masks_to_clips()

            container = av.open(file_path)
            stream = container.streams.video[0]

            first_frame_h = stream.height
            first_frame_w = stream.width

            fps_val = stream.average_rate
            if not fps_val:
                fps_val = stream.r_frame_rate
            fps = float(fps_val) if fps_val else VIDEO_DEFAULT_FPS

            frames = stream.frames
            if not frames or frames <= 0:
                if stream.duration and stream.time_base:
                    frames = int(float(stream.duration * stream.time_base) * fps)
                else:
                    frames = 1

            duration = frames / fps

            first_frame_array = None
            for frame in container.decode(video=0):
                first_frame_array = frame.to_ndarray(format='bgr24')
                break

            container.close()
            pixmap = convert_cv_to_pixmap(first_frame_array) if first_frame_array is not None else QPixmap()

            # 初始化参数，包含分离的静音控制
            vid_data = {
                'path': file_path,
                'frames': frames,
                'total_file_frames': frames,
                'in_point': 0,
                'out_point': frames,
                'fps': fps,
                'duration': duration,
                'pixmap': pixmap,
                'local_masks': {},
                'mute_all': False,
                'mute_original': False,
                'mute_bgm': False,
                'original_audio_volume': 1.0,
                'custom_audio_path': None,
                'custom_audio_volume': 1.0,
                'custom_audio_start_sec': 0.0,
                'bg_color': QColor(0, 0, 0),
                'bg_image_path': None,
                'bg_is_transparent': True
            }

            if not hasattr(self, 'virtual_timeline'):
                self.virtual_timeline = []
            is_first_video_in_timeline = (len(self.virtual_timeline) == 0)

            if target_index != -1 and target_index <= len(self.virtual_timeline):
                self.virtual_timeline.insert(target_index, vid_data)
            else:
                self.virtual_timeline.append(vid_data)

            if is_first_video_in_timeline:
                self.video_path = file_path
                self.video_fps = fps
                self.video_height = first_frame_h
                self.video_width = first_frame_w
            else:
                self.video_width = max(getattr(self, 'video_width', 0), first_frame_w)
                self.video_height = max(getattr(self, 'video_height', 0), first_frame_h)

            self.storyboard_list.blockSignals(True)
            self.storyboard_list.clear()
            for v_data in self.virtual_timeline:
                item_widget = StoryboardItemWidget(v_data['path'], v_data['duration'], v_data.get('pixmap'))
                # 同步初始化时的 UI 图标
                all_muted = v_data.get('mute_all', False) or (
                            v_data.get('mute_original', False) and v_data.get('mute_bgm', False))
                item_widget.sync_ui_with_data(all_muted)

                list_item = QListWidgetItem(self.storyboard_list)
                list_item.setSizeHint(QSize(240, 170))
                list_item.setData(Qt.ItemDataRole.UserRole + 1, v_data)
                self.storyboard_list.setItemWidget(list_item, item_widget)
            self.storyboard_list.blockSignals(False)

            select_idx = target_index if target_index != -1 else self.storyboard_list.count() - 1
            if 0 <= select_idx < self.storyboard_list.count():
                self.storyboard_list.setCurrentRow(select_idx)
                self.storyboard_list.item(select_idx).setSelected(True)

            new_frame_index = sum(vid['frames'] for vid in self.virtual_timeline[:select_idx])
            self.current_frame_index = new_frame_index

            self._invalidate_timeline_cache(keep_masks=True)
            self._recalc_global_timeline()
            self._update_storyboard_ui()
            if hasattr(self, '_update_storyboard_highlight'):
                self._update_storyboard_highlight()

            self._gather_global_masks_from_clips()
            self._skip_rebake_on_next_extraction = True
            self._trigger_global_timeline_rebuild()

        except Exception as e:
            print(f"导入解析异常: {e}")
            import traceback
            traceback.print_exc()

    def _sync_timeline_order_from_ui(self, source_row=-1, drop_row=-1):
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline: return

        # 解除文件锁，防止拖拽排序引发崩溃
        self._clear_audio_sources_safe()

        self._distribute_global_masks_to_clips()

        is_reordered = False
        if source_row != -1 and drop_row != -1 and source_row != drop_row:
            self._save_video_state()
            item_data = self.virtual_timeline.pop(source_row)
            self.virtual_timeline.insert(drop_row, item_data)
            is_reordered = True

        current_focus = getattr(self, '_current_crop_clip_idx', -1)
        if current_focus == -1:
            selected_items = self.storyboard_list.selectedItems()
            if selected_items: current_focus = self.storyboard_list.row(selected_items[0])

        self.storyboard_list.blockSignals(True)
        self.storyboard_list.clear()

        for vid_data in self.virtual_timeline:
            item_widget = StoryboardItemWidget(vid_data['path'], vid_data['duration'], vid_data.get('pixmap'))
            # [Core fix]: Force refresh UI audio icon state
            item_widget.sync_ui_with_data(vid_data.get('mute_original', False))
            list_item = QListWidgetItem(self.storyboard_list)
            list_item.setSizeHint(QSize(240, 170))
            list_item.setData(Qt.ItemDataRole.UserRole + 1, vid_data)
            self.storyboard_list.setItemWidget(list_item, item_widget)

        self.storyboard_list.blockSignals(False)

        if 0 <= current_focus < self.storyboard_list.count():
            self.storyboard_list.setCurrentRow(current_focus)
            self.storyboard_list.item(current_focus).setSelected(True)
        elif self.storyboard_list.count() > 0:
            last_idx = self.storyboard_list.count() - 1
            self.storyboard_list.setCurrentRow(last_idx)
            self.storyboard_list.item(last_idx).setSelected(True)

        self.video_inference_state = None
        self._timeline_signature = None
        self._recalc_global_timeline()

        if is_reordered:
            self.processed_masks = {}
            self.target_points = {}
            self._invalidate_timeline_cache(keep_masks=False)
            self._gather_global_masks_from_clips()
            self._trigger_global_timeline_rebuild()

        if getattr(self, 'is_playing', False): self.pause_video()
        if getattr(self, 'current_frame_index', 0) >= getattr(self, 'total_frames', 0):
            self.seek_video_via_scrubber(0)

    @Slot()
    def _delete_checked_storyboard_items(self):
        self._clear_audio_sources_safe()

        self._distribute_global_masks_to_clips()

        indices_to_delete = []
        for i in range(self.storyboard_list.count()):
            item = self.storyboard_list.item(i)
            widget = self.storyboard_list.itemWidget(item)
            if isinstance(widget, StoryboardItemWidget) and widget.is_checked():
                indices_to_delete.append(i)

        if not indices_to_delete:
            selected = self.storyboard_list.selectedItems()
            if selected: indices_to_delete.append(self.storyboard_list.row(selected[0]))
        if not indices_to_delete: return

        self._save_video_state()
        indices_to_delete.sort(reverse=True)
        for i in indices_to_delete:
            if i < len(self.virtual_timeline): self.virtual_timeline.pop(i)

        self.processed_masks = {}
        self.target_points = {}
        self._invalidate_timeline_cache(keep_masks=True)

        current_focus = -1
        selected_items = self.storyboard_list.selectedItems()
        if selected_items: current_focus = self.storyboard_list.row(selected_items[0])

        self.storyboard_list.blockSignals(True)
        self.storyboard_list.clear()

        for vid_data in self.virtual_timeline:
            item_widget = StoryboardItemWidget(vid_data['path'], vid_data['duration'], vid_data.get('pixmap'))
            # [Core fix]: Force refresh UI audio icon state
            item_widget.sync_ui_with_data(vid_data.get('mute_original', False))
            list_item = QListWidgetItem(self.storyboard_list)
            list_item.setSizeHint(QSize(240, 170))
            list_item.setData(Qt.ItemDataRole.UserRole + 1, vid_data)
            self.storyboard_list.setItemWidget(list_item, item_widget)

        self.storyboard_list.blockSignals(False)

        if 0 <= current_focus < self.storyboard_list.count():
            self.storyboard_list.setCurrentRow(current_focus)
            self.storyboard_list.item(current_focus).setSelected(True)
        elif self.storyboard_list.count() > 0:
            last_idx = self.storyboard_list.count() - 1
            self.storyboard_list.setCurrentRow(last_idx)
            self.storyboard_list.item(last_idx).setSelected(True)

        self.video_inference_state = None
        self._timeline_signature = None
        self._recalc_global_timeline()

        self._gather_global_masks_from_clips()
        self._update_storyboard_ui()
        self._skip_rebake_on_next_extraction = True
        self._trigger_global_timeline_rebuild()

    def _distribute_global_masks_to_clips(self):
        """[隔离保护分布]：将抠图蒙版和交互点无损锁定到它们物理所属的独立视频片段中"""
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline:
            return

        current_count = 0
        for idx, vid in enumerate(self.virtual_timeline):
            is_matting_active = getattr(self, '_matting_clip_idx', -1) != -1
            is_crop_active = getattr(self, '_current_crop_clip_idx', -1) != -1

            if is_matting_active and idx != self._matting_clip_idx:
                current_count += vid['frames']
                continue
            if is_crop_active and idx != self._current_crop_clip_idx:
                current_count += vid['frames']
                continue

            local_masks = vid.get('local_masks', {})
            local_targets = vid.get('local_targets', {})
            in_point = int(vid.get('in_point', 0))

            visible_physical_frames = set()
            for local_idx in range(vid['frames']):
                global_idx = current_count + local_idx
                physical_idx = in_point + local_idx
                visible_physical_frames.add(physical_idx)

                # 将全局可见数据绑定到物理帧
                if getattr(self, 'processed_masks', None) and global_idx in self.processed_masks:
                    import copy
                    local_masks[physical_idx] = copy.deepcopy(self.processed_masks[global_idx])
                else:
                    # 如果全局字典中没有了，但它在可见区间内，说明是被用户清空/擦除了，需要同步删除
                    if physical_idx in local_masks:
                        del local_masks[physical_idx]

            # 【对象隔离】：找出在可见区域内，但已经被用户全局删除的目标，并进行清理
            tids_to_remove = []
            for tid, tdata in local_targets.items():
                if tdata.get('annotation_frame') in visible_physical_frames:
                    if not getattr(self, 'target_points', None) or tid not in self.target_points:
                        tids_to_remove.append(tid)
            for tid in tids_to_remove:
                del local_targets[tid]

            if getattr(self, 'target_points', None):
                for tid, tdata in self.target_points.items():
                    global_ann_frame = tdata.get('annotation_frame')
                    if global_ann_frame is not None and current_count <= global_ann_frame < current_count + vid['frames']:
                        physical_ann = in_point + (global_ann_frame - current_count)
                        import copy
                        tdata_copy = copy.deepcopy(tdata)
                        tdata_copy['annotation_frame'] = physical_ann
                        local_targets[tid] = tdata_copy

            vid['local_masks'] = local_masks
            vid['local_targets'] = local_targets
            current_count += vid['frames']

    def _gather_global_masks_from_clips(self):
        """[全量汇总]：从片段收集蒙版时，如果对象的标注帧被剪裁掉了，自动映射它在当前区间的最新形态"""
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline:
            self.processed_masks = {}
            self.target_points = {}
            self.video_segmentation_finished = False
            return

        new_processed_masks = {}
        new_target_points = {}
        current_count = 0

        for vid in self.virtual_timeline:
            local_masks = vid.get('local_masks', {})
            local_targets = vid.get('local_targets', {})
            in_point = int(vid.get('in_point', 0))

            for local_idx in range(vid['frames']):
                physical_idx = in_point + local_idx
                global_idx = current_count + local_idx
                if physical_idx in local_masks:
                    new_processed_masks[global_idx] = local_masks[physical_idx]

            for tid, tdata in local_targets.items():
                import copy
                tdata_copy = copy.deepcopy(tdata)
                physical_idx = tdata_copy['annotation_frame']

                if in_point <= physical_idx < in_point + vid['frames']:
                    offset = physical_idx - in_point
                    tdata_copy['annotation_frame'] = current_count + offset
                    new_target_points[tid] = tdata_copy
                else:
                    # 【无损剪辑】：如果标注帧不在可视范围内(被剪掉了)，在可视范围内找第一帧存在蒙版的数据作为锚点
                    import numpy as np
                    has_mask_inside = False
                    first_visible_offset = -1
                    for local_idx in range(vid['frames']):
                        phys_idx = in_point + local_idx
                        if phys_idx in local_masks and tid in local_masks[phys_idx] and local_masks[phys_idx][
                            tid] is not None:
                            if np.any(local_masks[phys_idx][tid]):
                                has_mask_inside = True
                                first_visible_offset = local_idx
                                break

                    if has_mask_inside:
                        tdata_copy['annotation_frame'] = current_count + first_visible_offset
                        tdata_copy['points'] = []  # 原始锚点不在当前帧，置空防止坐标错乱
                        tdata_copy['box'] = None
                        new_target_points[tid] = tdata_copy

            current_count += vid['frames']

        self.processed_masks = new_processed_masks
        self.target_points = new_target_points
        self.video_segmentation_finished = len(self.processed_masks) > 0


    def _recalc_global_timeline(self):
        self.total_frames = sum(vid['frames'] for vid in getattr(self, 'virtual_timeline', []))
        total_time_sec = sum(vid['duration'] for vid in getattr(self, 'virtual_timeline', []))

        if hasattr(self, 'time_label_total'):
            self.time_label_total.setText(self._format_time(total_time_sec))

        # =========================================================
        # Refresh UI progress bar parameters to [Post-merge] full parameters
        # =========================================================
        if hasattr(self, 'video_thumbnail_scrubber'):
            self.video_thumbnail_scrubber.set_params(
                self.total_frames,
                getattr(self, 'video_thumbnail_paths', []),
                getattr(self, 'video_width', 0),
                getattr(self, 'video_height', 0),
                getattr(self, 'video_fps', 30.0),
                getattr(self, 'is_gif_input', False),
                getattr(self, 'gif_frame_durations_ms', [])
            )
            # When returning to home page, slider locates to the correct coordinate in current merged timeline
            self.video_thumbnail_scrubber.set_current_frame(getattr(self, 'current_frame_index', 0))

        if hasattr(self, 'video_frame_spinbox'):
            self.video_frame_spinbox.setMaximum(max(1, self.total_frames))

        if self.total_frames > 0:
            if getattr(self, 'current_frame_index', 0) >= self.total_frames:
                self.current_frame_index = 0
            self._display_frame_wrapper(self.current_frame_index)

            # 【核心修复3】：文件刚加载或时间线重组完成后（停留在0秒或开头处）
            # 立即触发一次后台音频预热，保障用户首次点击播放时实现瞬间秒开！
            self._sync_audio_engine_to_current_frame()
        else:
            self.current_frame_index = -1
            if hasattr(self, 'video_display_label'): self.video_display_label.clear_display()
            if hasattr(self, 'time_label_curr'): self.time_label_curr.setText("0:00.00")
            # 只有在列表彻底清空时，才彻底释放音频引擎句柄
            self._clear_audio_sources_safe()

    def _format_time(self, seconds: float):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 100)
        return f"{mins}:{secs:02d}.{ms:02d}"

    def _update_video_time_label(self):
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline or getattr(self, 'total_frames', 0) <= 0:
            if hasattr(self, 'video_thumbnail_scrubber'): self.video_thumbnail_scrubber.set_info_text("", "[0/0]")
            if hasattr(self, 'time_label_curr'): self.time_label_curr.setText("0:00.00")
            return

        is_isolated = getattr(self, '_matting_clip_idx', -1) != -1
        if is_isolated:
            clip_idx = self._matting_clip_idx
            if 0 <= clip_idx < len(self.virtual_timeline):
                clip = self.virtual_timeline[clip_idx]
                local_frame = self.current_frame_index - getattr(self, '_matting_global_start', 0)
                current_time_sec = max(0, local_frame) / clip['fps']
                if hasattr(self, 'time_label_curr'):
                    self.time_label_curr.setText(self._format_time(current_time_sec))
                if hasattr(self, 'video_thumbnail_scrubber'):
                    self.video_thumbnail_scrubber.set_info_text("", f"[{local_frame + 1}/{clip['frames']}]")
            return

        current_time_sec = 0.0
        current_count = 0
        for vid in self.virtual_timeline:
            if self.current_frame_index < current_count + vid['frames']:
                local_frame = self.current_frame_index - current_count
                current_time_sec += local_frame / vid['fps']
                break
            current_time_sec += vid['duration']
            current_count += vid['frames']

        if hasattr(self, 'time_label_curr'):
            self.time_label_curr.setText(self._format_time(current_time_sec))
        if hasattr(self, 'video_thumbnail_scrubber'):
            self.video_thumbnail_scrubber.set_info_text("", f"[{self.current_frame_index + 1}/{self.total_frames}]")

    def _invalidate_timeline_cache(self, keep_masks=False):
        """
        使时间线物理帧缓存失效。
        """
        # 1. 销毁原始帧解复用缓存
        if getattr(self, 'temp_frame_dir', None) and os.path.exists(self.temp_frame_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_frame_dir)
            except Exception:
                pass
        self.temp_frame_dir = None

        # 2. 强力销毁已过期的预渲染烘焙融合缓存
        if getattr(self, 'temp_render_dir', None) and os.path.exists(self.temp_render_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_render_dir)
            except Exception:
                pass
        self.temp_render_dir = None

        # 3. 彻底重设缩略图临时路径
        self.video_thumbnail_paths = []

        # 4. 重置烘焙播放标志位，迫使播放器读取全新生成的 temp_frame_dir
        self.video_segmentation_finished = False

        # 5. 释放解码器句柄
        if hasattr(self, '_cv_caps'):
            for uid, cap in list(self._cv_caps.items()):
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
            self._cv_caps.clear()

        # =========================================================================
        # 【核心修复】：物理帧缓存失效时，必须同步注销异步预加载池与内存预读字典，切断前向关联。
        # =========================================================================
        if hasattr(self, '_prefetch_pool'):
            try:
                self._prefetch_pool.shutdown(wait=False)
            except Exception:
                pass
            del self._prefetch_pool
        if hasattr(self, '_video_frame_cache'):
            self._video_frame_cache.clear()
        if hasattr(self, '_prefetching_indices'):
            self._prefetching_indices.clear()

        # 7. 根据需要决定是否保留遮罩草稿
        if not keep_masks:
            if hasattr(self, 'processed_masks'):
                self.processed_masks.clear()
            if hasattr(self, 'target_points'):
                for target_id in list(self.target_points.keys()):
                    self.target_points[target_id]['points'] = []
                    self.target_points[target_id]['box'] = None
                    self.target_points[target_id]['annotation_frame'] = -1
            self.video_segmentation_finished = False

        self.video_inference_state = None

        # 强制重置沙盒标识
        self._last_built_sandbox_clip_idx = -1


    @Slot()
    def _toggle_vid_left_panel_animated(self):
        current_width = self.vid_left_panel.width()
        is_collapsed = current_width < 100
        self.vid_left_content.setGraphicsEffect(None)

        if is_collapsed:
            # Expand left panel
            target_width = (self.vid_main_editor_page.width() - 40) // 2
            self.collapse_btn.setText("＜")
            self.vid_left_content.show()
            self.lib_title.show()
        else:
            # Collapse left panel
            target_width = 48
            self.collapse_btn.setText("＞")
            self.vid_left_content.hide()
            self.lib_title.hide()

        anim_panel_min = QPropertyAnimation(self.vid_left_panel, b"minimumWidth")
        anim_panel_max = QPropertyAnimation(self.vid_left_panel, b"maximumWidth")

        self._current_panel_anim = QParallelAnimationGroup(self)
        for anim in [anim_panel_min, anim_panel_max]:
            anim.setDuration(350)
            anim.setEasingCurve(QEasingCurve.Type.OutQuart)
            anim.setStartValue(current_width)
            anim.setEndValue(target_width)
            self._current_panel_anim.addAnimation(anim)

        self._current_panel_anim.start()

    def _init_vid_main_editor_page(self):
        self.vid_main_editor_page = QWidget()
        self.vid_main_editor_page.setObjectName("MainEditorPage")
        self.vid_main_editor_page.setStyleSheet(f"QWidget#MainEditorPage {{ background-color: {VID_DARK_BG}; }}")
        main_layout = QVBoxLayout(self.vid_main_editor_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_half_widget = QWidget()
        top_half_layout = QHBoxLayout(top_half_widget)
        top_half_layout.setContentsMargins(24, 24, 24, 16)
        top_half_layout.setSpacing(40)

        self.vid_left_panel = QWidget()
        self.vid_left_panel.setStyleSheet(f"background-color: {VID_PANEL_BG}; border-radius: 8px;")
        left_layout = QVBoxLayout(self.vid_left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.lib_header = QWidget()
        self.lib_header.setFixedHeight(48)
        header_layout = QHBoxLayout(self.lib_header)
        header_layout.setContentsMargins(8, 8, 8, 0)

        self.lib_title = QLabel("项目库")
        self.lib_title.setStyleSheet(
            f"color: {VID_TEXT_PRIMARY}; font-size: 14px; font-weight: bold; background:transparent;")
        self.lib_title.setContentsMargins(8, 0, 0, 0)

        self.collapse_btn = QToolButton()
        self.collapse_btn.setText("＜")
        self.collapse_btn.setFixedSize(32, 32)
        self.collapse_btn.setStyleSheet(
            f"color: {VID_ACCENT}; font-size: 18px; font-weight: bold; border-radius: 4px; background-color: transparent;")
        self.collapse_btn.clicked.connect(self._toggle_vid_left_panel_animated)

        header_layout.addWidget(self.lib_title)
        header_layout.addStretch()
        header_layout.addWidget(self.collapse_btn, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        left_layout.addWidget(self.lib_header)

        self.vid_left_content = QWidget()
        content_layout = QVBoxLayout(self.vid_left_content)
        content_layout.setContentsMargins(16, 0, 16, 16)

        add_bar = QHBoxLayout()
        self.load_video_button_top = QPushButton("+ 添加")
        self.load_video_button_top.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {VID_TEXT_PRIMARY}; border: 1px solid #555555; border-radius: 4px; padding: 6px 16px; font-weight: bold; font-size: 13px; }} QPushButton:hover {{ background-color: #333333; }}")
        self.load_video_button_top.clicked.connect(self.select_multiple_videos_for_library)

        add_bar.addWidget(self.load_video_button_top)
        add_bar.addStretch()
        content_layout.addLayout(add_bar)

        self.vid_asset_stack = QStackedWidget()
        self.vid_asset_stack.setStyleSheet("background: transparent;")
        self.vid_empty_state_widget = QWidget()
        empty_layout = QVBoxLayout(self.vid_empty_state_widget)
        empty_icon = QLabel()
        if hasattr(self, '_create_svg_icon'): empty_icon.setPixmap(
            self._create_svg_icon("images.svg", size=56, color=VID_TEXT_SECONDARY).pixmap(56, 56))
        empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_title = QLabel("你的项目库为空")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_title.setStyleSheet(
            f"color: {VID_TEXT_PRIMARY}; font-weight: bold; font-size: 14px; margin-top: 10px; background:transparent;")
        empty_layout.addStretch(1)
        empty_layout.addWidget(empty_icon)
        empty_layout.addWidget(empty_title)
        empty_layout.addStretch(2)

        self.vid_asset_list = AssetLibraryListWidget()
        self.vid_asset_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.vid_asset_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.vid_asset_stack.addWidget(self.vid_empty_state_widget)
        self.vid_asset_stack.addWidget(self.vid_asset_list)
        content_layout.addWidget(self.vid_asset_stack, 1)
        left_layout.addWidget(self.vid_left_content, 1)

        self.main_workspace_wrapper = QWidget()
        self.main_workspace_wrapper.setStyleSheet("background: transparent;")
        top_half_layout.addWidget(self.vid_left_panel, 1)
        top_half_layout.addWidget(self.main_workspace_wrapper, 1)
        main_layout.addWidget(top_half_widget, 1)

        bottom_half_widget = QWidget()
        bottom_half_widget.setFixedHeight(280)
        bottom_half_widget.setStyleSheet(f"background-color: {VID_PANEL_BG}; border: none;")

        bottom_layout = QVBoxLayout(bottom_half_widget)
        bottom_layout.setContentsMargins(24, 16, 24, 16)

        sb_toolbar = QHBoxLayout()
        sb_title = QLabel("合并栏")
        sb_title.setStyleSheet(
            f"color: {VID_TEXT_PRIMARY}; font-weight: bold; font-size: 14px; background:transparent;")
        sb_toolbar.addWidget(sb_title)
        sb_toolbar.addStretch(1)

        tools = [
            ("进入剪辑", "scissors.svg", self._enter_dedicated_crop_mode),
            ("智能抠图", "person-bounding-box.svg", self._enter_dedicated_matting_mode),
        ]
        for text, icon, func in tools:
            btn = QToolButton()
            btn.setText(text)
            if hasattr(self, '_create_svg_icon'): btn.setIcon(self._create_svg_icon(icon, color=VID_TEXT_PRIMARY))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setStyleSheet(VID_MODERN_BTN_STYLE)
            if func: btn.clicked.connect(func)
            sb_toolbar.addWidget(btn)

        btn_trash = QToolButton()
        if hasattr(self, '_create_svg_icon'): btn_trash.setIcon(
            self._create_svg_icon("trash.svg", color=VID_TEXT_PRIMARY))
        btn_trash.setStyleSheet(VID_MODERN_BTN_STYLE)
        btn_trash.clicked.connect(self._delete_checked_storyboard_items)
        sb_toolbar.addWidget(btn_trash)

        bottom_layout.addLayout(sb_toolbar)
        bottom_layout.addSpacing(10)

        track_overlay_container = QWidget()
        track_overlay_container.setStyleSheet("background: transparent;")

        # 1. Use QStackedLayout to ensure three layers absolute top-left alignment
        stack_layout = QStackedLayout(track_overlay_container)
        stack_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        stack_layout.setContentsMargins(0, 0, 0, 0)

        # ------------------- Bottom layer: empty dashed box layer -------------------
        self.sb_empty_container = QWidget()
        self.sb_empty_container.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        sb_empty_layout = QHBoxLayout(self.sb_empty_container)

        # Precise mathematical alignment
        sb_empty_layout.setContentsMargins(16, 17, 0, 0)
        sb_empty_layout.setSpacing(16)
        sb_empty_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        for i in range(6):
            box = QFrame()
            box.setFixedSize(240, 135)
            box.setStyleSheet(
                "background-color: rgba(255,255,255,0.02); border-radius: 6px; border: 1px dashed #404040;")
            sb_empty_layout.addWidget(box)

        sb_empty_layout.addStretch(1)

        # ------------------- Middle layer: globally centered HD guide text -------------------
        # We make text an independent layer, cover above the dashed box, and set absolute horizontal and vertical center
        self.sb_guide_text_layer = QWidget()
        self.sb_guide_text_layer.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        guide_layout = QVBoxLayout(self.sb_guide_text_layer)
        # Move text down slightly as a whole, align visual center with dashed box below
        guide_layout.setContentsMargins(0, 15, 0, 0)
        guide_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        guide_icon = QLabel()
        if hasattr(self, '_create_svg_icon'):
            # Enlarge icon, size set to 48x48, use pure HD rendering
            guide_icon.setPixmap(
                self._create_svg_icon("file-earmark-plus.svg", size=48, color=QColor("#666666")).pixmap(48, 48))
        guide_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        guide_text = QLabel("拖拽到此添加并合并视频")
        # [Core]: Enlarge font size (18px), increase letter spacing, adjust text color to softer advanced gray
        guide_text.setStyleSheet("""
                    color: #666666; 
                    font-size: 18px; 
                    font-weight: bold; 
                    letter-spacing: 2px;
                    border: none; 
                    background: transparent;
                """)
        guide_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        guide_layout.addWidget(guide_icon)
        guide_layout.addSpacing(16)
        guide_layout.addWidget(guide_text)

        # ------------------- Top layer: actual storyboard list -------------------
        self.storyboard_list = StoryboardTrackWidget()

        self.storyboard_list.setStyleSheet("""
                    QListWidget { background-color: transparent; border: none; outline: none; padding: 0px; margin: 0px; }
                    QListWidget::item { padding: 0px; margin: 0px; background-color: transparent; border: none; }
                    QScrollBar:horizontal { border: none; background: transparent; height: 8px; margin: 0px; }
                    QScrollBar::handle:horizontal { background: #333333; min-width: 40px; border-radius: 4px; }
                    QScrollBar::handle:horizontal:hover { background: #1A73E8; }
                    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { border: none; background: none; width: 0px; }
                    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
                """)

        self.storyboard_list.item_dropped.connect(self._add_video_to_storyboard)
        self.storyboard_list.track_reordered.connect(self._sync_timeline_order_from_ui)
        self.storyboard_list.itemClicked.connect(self._on_storyboard_item_clicked)

        # Push into stacked layout: bottom is dashed box -> middle is text -> top is video track
        stack_layout.addWidget(self.sb_empty_container)
        stack_layout.addWidget(self.sb_guide_text_layer)
        stack_layout.addWidget(self.storyboard_list)

        bottom_layout.addWidget(track_overlay_container, 1)
        main_layout.addWidget(bottom_half_widget, 0)

    @Slot(QListWidgetItem)
    def _on_storyboard_item_clicked(self, item):
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline: return
        idx = self.storyboard_list.row(item)
        if idx < 0 or idx >= len(self.virtual_timeline): return

        if getattr(self, 'is_playing', False): self.pause_video()

        # =========================================================
        # [Core fix 1]: If currently in independent sandbox of "Smart Matting" or "Crop"
        # Clicking storyboard should not just jump time, but should reload entire sandbox environment!
        # =========================================================
        current_page = getattr(self, 'vid_editor_stack', None) and self.vid_editor_stack.currentWidget()
        if current_page == getattr(self, 'vid_dedicated_matting_page', None):
            self._enter_dedicated_matting_mode()
            return
        elif current_page == getattr(self, 'vid_dedicated_crop_page', None):
            self._enter_dedicated_crop_mode()
            return

        # Normal home page jump logic
        target_global_frame = sum(vid['frames'] for vid in self.virtual_timeline[:idx])
        self.current_frame_index = target_global_frame

        if hasattr(self, '_cv_caps'):
            for uid, cap in list(self._cv_caps.items()):
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
            self._cv_caps.clear()

        if hasattr(self, '_last_valid_frames'): self._last_valid_frames.clear()

        for vid in self.virtual_timeline: vid['_manual_pos'] = -1

        self._display_frame_wrapper(self.current_frame_index)
        if hasattr(self, '_update_storyboard_highlight'): self._update_storyboard_highlight()
        QApplication.processEvents()

    def _update_storyboard_ui(self):
        # If there is video in storyboard, hide dashed box and guide text
        if getattr(self, 'storyboard_list', None) and self.storyboard_list.count() > 0:
            if hasattr(self, 'sb_empty_container'): self.sb_empty_container.hide()
            if hasattr(self, 'sb_guide_text_layer'): self.sb_guide_text_layer.hide()
        else:
            # If all videos are cleared, re-show dashed box and guide text
            if hasattr(self, 'sb_empty_container'): self.sb_empty_container.show()
            if hasattr(self, 'sb_guide_text_layer'): self.sb_guide_text_layer.show()

    def _on_video_asset_added(self):
        self._update_asset_library_ui()
        self._update_storyboard_ui()

    @Slot(bool)
    def _on_direct_matting_checkbox_toggled(self, checked):
        """
        [新增]：监听无需定位直达模式的勾选状态。
        一旦用户取消勾选（想使用粗追踪定位），立刻自动后台提取时序特征。
        """
        if not checked:
            target_dir = getattr(self, 'clip_sandbox_dir', getattr(self, 'temp_frame_dir', None))
            if getattr(self, 'video_path', None) and target_dir:
                if getattr(self, 'video_inference_state', None) is None:
                    if not getattr(self, 'is_extracting_frames', False) and getattr(self, 'video_predictor_loaded',
                                                                                    False):
                        self._initialize_video_predictor_state()

    def _init_vid_dedicated_matting_page(self):
        self.vid_dedicated_matting_page = QWidget()
        self.vid_dedicated_matting_page.setObjectName("VidMattingPage")
        self.vid_dedicated_matting_page.setStyleSheet(f"QWidget#VidMattingPage {{ background-color: {VID_DARK_BG}; }}")

        # =========================================================
        # 全屏磨砂大图预览背景组件
        # =========================================================
        self._bg_full_preview_overlay = FullScreenImageOverlay(self.window())

        # 主体布局结构
        main_layout = QHBoxLayout(self.vid_dedicated_matting_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left_area = QWidget()
        left_layout = QVBoxLayout(left_area)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # 顶部切换栏
        switcher_container = QWidget()
        switcher_container.setFixedHeight(54)
        switcher_layout = QHBoxLayout(switcher_container)
        switcher_layout.setContentsMargins(0, 10, 0, 0)
        switcher_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        tools = [
            ("剪裁", "scissors.svg", lambda: (self._apply_matting_and_exit(), self._enter_dedicated_crop_mode())),
            ("智能抠图", "person-bounding-box.svg", None),
        ]

        switcher_group = QWidget()
        switcher_group.setStyleSheet("background-color: transparent;")
        sg_layout = QHBoxLayout(switcher_group)
        sg_layout.setContentsMargins(0, 0, 0, 0)
        sg_layout.setSpacing(6)

        for text, icon, func in tools:
            btn = QToolButton()
            btn.setText(text)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            if hasattr(self, '_create_svg_icon'):
                btn.setIcon(self._create_svg_icon(icon, color=VID_TEXT_PRIMARY))

            if text == "智能抠图":
                btn.setStyleSheet(
                    "QToolButton { background-color: #0A84FF; color: #FFFFFF; border-radius: 8px; padding: 6px 14px; font-size: 13px; font-weight: bold; border: none; }")
            else:
                btn.setStyleSheet(
                    "QToolButton { background-color: transparent; color: #E0E0E0; border-radius: 8px; padding: 6px 14px; font-size: 13px; border: none; } QToolButton:hover { background-color: #333333; }")
                if func: btn.clicked.connect(func)
            sg_layout.addWidget(btn)

        switcher_layout.addWidget(switcher_group)
        left_layout.addWidget(switcher_container, 0)

        self.matting_workspace_wrapper = QWidget()
        self.matting_workspace_wrapper.setStyleSheet("background: transparent;")
        left_layout.addWidget(self.matting_workspace_wrapper, 1)

        main_layout.addWidget(left_area, 1)

        # 右侧操作属性面板 (限定宽度 320px)
        right_panel = QWidget()
        right_panel.setFixedWidth(320)
        right_panel.setStyleSheet("background-color: #1C1C1E; border: none;")

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 0, 16, 24)
        right_layout.setSpacing(16)

        rp_title_container = QWidget()
        rp_title_container.setFixedHeight(44)
        rp_title_layout = QHBoxLayout(rp_title_container)
        rp_title_layout.setContentsMargins(0, 0, 0, 0)
        rp_title = QLabel("智能抠图")
        rp_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rp_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        rp_title_layout.addWidget(rp_title)
        right_layout.addWidget(rp_title_container, 0)

        def create_section_title(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size: 12px; color: #8E8E93; font-weight: bold; margin-bottom: 2px;")
            return lbl

        UNIFORM_BTN_STYLE = """
            QPushButton { 
                background-color: #2C2C2E; 
                color: #FFFFFF; 
                border-radius: 6px; 
                padding: 10px 12px; 
                font-weight: bold; 
                font-size: 13px; 
                border: 1px solid #3A3A3C; 
            } 
            QPushButton:hover { 
                background-color: #3A3A3C; 
                border: 1px solid #555;
            }
            QPushButton:disabled {
                color: #555558;
                background-color: #1C1C1E;
                border: 1px solid #2C2C2E;
            }
        """

        # ==================== 1. 追踪对象管理 ====================
        right_layout.addWidget(create_section_title("追踪对象"))

        self.video_objects_list = QListWidget()
        self.video_objects_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.video_objects_list.setFixedHeight(140)
        self.video_objects_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; outline: none; padding: 0px; margin: 0px;} 
            QListWidget::item { background: transparent; color: #FFF; border-radius: 6px; margin-bottom: 2px; border: 1px solid transparent; } 
            QListWidget::item:selected { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.15); }
            QScrollBar:vertical { width: 0px; }
        """)

        if hasattr(self, '_on_video_object_selection_changed'):
            self.video_objects_list.itemSelectionChanged.connect(self._on_video_object_selection_changed)

        original_mouse_press = self.video_objects_list.mousePressEvent

        def custom_mouse_press(event):
            item = self.video_objects_list.itemAt(event.pos())
            if not item: self.video_objects_list.clearSelection()
            original_mouse_press(event)

        self.video_objects_list.mousePressEvent = custom_mouse_press
        right_layout.addWidget(self.video_objects_list, 0)

        obj_actions_layout = QHBoxLayout()
        obj_actions_layout.setContentsMargins(0, 0, 0, 0)
        obj_actions_layout.setSpacing(10)

        self.add_target_button = QPushButton("+ 添加新对象")
        self.add_target_button.setStyleSheet(UNIFORM_BTN_STYLE)
        if hasattr(self, 'add_new_video_target'):
            self.add_target_button.clicked.connect(self.add_new_video_target)

        self.delete_current_target_button = QPushButton("删除对象")
        self.delete_current_target_button.setStyleSheet(UNIFORM_BTN_STYLE)
        if hasattr(self, 'clear_current_video_target'):
            self.delete_current_target_button.clicked.connect(self.clear_current_video_target)

        obj_actions_layout.addWidget(self.add_target_button)
        obj_actions_layout.addWidget(self.delete_current_target_button)
        right_layout.addLayout(obj_actions_layout, 0)

        undo_redo_layout = QHBoxLayout()
        undo_redo_layout.setContentsMargins(0, 0, 0, 0)
        undo_redo_layout.setSpacing(10)

        self.vid_mat_undo_button = QPushButton("撤销")
        self.vid_mat_undo_button.setStyleSheet(UNIFORM_BTN_STYLE)
        self.vid_mat_undo_button.clicked.connect(self.undo_video_action)

        self.vid_mat_redo_button = QPushButton("重做")
        self.vid_mat_redo_button.setStyleSheet(UNIFORM_BTN_STYLE)
        self.vid_mat_redo_button.clicked.connect(self.redo_video_action)

        undo_redo_layout.addWidget(self.vid_mat_undo_button)
        undo_redo_layout.addWidget(self.vid_mat_redo_button)
        right_layout.addLayout(undo_redo_layout, 0)

        right_layout.addSpacing(4)

        # ==================== 2. 交互式蒙版修复 ====================
        right_layout.addWidget(create_section_title("微调与修补"))

        seg_container = QWidget()
        seg_container.setFixedHeight(30)
        seg_container.setStyleSheet("background-color: rgba(255,255,255,0.05); border-radius: 6px; padding: 2px;")
        seg_layout = QHBoxLayout(seg_container)
        seg_layout.setContentsMargins(2, 2, 2, 2)
        seg_layout.setSpacing(2)

        self.vid_mode_point_btn = QPushButton("点 / 框 标注")
        self.vid_mode_brush_btn = QPushButton("画笔微调")

        def set_seg_style(btn, is_active):
            if is_active:
                btn.setStyleSheet(
                    "background: #3A3A3C; color: #FFF; border-radius: 4px; font-weight: bold; font-size: 12px; border: none;")
            else:
                btn.setStyleSheet(
                    "background: transparent; color: #888; border-radius: 4px; font-weight: bold; font-size: 12px; border: none;")

        set_seg_style(self.vid_mode_point_btn, True)
        set_seg_style(self.vid_mode_brush_btn, False)

        self.vid_mode_point_btn.setCheckable(True)
        self.vid_mode_brush_btn.setCheckable(True)
        self.vid_mode_point_btn.setChecked(True)

        mode_group = QButtonGroup(self)
        mode_group.addButton(self.vid_mode_point_btn)
        mode_group.addButton(self.vid_mode_brush_btn)

        def on_seg_changed():
            is_point = self.vid_mode_point_btn.isChecked()
            set_seg_style(self.vid_mode_point_btn, is_point)
            set_seg_style(self.vid_mode_brush_btn, not is_point)

        self.vid_mode_point_btn.toggled.connect(on_seg_changed)

        seg_layout.addWidget(self.vid_mode_point_btn)
        seg_layout.addWidget(self.vid_mode_brush_btn)
        right_layout.addWidget(seg_container)

        def create_minimal_slider_row(name, s_min, s_max, s_val, suffix=""):
            row = QHBoxLayout()
            row.setContentsMargins(4, 0, 4, 0)
            lbl = QLabel(name)
            lbl.setFixedWidth(30)
            lbl.setStyleSheet("color: #FFFFFF; font-size: 13px; background: transparent;")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(s_min, s_max)
            slider.setValue(s_val)
            slider.setStyleSheet("""
                QSlider { min-height: 20px; background: transparent; }
                QSlider::groove:horizontal { height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; }
                QSlider::sub-page:horizontal { background: #0A84FF; border-radius: 2px; }
                QSlider::handle:horizontal { background: #FFFFFF; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; border: none; }
                QSlider::handle:horizontal:hover { width: 16px; height: 16px; margin: -6px -1px; border-radius: 8px; }
            """)
            val_lbl = QLabel(f"{s_val}{suffix}")
            val_lbl.setFixedWidth(35)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet("color: #8E8E93; font-size: 12px; background: transparent;")
            slider.valueChanged.connect(lambda v: val_lbl.setText(f"{v}{suffix}"))
            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(val_lbl)
            return row, slider, val_lbl

        brush_row, self.vid_brush_slider, self.vid_brush_label = create_minimal_slider_row("粗细", 1, 100, 15, "px")
        morph_row, self.vid_morph_slider, self.vid_morph_label = create_minimal_slider_row("缩放", -20, 20, 0, "px")

        right_layout.addLayout(brush_row)
        right_layout.addLayout(morph_row)

        right_layout.addSpacing(8)

        # ==================== 3. 发丝时序渲染配置 ====================
        right_layout.addWidget(create_section_title("发丝渲染配置"))

        chk_container = QWidget()
        chk_container.setStyleSheet("background: transparent;")
        chk_layout = QVBoxLayout(chk_container)
        chk_layout.setContentsMargins(4, 4, 4, 4)
        chk_layout.setSpacing(10)

        # 唯一的发丝精雕渲染开关，未勾选走标准 SAM2（硬边），勾选走 MatAnyone2 时序精雕
        self.vid_matteformer_checkbox = QCheckBox("激活发丝级时序精雕")
        self.vid_matteformer_checkbox.setToolTip(
            "开启后，系统将在首帧利用您的标注提取高精度发丝，并驱动 MatAnyone2 引擎进行全局平滑发丝级渲染；若不开启，则只运行标准 SAM2 粗追踪（适合刚性无发丝主体）。"
        )
        self.vid_matteformer_checkbox.setStyleSheet("""
            QCheckBox { color: #FFF; font-size: 13px; font-weight: bold; background: transparent;} 
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; background: rgba(255,255,255,0.05); border: 1px solid #555; } 
            QCheckBox::indicator:checked { background: #0A84FF; border: none; }
            QCheckBox:disabled { color: #555558; }
        """)
        self.vid_matteformer_checkbox.setEnabled(getattr(self, 'matteformer_loaded', False))
        chk_layout.addWidget(self.vid_matteformer_checkbox)
        right_layout.addWidget(chk_container)

        # 为 MatAnyone2 预备的精细腐蚀与膨胀配置，不可删除
        erode_row, self.vid_erode_slider, self.vid_erode_label = create_minimal_slider_row("腐蚀", 0, 30, 10)
        dilate_row, self.vid_dilate_slider, self.vid_dilate_label = create_minimal_slider_row("膨胀", 0, 30, 10)
        right_layout.addLayout(erode_row)
        right_layout.addLayout(dilate_row)

        right_layout.addSpacing(4)

        # ==================== 4. 视频背景调整 ====================
        right_layout.addWidget(create_section_title("渲染背景色"))

        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(4, 4, 4, 0)

        self.start_video_seg_button = QToolButton()
        if hasattr(self, '_create_svg_icon'):
            self.start_video_seg_button.setIcon(self._create_svg_icon("magic.svg", size=14, color="#FFFFFF"))
        self.start_video_seg_button.setStyleSheet("""
            QToolButton { background-color: #2C2C2E; border-radius: 6px; border: 1px solid #3A3A3C; padding: 4px; } 
            QToolButton:hover { background-color: #3A3A3C; border: 1px solid #555; }
        """)
        self.start_video_seg_button.setFixedSize(26, 26)
        self.start_video_seg_button.setToolTip("开始智能渲染")
        if hasattr(self, 'start_video_segmentation_propagation'):
            self.start_video_seg_button.clicked.connect(self.start_video_segmentation_propagation)

        bg_label = QLabel("背景:")
        bg_label.setStyleSheet("color: #8E8E93; font-size: 12px; background: transparent;")

        self.vid_bg_transparent_btn = QToolButton()
        if hasattr(self, '_create_svg_icon'):
            self.vid_bg_transparent_btn.setIcon(
                self._create_svg_icon("grid-3x3-gap-fill.svg", size=14, color="#FFFFFF"))
        self.vid_bg_transparent_btn.setStyleSheet(
            "QToolButton { background-color: rgba(255,255,255,0.08); border-radius: 9px; padding: 2px; } QToolButton:hover { background-color: rgba(255,255,255,0.2); }")
        self.vid_bg_transparent_btn.setToolTip("透明背景 (GIF合并支持)")

        self.vid_bg_color_btn = QPushButton()
        self.vid_bg_color_btn.setFixedSize(18, 18)
        self.video_save_bg_color = QColor(0, 0, 0)
        self.vid_bg_color_btn.setStyleSheet("background-color: #000000; border-radius: 9px; border: 1px solid #555;")

        self.vid_bg_image_btn = QToolButton()
        if hasattr(self, '_create_svg_icon'):
            self.vid_bg_image_btn.setIcon(self._create_svg_icon("image.svg", size=14, color="#FFFFFF"))
        self.vid_bg_image_btn.setStyleSheet(
            "QToolButton { background-color: rgba(255,255,255,0.08); border-radius: 9px; padding: 2px; } QToolButton:hover { background-color: rgba(255,255,255,0.2); }")

        self.vid_bg_preview_label = QLabel()
        self.vid_bg_preview_label.setFixedSize(24, 18)
        self.vid_bg_preview_label.setStyleSheet("border-radius: 4px; border: 1px solid #444; background: #2A2A2A;")
        self.vid_bg_preview_label.hide()
        self.vid_bg_preview_label.setCursor(Qt.CursorShape.PointingHandCursor)

        def on_preview_label_clicked(event):
            if event.button() == Qt.MouseButton.LeftButton:
                # 【核心修复】：由于支持了分段独立背景，必须从当前活跃的片段(clip)配置中获取对应的自定义图片路径
                clip_idx = self._get_active_editing_clip_idx()
                if clip_idx != -1 and hasattr(self, 'virtual_timeline') and clip_idx < len(self.virtual_timeline):
                    clip = self.virtual_timeline[clip_idx]
                    bg_image_path = clip.get('bg_image_path', None)

                    if bg_image_path and os.path.exists(bg_image_path):
                        if hasattr(self, '_bg_full_preview_overlay') and self._bg_full_preview_overlay:
                            # 传入真实存在的片段背景图片路径，并指定 vid_bg_preview_label 作为动画射起点
                            self._bg_full_preview_overlay.show_image(bg_image_path, self.vid_bg_preview_label)

            super(QLabel, self.vid_bg_preview_label).mousePressEvent(event)

        self.vid_bg_preview_label.mousePressEvent = on_preview_label_clicked

        # 关联相关槽函数
        self.vid_bg_transparent_btn.clicked.connect(lambda: self.set_video_bg_mode("transparent"))
        self.vid_bg_color_btn.clicked.connect(self._select_video_bg_color)
        self.vid_bg_image_btn.clicked.connect(lambda: self.set_video_bg_mode("image"))

        bottom_row.addWidget(self.start_video_seg_button)
        bottom_row.addStretch()
        bottom_row.addWidget(bg_label)
        bottom_row.addWidget(self.vid_bg_transparent_btn)
        bottom_row.addWidget(self.vid_bg_color_btn)
        bottom_row.addWidget(self.vid_bg_image_btn)
        bottom_row.addWidget(self.vid_bg_preview_label)
        right_layout.addLayout(bottom_row, 0)

        # ------------------- 单片段原声静音配置 -------------------
        self.matting_mute_checkbox = QCheckBox("静音当前视频片段")
        self.matting_mute_checkbox.setStyleSheet("""
            QCheckBox { color: #8E8E93; font-size: 13px; background: transparent; margin-top: 5px;} 
            QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; background: rgba(255,255,255,0.05); border: 1px solid #555; } 
            QCheckBox::indicator:checked { background: #EF4444; border: none; }
        """)
        self.matting_mute_checkbox.toggled.connect(lambda c: self.set_clip_mute_state(c))
        right_layout.addWidget(self.matting_mute_checkbox, 0)

        right_layout.addStretch(1)

        # ==================== 5. 底部应用/取消操作 ====================
        bottom_actions_layout = QHBoxLayout()
        bottom_actions_layout.setContentsMargins(0, 0, 0, 0)
        bottom_actions_layout.setSpacing(10)

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(UNIFORM_BTN_STYLE)
        cancel_btn.clicked.connect(self._discard_matting_and_exit)

        apply_btn = QPushButton("确定")
        apply_btn.setStyleSheet(UNIFORM_BTN_STYLE)
        apply_btn.clicked.connect(self._apply_matting_and_exit)

        bottom_actions_layout.addWidget(cancel_btn, 1)
        bottom_actions_layout.addWidget(apply_btn, 1)

        right_layout.addLayout(bottom_actions_layout, 0)
        main_layout.addWidget(right_panel, 0)

        # 绑定事件
        self._bind_dedicated_matting_events()

    def _bind_dedicated_matting_events(self):
        """
        完整版事件绑定：包含标注模式切换、画笔、局部缩放以及
        MatAnyone 专用的腐蚀/膨胀参数滑块信号关联。
        """
        def update_vid_interaction_mode():
            mode = 'brush' if self.vid_mode_brush_btn.isChecked() else 'point'
            if hasattr(self, 'video_display_label'):
                self.video_display_label.set_interaction_mode(mode)

        self.vid_mode_point_btn.toggled.connect(update_vid_interaction_mode)
        self.vid_mode_brush_btn.toggled.connect(update_vid_interaction_mode)

        self.vid_brush_slider.valueChanged.connect(lambda v: self.vid_brush_label.setText(f"{v} px"))
        self.vid_brush_slider.valueChanged.connect(
            lambda v: getattr(self, 'video_display_label').set_brush_size(v) if hasattr(self, 'video_display_label') else None
        )

        # 3. 绑定蒙版局部缩放滑块（带松开自复位）
        def _on_morph_released():
            if hasattr(self, 'video_display_label'):
                self.video_display_label.end_mask_shift()

            # 【核心修复】：拖拽结束后复位滑块至 0 准备下一次调节。
            # 必须使用 blockSignals 拦截信号，防止触发 valueChanged(0) 导致刚刚确认的蒙版被抹除！
            self.vid_morph_slider.blockSignals(True)
            self.vid_morph_slider.setValue(0)
            self.vid_morph_label.setText("0 px")
            self.vid_morph_slider.blockSignals(False)

        self.vid_morph_slider.sliderPressed.connect(
            lambda: getattr(self, 'video_display_label').begin_mask_shift() if hasattr(self,
                                                                                       'video_display_label') else None
        )
        self.vid_morph_slider.valueChanged.connect(lambda v: self.vid_morph_label.setText(f"{v} px"))
        self.vid_morph_slider.valueChanged.connect(
            lambda v: getattr(self, 'video_display_label').apply_mask_shift(v) if hasattr(self,
                                                                                          'video_display_label') else None
        )
        self.vid_morph_slider.sliderReleased.connect(_on_morph_released)
        # ----------------------------------------------

        # 4. 绑定 MatAnyone2 专用的腐蚀与膨胀数值指示器信号（不可或缺）
        self.vid_erode_slider.valueChanged.connect(lambda v: self.vid_erode_label.setText(str(v)))
        self.vid_dilate_slider.valueChanged.connect(lambda v: self.vid_dilate_label.setText(str(v)))

    def _init_vid_dedicated_crop_page(self):
        """
        Initialize dedicated crop and audio debug page.
        Refactored: Added independent preview play button (btn_bgm_play_preview) on the left side of BGM timeline header, supporting independent preview of music trim interval.
        """
        self.vid_dedicated_crop_page = QWidget()
        self.vid_dedicated_crop_page.setObjectName("VidCropPage")
        self.vid_dedicated_crop_page.setStyleSheet(f"QWidget#VidCropPage {{ background-color: {VID_DARK_BG}; }}")

        main_layout = QHBoxLayout(self.vid_dedicated_crop_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left_area = QWidget()
        left_layout = QVBoxLayout(left_area)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # 1. Header page switch bar
        switcher_container = QWidget()
        switcher_container.setFixedHeight(54)
        switcher_layout = QHBoxLayout(switcher_container)
        switcher_layout.setContentsMargins(0, 10, 0, 0)
        switcher_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        tools = [
            ("剪裁", "scissors.svg", None),
            ("智能抠图", "person-bounding-box.svg",
             lambda: (self._apply_crop_and_exit(), self._enter_dedicated_matting_mode())),
        ]

        switcher_group = QWidget()
        switcher_group.setStyleSheet("background-color: transparent;")
        sg_layout = QHBoxLayout(switcher_group)
        sg_layout.setContentsMargins(0, 0, 0, 0)
        sg_layout.setSpacing(6)

        for text, icon, func in tools:
            btn = QToolButton()
            btn.setText(text)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            if hasattr(self, '_create_svg_icon'):
                btn.setIcon(self._create_svg_icon(icon, color=VID_TEXT_PRIMARY))

            if text == "剪裁":
                btn.setStyleSheet(
                    f"QToolButton {{ background-color: {VID_ACCENT}; color: #FFFFFF; border-radius: 8px; padding: 6px 14px; font-size: 13px; font-weight: bold; border: none; }}")
            else:
                btn.setStyleSheet(
                    f"QToolButton {{ background-color: transparent; color: {VID_TEXT_PRIMARY}; border-radius: 8px; padding: 6px 14px; font-size: 13px; border: none; }} QToolButton:hover {{ background-color: #333333; }}")
                if func:
                    btn.clicked.connect(func)
            sg_layout.addWidget(btn)

        switcher_layout.addWidget(switcher_group)
        left_layout.addWidget(switcher_container, 0)

        # 2. Crop main screen area
        self.crop_workspace_wrapper = QWidget()
        self.crop_workspace_wrapper.setStyleSheet("background: transparent;")
        left_layout.addWidget(self.crop_workspace_wrapper, 1)

        main_layout.addWidget(left_area, 1)

        # 3. Right property control panel
        right_panel = QWidget()
        right_panel.setFixedWidth(320)
        right_panel.setStyleSheet("background-color: #1C1C1E; border: none;")

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 0, 16, 24)
        right_layout.setSpacing(16)

        rp_title_container = QWidget()
        rp_title_container.setFixedHeight(44)
        rp_title_layout = QHBoxLayout(rp_title_container)
        rp_title_layout.setContentsMargins(0, 0, 0, 0)
        rp_title = QLabel("剪裁与音频")
        rp_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rp_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #FFFFFF;")
        rp_title_layout.addWidget(rp_title)
        right_layout.addWidget(rp_title_container, 0)

        def create_section_title(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size: 12px; color: #8E8E93; font-weight: bold; margin-bottom: 2px;")
            return lbl

        UNIFORM_BTN_STYLE = """
            QPushButton { 
                background-color: #2C2C2E; 
                color: #FFFFFF; 
                border-radius: 6px; 
                padding: 10px 12px; 
                font-weight: bold; 
                font-size: 13px; 
                border: 1px solid #3A3A3C; 
            } 
            QPushButton:hover { 
                background-color: #3A3A3C; 
                border: 1px solid #555555;
            }
            QPushButton:pressed {
                background-color: #1C1C1E;
            }
            QPushButton:disabled {
                color: #555558;
                background-color: #1C1C1E;
                border: 1px solid #2C2C2E;
            }
        """

        # Module A: Crop info
        right_layout.addWidget(create_section_title("剪辑时间 info"))

        info_container = QWidget()
        info_container.setStyleSheet("background-color: rgba(255, 255, 255, 0.03); border-radius: 8px;")
        info_layout = QVBoxLayout(info_container)
        info_layout.setContentsMargins(14, 14, 14, 14)

        len_layout = QHBoxLayout()
        len_layout.addWidget(
            QLabel("剪辑长度",
                   styleSheet="color: #8E8E93; font-size: 13px; font-weight: bold; background: transparent;"))
        self.crop_length_label = QLabel("0:00.00")
        self.crop_length_label.setStyleSheet(
            "color: #FFFFFF; font-size: 14px; font-weight: bold; background: transparent;")
        self.crop_length_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        len_layout.addWidget(self.crop_length_label)

        info_layout.addLayout(len_layout)
        right_layout.addWidget(info_container, 0)

        # Module B: Original audio track
        right_layout.addWidget(create_section_title("原声音轨"))

        self.btn_toggle_mute = QPushButton("关闭原声")
        self.btn_toggle_mute.setCheckable(True)
        self.btn_toggle_mute.setStyleSheet(UNIFORM_BTN_STYLE)
        self.btn_toggle_mute.clicked.connect(self._on_audio_mute_toggled)
        right_layout.addWidget(self.btn_toggle_mute, 0)

        def create_minimal_slider_row(name, s_min, s_max, s_val, suffix=""):
            row = QHBoxLayout()
            row.setContentsMargins(4, 0, 4, 0)
            lbl = QLabel(name)
            lbl.setFixedWidth(50)
            lbl.setStyleSheet("color: #FFFFFF; font-size: 13px; background: transparent;")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(s_min, s_max)
            slider.setValue(s_val)
            slider.setStyleSheet("""
                QSlider { min-height: 20px; background: transparent; }
                QSlider::groove:horizontal { height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; }
                QSlider::sub-page:horizontal { background: #0A84FF; border-radius: 2px; }
                QSlider::handle:horizontal { background: #FFFFFF; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; border: none; }
                QSlider::handle:horizontal:hover { width: 16px; height: 16px; margin: -6px -1px; border-radius: 8px; }
            """)
            val_lbl = QLabel(f"{s_val}{suffix}")
            val_lbl.setFixedWidth(35)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet("color: #8E8E93; font-size: 12px; background: transparent;")
            slider.valueChanged.connect(lambda v: val_lbl.setText(f"{v}{suffix}"))
            row.addWidget(lbl)
            row.addWidget(slider, 1)
            row.addWidget(val_lbl)
            return row, slider, val_lbl

        orig_vol_row, self.orig_volume_slider, self.orig_vol_val_lbl = create_minimal_slider_row("原音量", 0, 100,
                                                                                                 100,
                                                                                                 "%")
        self.orig_volume_slider.valueChanged.connect(self._on_orig_volume_changed)
        right_layout.addLayout(orig_vol_row, 0)

        # Module C: Background Music (BGM)
        right_layout.addWidget(create_section_title("背景音乐 (BGM)"))

        self.bgm_info_label = QLabel("无背景音乐")
        self.bgm_info_label.setStyleSheet(
            "color: #8E8E93; font-size: 12px; font-style: italic; background: transparent; padding: 2px 4px;")
        right_layout.addWidget(self.bgm_info_label, 0)

        self.btn_add_bgm = QPushButton("添加背景音乐")
        self.btn_add_bgm.setStyleSheet(UNIFORM_BTN_STYLE)
        self.btn_add_bgm.clicked.connect(self._on_add_bgm_clicked)
        right_layout.addWidget(self.btn_add_bgm, 0)

        self.btn_remove_bgm = QPushButton("移除背景音乐")
        # Styling modification: fully matches the unified neutral dark gray buttons to avoid intrusive red highlights
        self.btn_remove_bgm.setStyleSheet(UNIFORM_BTN_STYLE)
        self.btn_remove_bgm.clicked.connect(self._on_remove_bgm_clicked)
        self.btn_remove_bgm.hide()
        right_layout.addWidget(self.btn_remove_bgm, 0)

        # BGM double slider clipping control axis
        self.bgm_clip_start_container = QWidget()
        self.bgm_clip_start_container.setStyleSheet("background: transparent;")
        bgm_clip_start_layout = QVBoxLayout(self.bgm_clip_start_container)
        bgm_clip_start_layout.setContentsMargins(4, 0, 4, 0)
        bgm_clip_start_layout.setSpacing(4)

        bgm_slider_hdr = QHBoxLayout()

        # [Added]: Independent preview BGM play/pause mini button
        self.btn_bgm_play_preview = QToolButton()
        self.btn_bgm_play_preview.setFixedSize(22, 22)
        self.btn_bgm_play_preview.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_bgm_play_preview.setStyleSheet(VID_MODERN_BTN_STYLE + "QToolButton { padding: 2px; }")
        self.btn_bgm_play_preview.clicked.connect(self._on_bgm_preview_play_clicked)
        self._update_bgm_preview_button_ui()

        bgm_slider_lbl = QLabel("音乐剪切区间:")
        bgm_slider_lbl.setStyleSheet("color: #FFFFFF; font-size: 13px; background: transparent;")
        self.bgm_clip_time_label = QLabel("0:00 - 0:00")
        self.bgm_clip_time_label.setStyleSheet("color: #8E8E93; font-size: 12px; background: transparent;")
        self.bgm_clip_time_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        bgm_slider_hdr.addWidget(self.btn_bgm_play_preview)
        bgm_slider_hdr.addWidget(bgm_slider_lbl)
        bgm_slider_hdr.addStretch()
        bgm_slider_hdr.addWidget(self.bgm_clip_time_label)
        bgm_clip_start_layout.addLayout(bgm_slider_hdr)

        self.bgm_trim_slider = VideoTrimSlider()
        self.bgm_trim_slider.range_changed.connect(self._on_bgm_trim_range_changed)
        bgm_clip_start_layout.addWidget(self.bgm_trim_slider)

        self.bgm_clip_start_container.hide()
        right_layout.addWidget(self.bgm_clip_start_container, 0)

        # Addition moment in video (at which second of the video this music is added)
        self.bgm_start_container = QWidget()
        self.bgm_start_container.setStyleSheet("background: transparent;")
        bgm_start_layout = QHBoxLayout(self.bgm_start_container)
        bgm_start_layout.setContentsMargins(4, 0, 4, 0)

        self.bgm_start_label = QLabel("视频加入时刻:")
        self.bgm_start_label.setStyleSheet("color: #FFFFFF; font-size: 13px; background: transparent;")

        self.bgm_start_spinbox = QDoubleSpinBox()
        self.bgm_start_spinbox.setRange(0.0, 9999.0)
        self.bgm_start_spinbox.setDecimals(1)
        self.bgm_start_spinbox.setSingleStep(1.0)
        self.bgm_start_spinbox.setSuffix(" 秒")
        self.bgm_start_spinbox.setFixedWidth(100)
        self.bgm_start_spinbox.setStyleSheet("""
            QDoubleSpinBox { 
                background-color: #2C2C2E; 
                color: #FFFFFF; 
                border: 1px solid #3A3A3C; 
                border-radius: 4px; 
                padding: 4px; 
            }
            QDoubleSpinBox:focus { 
                border: 1px solid #0A84FF; 
            }
        """)
        self.bgm_start_spinbox.valueChanged.connect(self._on_bgm_start_changed)

        bgm_start_layout.addWidget(self.bgm_start_label)
        bgm_start_layout.addStretch()
        bgm_start_layout.addWidget(self.bgm_start_spinbox)
        self.bgm_start_container.hide()
        right_layout.addWidget(self.bgm_start_container, 0)

        # BGM volume slider
        self.bgm_vol_row_layout, self.bgm_volume_slider, self.bgm_vol_val_lbl = create_minimal_slider_row("乐音量",
                                                                                                          0,
                                                                                                          100, 100,
                                                                                                          "%")
        self.bgm_volume_slider.valueChanged.connect(self._on_bgm_volume_changed)

        self.bgm_vol_container = QWidget()
        self.bgm_vol_container.setStyleSheet("background: transparent;")
        self.bgm_vol_container.setLayout(self.bgm_vol_row_layout)
        self.bgm_vol_container.hide()
        right_layout.addWidget(self.bgm_vol_container, 0)

        right_layout.addStretch(1)

        # Module D: Bottom action buttons
        bottom_actions = QWidget()
        bottom_actions.setStyleSheet(f"border-top: 1px solid #2A2A2A; background-color: #1A1A1A;")
        bottom_actions_layout = QHBoxLayout(bottom_actions)
        bottom_actions_layout.setContentsMargins(16, 16, 16, 16)
        bottom_actions_layout.setSpacing(12)

        apply_btn = QPushButton("完成")
        apply_btn.setStyleSheet(
            f"QPushButton {{ background-color: {VID_ACCENT}; color: #FFF; border-radius: 8px; padding: 12px; font-weight: bold; font-size: 14px; border: none; }} QPushButton:hover {{ background-color: #1B66C9; }}")
        apply_btn.clicked.connect(self._apply_crop_and_exit)

        cancel_btn = QPushButton("取消")
        cancel_btn.setStyleSheet(
            "QPushButton { background-color: #333333; color: #CCC; border-radius: 8px; padding: 12px; border: 1px solid #444; font-size: 14px; font-weight: bold;} QPushButton:hover { background-color: #383838; }")
        cancel_btn.clicked.connect(self._exit_dedicated_crop_mode)

        bottom_actions_layout.addWidget(cancel_btn, 1)
        bottom_actions_layout.addWidget(apply_btn, 1)

        right_layout.addWidget(bottom_actions, 0)
        main_layout.addWidget(right_panel, 0)

    def _mount_crop_player_to_workspace(self, workspace_widget):
        if workspace_widget.layout() is not None:
            while workspace_widget.layout().count():
                item = workspace_widget.layout().takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
        else:
            layout = QVBoxLayout(workspace_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(16)

        if hasattr(self, 'video_margin_wrapper'):
            self.video_margin_wrapper.layout().setContentsMargins(60, 40, 60, 0)
            self.video_margin_wrapper.setParent(workspace_widget)
            self.video_margin_wrapper.show()
            workspace_widget.layout().addWidget(self.video_margin_wrapper, 1)

        if not hasattr(self, 'crop_playback_bar_wrapper'):
            self.crop_playback_bar_wrapper = QWidget()
            self.crop_playback_bar_wrapper.setFixedHeight(62)

            pb_wrapper_layout = QHBoxLayout(self.crop_playback_bar_wrapper)
            pb_wrapper_layout.setContentsMargins(16, 0, 16, 16)

            self.crop_playback_bar = QWidget()
            self.crop_playback_bar.setFixedHeight(46)
            self.crop_playback_bar.setStyleSheet("background-color: #262626; border-radius: 6px;")

            pb_layout = QHBoxLayout(self.crop_playback_bar)
            pb_layout.setContentsMargins(16, 0, 16, 0)
            pb_layout.setSpacing(12)

            btn_prev = QToolButton()
            if hasattr(self, '_create_svg_icon'): btn_prev.setIcon(
                self._create_svg_icon("caret-left-fill.svg", color=VID_TEXT_PRIMARY))
            btn_prev.setStyleSheet(VID_MODERN_BTN_STYLE)

            crop_play_btn = QToolButton()
            if hasattr(self, '_create_svg_icon'):
                crop_play_btn.setIcon(self._create_svg_icon("play-fill.svg", color=VID_TEXT_PRIMARY))
            crop_play_btn.setStyleSheet(VID_MODERN_BTN_STYLE)

            btn_next = QToolButton()
            if hasattr(self, '_create_svg_icon'): btn_next.setIcon(
                self._create_svg_icon("caret-right-fill.svg", color=VID_TEXT_PRIMARY))
            btn_next.setStyleSheet(VID_MODERN_BTN_STYLE)

            self.btn_crop_global_mute = QToolButton()
            if hasattr(self, '_create_svg_icon'):
                icon_name = "volume-mute-fill.svg" if getattr(self, '_is_global_muted',
                                                              False) else "volume-up-fill.svg"
                self.btn_crop_global_mute.setIcon(self._create_svg_icon(icon_name, color=VID_TEXT_PRIMARY))
            self.btn_crop_global_mute.setStyleSheet(VID_MODERN_BTN_STYLE)
            self.btn_crop_global_mute.clicked.connect(self._toggle_global_mute)

            self.crop_time_curr = QLabel("0:00.00")
            self.crop_time_curr.setStyleSheet("color:#FFF; font-size:12px;")

            self.crop_time_total = QLabel("0:00.00")
            self.crop_time_total.setStyleSheet("color:#FFF; font-size:12px;")

            self.video_trim_slider = VideoTrimSlider()
            self.video_trim_slider.range_changed.connect(self._on_trim_range_changed)
            #self.video_trim_slider.preview_frame.connect(self._preview_raw_frame)

            pb_layout.addWidget(btn_prev)
            pb_layout.addWidget(crop_play_btn)
            pb_layout.addWidget(btn_next)
            pb_layout.addWidget(self.btn_crop_global_mute)
            pb_layout.addSpacing(8)
            pb_layout.addWidget(self.crop_time_curr)
            pb_layout.addWidget(self.video_trim_slider, 1)
            pb_layout.addWidget(self.crop_time_total)

            pb_wrapper_layout.addWidget(self.crop_playback_bar)

        self.crop_playback_bar_wrapper.setParent(workspace_widget)
        self.crop_playback_bar_wrapper.show()
        workspace_widget.layout().addWidget(self.crop_playback_bar_wrapper, 0)

    def _get_current_clip_info(self):
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline: return None, 0

        current_count = 0
        global_idx = getattr(self, 'current_frame_index', 0)

        for idx, vid in enumerate(self.virtual_timeline):
            if global_idx < current_count + vid['frames']:
                local_frame = global_idx - current_count
                return idx, local_frame
            current_count += vid['frames']

        if len(self.virtual_timeline) > 0: return len(self.virtual_timeline) - 1, self.virtual_timeline[-1][
            'frames'] - 1
        return None, 0

    def _stop_async_scrub_reader(self):
        """
        安全停止并销毁异步寻址解码器及其对应的后台线程。
        保证 native PyAV 容器锁在对应的子线程中被同步释放，消除 SegFault 隐患。
        """
        if hasattr(self, '_async_scrub_reader') and self._async_scrub_reader is not None:
            reader = self._async_scrub_reader
            thread = getattr(self, '_scrub_thread', None)

            try:
                # 1. 如果子线程正在运行，跨线程阻断式调用其 close 方法，确保在子线程中完成释放
                if thread and thread.isRunning():
                    QMetaObject.invokeMethod(reader, "close", Qt.ConnectionType.BlockingQueuedConnection)
                else:
                    reader.close()
            except Exception as e:
                print(f"[DEBUG] Error invoking thread-safe close on scrub reader: {e}")

            reader.deleteLater()
            self._async_scrub_reader = None

        if hasattr(self, '_scrub_thread') and self._scrub_thread is not None:
            if self._scrub_thread.isRunning():
                self._scrub_thread.quit()
                # 2. 等待最多 1000 毫秒让子线程安全退出，超时则进行保护性终止
                if not self._scrub_thread.wait(1000):
                    print("[DEBUG] Scrub thread did not exit in time. Forcing termination.")
                    self._scrub_thread.terminate()
                    self._scrub_thread.wait()
            self._scrub_thread.deleteLater()
            self._scrub_thread = None

    @Slot(object, int)
    def _on_async_frame_decoded(self, frame_bgr, frame_index):
        """
        [重构版]：异步解码帧在剪裁界面的高精度渲染回调
        """
        if getattr(self, '_current_crop_clip_idx', -1) == -1:
            return

        try:
            if getattr(self, 'video_width', 0) > 0 and getattr(self, 'video_height', 0) > 0:
                if frame_bgr.shape[:2] != (self.video_height, self.video_width):
                    frame_bgr = cv2.resize(frame_bgr, (self.video_width, self.video_height),
                                           interpolation=cv2.INTER_LANCZOS4)

            clip_idx = self._current_crop_clip_idx
            clip = self.virtual_timeline[clip_idx]
            local_masks = clip.get('local_masks', {})

            # =========================================================================
            # 【高精度重构区】：实施连续域浮点 Alpha 融合算法
            # =========================================================================
            if frame_index in local_masks and local_masks[frame_index]:
                h, w = frame_bgr.shape[:2]
                solid_bg = self._get_current_bg_frame(h, w)
                combined_alpha = np.zeros((h, w), dtype=np.float32)
                has_mask = False

                for mask_raw in local_masks[frame_index].values():
                    if mask_raw is not None:
                        if mask_raw.dtype == bool:
                            mask_float = mask_raw.astype(np.float32)
                        else:
                            mask_float = np.clip(mask_raw.astype(np.float32), 0.0, 1.0)

                        if mask_float.shape != (h, w):
                            mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)

                        combined_alpha = np.maximum(combined_alpha, mask_float)
                        has_mask = True

                if has_mask:
                    alpha_3d = combined_alpha[:, :, np.newaxis]
                    blended = frame_bgr.astype(np.float32) * alpha_3d + solid_bg.astype(np.float32) * (1.0 - alpha_3d)
                    frame_bgr = np.clip(blended, 0.0, 255.0).astype(np.uint8)

            if hasattr(self, 'video_display_label'):
                self.video_display_label.set_frame(frame_bgr, frame_index)
                self.video_display_label.repaint()

        except Exception as e:
            print(f"Async frame render error: {e}")

    @Slot()
    def _enter_dedicated_crop_mode(self):
        """[Async High-Speed Version] Enter crop mode"""
        if getattr(self, 'is_extracting_frames', False):
            self.show_status_message("视频序列正在后台同步中，完成后将自动进入剪辑...", 3000)
            self._pending_mode_switch = "crop"
            return

        if getattr(self, 'is_playing', False): self.pause_video()
        if hasattr(self, 'top_bar_main'): self.top_bar_main.hide()

        clip_idx = -1
        if hasattr(self, 'storyboard_list') and self.storyboard_list.selectedItems():
            clip_idx = self.storyboard_list.row(self.storyboard_list.selectedItems()[0])
        else:
            clip_idx, _ = self._get_current_clip_info()

        if clip_idx is None or clip_idx < 0:
            if hasattr(self, 'virtual_timeline') and len(self.virtual_timeline) > 0:
                clip_idx = 0
            else:
                QMessageBox.information(self, "提示", "请先在主页添加视频片段。")
                if hasattr(self, 'top_bar_main'): self.top_bar_main.show()
                return

        self._distribute_global_masks_to_clips()
        self._current_crop_clip_idx = clip_idx
        clip = self.virtual_timeline[clip_idx]

        cap = cv2.VideoCapture(clip['path'])
        actual_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        clip['total_file_frames'] = actual_total_frames

        total_f = max(1, clip['total_file_frames'])
        in_pt = int(clip.get('in_point', 0))
        out_pt = int(clip.get('out_point', total_f))

        self._crop_clip_fps = clip.get('fps', 30.0)
        self._crop_clip_path = clip['path']

        self._mount_crop_player_to_workspace(self.crop_workspace_wrapper)
        self.vid_editor_stack.setCurrentWidget(self.vid_dedicated_crop_page)

        # =========================================================================
        # [Core thread allocation]: Start and migrate decode tasks to independent background sub-threads
        # =========================================================================
        self._stop_async_scrub_reader()

        self._scrub_thread = QThread(self)
        self._async_scrub_reader = AsyncScrubReader(clip['path'])
        self._async_scrub_reader.moveToThread(self._scrub_thread)
        self._async_scrub_reader.frame_decoded.connect(self._on_async_frame_decoded)
        self._scrub_thread.start()

        if hasattr(self, 'video_trim_slider'):
            self.video_trim_slider.blockSignals(True)
            self.video_trim_slider.set_range(0, total_f)
            self.video_trim_slider.set_values(in_pt, out_pt)

            # Disconnect original synchronous connection and redirect to non-blocking high-speed async interface
            try:
                self.video_trim_slider.preview_frame.disconnect()
            except:
                pass
            self.video_trim_slider.preview_frame.connect(
                lambda f_idx: self._async_scrub_reader.request_frame(f_idx) if getattr(self, '_async_scrub_reader',
                                                                                       None) else None
            )
            self.video_trim_slider.blockSignals(False)

        if hasattr(self, 'crop_time_total'):
            self.crop_time_total.setText(self._format_time(total_f / max(1.0, self._crop_clip_fps)))

        self._on_trim_range_changed(in_pt, out_pt)

        if hasattr(self, 'video_display_label'): self.video_display_label.clear_display()

        # Initial first frame decode
        self._async_scrub_reader.request_frame(in_pt)
        self._update_audio_console_ui()

    @Slot(int, int)
    def _on_trim_range_changed(self, start, end):
        duration = (end - start) / self._crop_clip_fps
        if hasattr(self, 'crop_length_label'): self.crop_length_label.setText(self._format_time(duration))
        if hasattr(self, 'crop_time_curr'): self.crop_time_curr.setText(
            self._format_time(start / self._crop_clip_fps))

    @Slot(int)
    def _preview_raw_frame(self, frame_index):
        """
        [重构版]：剪裁专属逐帧渲染预览
        """
        path = getattr(self, '_crop_clip_path', None)
        clip_idx = getattr(self, '_current_crop_clip_idx', -1)
        if not path or clip_idx == -1: return

        try:
            if not hasattr(self, '_crop_pyav_container') or self._crop_pyav_container is None:
                self._crop_pyav_container = av.open(path)
                self._crop_pyav_stream = self._crop_pyav_container.streams.video[0]

            container = self._crop_pyav_container
            stream = self._crop_pyav_stream

            fps = stream.average_rate
            if not fps:
                fps = stream.r_frame_rate
            fps = float(fps) if fps else 30.0

            time_base = float(stream.time_base)
            frame_duration_pts = 1.0 / (fps * time_base)

            seek_frame = max(0, frame_index - 5)
            seek_pts = int(seek_frame * frame_duration_pts)

            try:
                container.seek(seek_pts, stream=stream)
            except Exception:
                container.seek(0, stream=stream)

            frame = None
            for av_frame in container.decode(video=0):
                current_idx = int(round(av_frame.pts * time_base * fps))
                if current_idx == frame_index:
                    frame = av_frame.to_ndarray(format='bgr24')
                    break
                if current_idx > frame_index:
                    break

            if frame is None:
                container.seek(0, stream=stream)
                for av_frame in container.decode(video=0):
                    current_idx = int(round(av_frame.pts * time_base * fps))
                    if current_idx == frame_index:
                        frame = av_frame.to_ndarray(format='bgr24')
                        break
                    if current_idx > frame_index:
                        break

            # =========================================================================
            # 【高精度重构区】：实施连续域浮点 Alpha 融合算法
            # =========================================================================
            if frame is not None:
                if getattr(self, 'video_width', 0) > 0 and getattr(self, 'video_height', 0) > 0:
                    if frame.shape[:2] != (self.video_height, self.video_width):
                        frame = cv2.resize(frame, (self.video_width, self.video_height),
                                           interpolation=cv2.INTER_LANCZOS4)

                clip = self.virtual_timeline[clip_idx]
                local_masks = clip.get('local_masks', {})
                if frame_index in local_masks and local_masks[frame_index]:
                    h, w = frame.shape[:2]
                    solid_bg = self._get_current_bg_frame(h, w)
                    combined_alpha = np.zeros((h, w), dtype=np.float32)
                    has_mask = False

                    for mask_raw in local_masks[frame_index].values():
                        if mask_raw is not None:
                            if mask_raw.dtype == bool:
                                mask_float = mask_raw.astype(np.float32)
                            else:
                                mask_float = np.clip(mask_raw.astype(np.float32), 0.0, 1.0)

                            if mask_float.shape != (h, w):
                                mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)

                            combined_alpha = np.maximum(combined_alpha, mask_float)
                            has_mask = True

                    if has_mask:
                        alpha_3d = combined_alpha[:, :, np.newaxis]
                        blended = frame.astype(np.float32) * alpha_3d + solid_bg.astype(np.float32) * (1.0 - alpha_3d)
                        frame = np.clip(blended, 0.0, 255.0).astype(np.uint8)

                if hasattr(self, 'video_display_label'):
                    self.video_display_label.set_frame(frame, frame_index)
                    self.video_display_label.repaint()

        except Exception as err:
            print(f"剪裁实时预览异常: {err}")

    @Slot()
    def _exit_dedicated_crop_mode(self):
        """
        安全退出剪辑模式并切断所有的多媒体句柄锁。
        已修复：在退出或确定应用剪裁成果时，强制重置对比滑块及对比按钮为未选中状态。
        """
        if getattr(self, 'is_playing', False):
            self.pause_video()

        # 1. 物理销毁 BGM 预览监听器
        self._stop_bgm_preview_monitoring()
        self._is_playing_bgm_preview = False
        self._update_bgm_preview_button_ui()

        # 2. 终止异步寻道解码子线程
        self._stop_async_scrub_reader()

        # 3. 释放 PyAV 视频容器句柄
        if hasattr(self, '_crop_pyav_container') and self._crop_pyav_container is not None:
            try:
                self._crop_pyav_container.close()
            except Exception:
                pass
            self._crop_pyav_container = None
            self._crop_pyav_stream = None

        # 4. 彻底切断音源占位，以便重载
        self._clear_audio_sources_safe()

        self._current_crop_clip_idx = -1

        if hasattr(self, 'top_bar_main'):
            self.top_bar_main.show()
        if hasattr(self, 'video_margin_wrapper'):
            self.ratio_container.setParent(self.video_margin_wrapper)

        self._mount_player_to_workspace(self.main_workspace_wrapper)
        self.vid_editor_stack.setCurrentWidget(self.vid_main_editor_page)

        # =========================================================================
        # 【滑块复位核心修复】：
        # 退出或应用剪辑时，强制关闭滑块遮挡，使主界面预览恢复最纯净无遮拦的成品画面
        # =========================================================================
        if hasattr(self, 'video_compare_mode_button') and self.video_compare_mode_button:
            self.video_compare_mode_button.blockSignals(True)
            self.video_compare_mode_button.setChecked(False)
            self.video_compare_mode_button.blockSignals(False)

        if hasattr(self, 'video_display_label') and self.video_display_label:
            self.video_display_label.set_compare_mode(False)
            self.video_display_label.set_allow_interaction(False)
            self.video_display_label.allow_zoom_pan = False
            self.video_display_label.set_active_object(-1)
            self.video_display_label.interaction_points_for_display = {}
            self.video_display_label.interaction_boxes_for_display = {}
            self.video_display_label.segmentation_masks_for_display = {}
            if hasattr(self.video_display_label, 'temp_multi_masks'):
                self.video_display_label.temp_multi_masks.clear()
            self.video_display_label.temp_annotation_frame_mask = None
            self.video_display_label.temp_annotation_target_id = -1
            self.video_display_label.temp_annotation_mask_frame_idx = -1
            self.video_display_label.fit_frame_to_view()
            self.video_display_label.update_cursor()

        self._recalc_global_timeline()
        self._display_frame_wrapper(self.current_frame_index)

        QTimer.singleShot(50, lambda: self._display_frame_wrapper(self.current_frame_index))
        QTimer.singleShot(100, lambda: self._update_video_time_label())

    @Slot()
    def _apply_crop_and_exit(self):
        """
        [安全重构版]：应用剪裁与音频修改并退出剪裁模式。
        核心：自动验证片段是否被裁剪，如果是，则销毁老缓存；如果此片段包含抠图蒙版，则触发单片段重烘焙！
        """
        import copy
        import numpy as np
        import cv2
        import os
        import time

        try:
            if getattr(self, 'is_playing', False):
                self.pause_video()

            self._stop_bgm_preview_monitoring()
            self._is_playing_bgm_preview = False
            self._stop_async_scrub_reader()

            if hasattr(self, '_crop_pyav_container') and self._crop_pyav_container is not None:
                try:
                    self._crop_pyav_container.close()
                except Exception:
                    pass
                self._crop_pyav_container = None
                self._crop_pyav_stream = None

            self._safely_reset_media_player(getattr(self, 'mpv_audio', None))
            self._safely_reset_media_player(getattr(self, 'mpv_bgm', None))

            idx = getattr(self, '_current_crop_clip_idx', -1)
            if idx == -1 or not hasattr(self, 'virtual_timeline') or idx >= len(self.virtual_timeline):
                self._exit_dedicated_crop_mode()
                return

            self._save_video_state()
            self._distribute_global_masks_to_clips()
            clip = self.virtual_timeline[idx]

            start = int(self.video_trim_slider.start_val)
            end = int(self.video_trim_slider.end_val)

            video_trim_changed = (start != clip.get('in_point', 0)) or (end != clip.get('out_point', 0))

            clip['in_point'] = start
            clip['out_point'] = end
            clip['frames'] = max(1, end - start)
            clip['duration'] = clip['frames'] / max(1.0, clip.get('fps', 30.0))

            # =========================================================================
            # 【缓存控制区】：如果你裁剪了带有蒙版的视频，老的缓存就会被清除，并标记需要重新烘焙！
            # =========================================================================
            clip_has_masks = False
            for masks in clip.get('local_masks', {}).values():
                if any(m is not None and np.any(m) for m in masks.values()):
                    clip_has_masks = True
                    break

            if video_trim_changed:
                if clip.get('baked_preview_dir') and os.path.exists(clip['baked_preview_dir']):
                    import shutil
                    shutil.rmtree(clip['baked_preview_dir'], ignore_errors=True)
                clip['baked_preview_dir'] = None

                # 置入等待重渲染队列
                if clip_has_masks:
                    self._pending_crop_rebake_clip_idx = idx

            self._invalidate_timeline_cache(keep_masks=True)

            if hasattr(self, '_cv_caps'):
                for uid, cap in list(self._cv_caps.items()):
                    try:
                        cap.release()
                    except Exception:
                        pass
                self._cv_caps.clear()

            if hasattr(self, '_last_valid_frames'):
                self._last_valid_frames.clear()

            if video_trim_changed or 'pixmap' not in clip or clip['pixmap'].isNull():
                frame = None
                try:
                    import av
                    with av.open(clip['path']) as container:
                        stream = container.streams.video[0]
                        fps = stream.average_rate
                        if not fps: fps = stream.r_frame_rate
                        fps = float(fps) if fps else clip.get('fps', 30.0)

                        time_base = float(stream.time_base)
                        frame_duration_pts = 1.0 / (fps * time_base)

                        seek_frame = max(0, start - 5)
                        seek_pts = int(seek_frame * frame_duration_pts)

                        try:
                            container.seek(seek_pts, stream=stream)
                        except Exception:
                            container.seek(0, stream=stream)

                        for av_frame in container.decode(video=0):
                            current_idx = int(round(av_frame.pts * time_base * fps))
                            if current_idx >= start:
                                frame = av_frame.to_ndarray(format='bgr24')
                                break
                except Exception as av_ex:
                    print(f"[DEBUG] PyAV cover extraction failed ({av_ex}). Switching to OpenCV fallback.")

                if frame is None:
                    try:
                        cap = cv2.VideoCapture(clip['path'])
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                            ret, cv_frame = cap.read()
                            if ret and cv_frame is not None: frame = cv_frame
                            cap.release()
                    except Exception as cv_ex:
                        print(f"[DEBUG] OpenCV fallback extraction also failed: {cv_ex}")

                if frame is not None:
                    target_w = getattr(self, 'video_width', 0)
                    target_h = getattr(self, 'video_height', 0)

                    if target_w > 0 and target_h > 0 and frame.shape[:2] != (target_h, target_w):
                        h, w = frame.shape[:2]
                        scale = min(target_w / w, target_h / h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                        x_off = (target_w - new_w) // 2
                        y_off = (target_h - new_h) // 2
                        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
                        frame = canvas

                    current_local_masks = clip.get('local_masks', {})
                    if start in current_local_masks and current_local_masks[start]:
                        h, w = frame.shape[:2]
                        solid_bg = self._get_current_bg_frame(h, w)
                        combined_alpha = np.zeros((h, w), dtype=np.float32)
                        has_mask = False

                        for mask_raw in current_local_masks[start].values():
                            if mask_raw is not None:
                                mask_float = mask_raw.astype(np.float32) if mask_raw.dtype == bool else np.clip(
                                    mask_raw.astype(np.float32), 0.0, 1.0)
                                if mask_float.shape != (h, w):
                                    mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)
                                combined_alpha = np.maximum(combined_alpha, mask_float)
                                has_mask = True

                        if has_mask:
                            alpha_3d = combined_alpha[:, :, np.newaxis]
                            blended = frame.astype(np.float32) * alpha_3d + solid_bg.astype(np.float32) * (
                                    1.0 - alpha_3d)
                            frame = np.clip(blended, 0.0, 255.0).astype(np.uint8)

                    pixmap = convert_cv_to_pixmap(frame)
                    if pixmap and not pixmap.isNull():
                        clip['pixmap'] = pixmap
                        item = self.storyboard_list.item(idx)
                        if item:
                            widget = self.storyboard_list.itemWidget(item)
                            if hasattr(widget, 'bg_label'):
                                scaled_pix = pixmap.scaled(240, 135, Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation)
                                widget.bg_label.setPixmap(scaled_pix)

            self.total_frames = sum(v['frames'] for v in self.virtual_timeline)
            total_time_sec = sum(v['duration'] for v in self.virtual_timeline)

            if hasattr(self, 'time_label_total'):
                self.time_label_total.setText(self._format_time(total_time_sec))
            if hasattr(self, 'video_frame_spinbox'):
                self.video_frame_spinbox.setMaximum(max(1, self.total_frames))

            self.storyboard_list.blockSignals(True)
            self.storyboard_list.clear()

            for vid_data in self.virtual_timeline:
                item_widget = StoryboardItemWidget(vid_data['path'], vid_data['duration'], vid_data.get('pixmap'))
                item_widget.sync_ui_with_data(vid_data.get('mute_original', False))

                list_item = QListWidgetItem(self.storyboard_list)
                list_item.setSizeHint(QSize(240, 170))
                list_item.setData(Qt.ItemDataRole.UserRole + 1, vid_data)
                self.storyboard_list.setItemWidget(list_item, item_widget)
            self.storyboard_list.blockSignals(False)

            if 0 <= idx < self.storyboard_list.count():
                self.storyboard_list.setCurrentRow(idx)
                self.storyboard_list.item(idx).setSelected(True)

            if hasattr(self, 'video_thumbnail_scrubber'):
                self.video_thumbnail_scrubber.set_params(
                    self.total_frames, getattr(self, 'video_thumbnail_paths', []),
                    getattr(self, 'video_width', 0), getattr(self, 'video_height', 0),
                    getattr(self, 'video_fps', 30.0), getattr(self, 'is_gif_input', False),
                    getattr(self, 'gif_frame_durations_ms', [])
                )

            self._update_bg_color_button_style()

            self._current_crop_clip_idx = -1
            if hasattr(self, 'top_bar_main'):
                self.top_bar_main.show()
            if hasattr(self, 'video_margin_wrapper'):
                self.ratio_container.setParent(self.video_margin_wrapper)

            self._mount_player_to_workspace(self.main_workspace_wrapper)
            self.vid_editor_stack.setCurrentWidget(self.vid_main_editor_page)

            new_global_start = sum(self.virtual_timeline[i]['frames'] for i in range(idx))
            self.current_frame_index = new_global_start

            self._gather_global_masks_from_clips()

            # =========================================================================
            # 【完美接轨】：告诉提取器完成抽帧后直接放行，由我们在 _on_global_extraction_finished 中拦截处理烘焙！
            # =========================================================================
            self._skip_rebake_on_next_extraction = True
            self._trigger_global_timeline_rebuild()

        except Exception as e:
            print(f"[FATAL CROP ERROR] Error applying crop values: {e}")
            import traceback
            traceback.print_exc()
            self._exit_dedicated_crop_mode()

    @Slot()
    def _enter_dedicated_matting_mode(self):
        """
        进入智能抠图沙盒模式。
        【修复优化】：智能评估已抠图或剪裁后片段的历史对比成果与自定义背景状态。
        """
        if getattr(self, 'is_extracting_frames', False):
            self.show_status_message("视频序列正在后台同步中，完成后将自动进入智能抠图...", 3000)
            self._pending_mode_switch = "matting"
            return

        if getattr(self, 'is_playing', False):
            self.pause_video()
        if hasattr(self, 'top_bar_main'):
            self.top_bar_main.hide()

        clip_idx = -1
        if hasattr(self, 'storyboard_list') and self.storyboard_list.selectedItems():
            clip_idx = self.storyboard_list.row(self.storyboard_list.selectedItems()[0])
        else:
            clip_idx, _ = self._get_current_clip_info()

        if clip_idx is None or clip_idx < 0:
            if hasattr(self, 'virtual_timeline') and len(self.virtual_timeline) > 0:
                clip_idx = 0
            else:
                QMessageBox.information(self, "提示", "请先在主页添加视频片段。")
                if hasattr(self, 'top_bar_main'):
                    self.top_bar_main.show()
                return

        if not hasattr(self, 'virtual_timeline') or clip_idx >= len(self.virtual_timeline):
            return

        clip = self.virtual_timeline[clip_idx]
        fps = clip.get('fps', 30.0)
        duration_sec = clip['frames'] / max(1.0, fps)

        if duration_sec > 10.0:
            QMessageBox.warning(
                self,
                "时长超限保护",
                f"当前选中片段时长为 {duration_sec:.1f} 秒。\n\n"
                "为了保证显存安全和抠图精细度，单次智能抠图时长最多不能超过 10 秒。\n"
                "请先使用【剪裁】功能，将其修剪至 10 秒以内！"
            )
            if hasattr(self, 'top_bar_main'):
                self.top_bar_main.show()
            return

        # =========================================================================
        # 核心评估：即使预渲染文件夹被剪裁机制清除，只要内存掩码有效，即断定抠图成果依然存在
        # =========================================================================
        has_baked = bool(clip.get('baked_preview_dir') and os.path.exists(clip['baked_preview_dir']))
        has_masks = False
        for masks in clip.get('local_masks', {}).values():
            if any(m is not None and np.any(m) for m in masks.values()):
                has_masks = True
                break

        # 即使剪裁后无烘焙文件，只要有掩码数据，就代表已完成抠图，应自动开启对比展示
        self.video_segmentation_finished = has_baked or has_masks

        # 释放与本片段不相关的预读缓存池
        if hasattr(self, '_prefetch_pool'):
            try:
                self._prefetch_pool.shutdown(wait=False)
            except Exception:
                pass
            del self._prefetch_pool
        if hasattr(self, '_video_frame_cache'):
            self._video_frame_cache.clear()
        if hasattr(self, '_prefetching_indices'):
            self._prefetching_indices.clear()

        if hasattr(self, 'video_display_label'):
            self.video_display_label.clear_display()
            self.video_display_label.repaint()

        self._save_video_state()
        self.show_global_loading_overlay("正在提取独立剪辑沙盒，请稍候...")
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

        self._distribute_global_masks_to_clips()

        self._undo_stack_size_before_matting = len(getattr(self, 'video_undo_stack', []))

        self._matting_clip_idx = clip_idx
        self._matting_global_start = sum(v['frames'] for v in self.virtual_timeline[:clip_idx])
        self._matting_global_end = self._matting_global_start + clip['frames'] - 1

        self._matting_backup_local_targets = copy.deepcopy(clip.get('local_targets', {}))
        self._matting_backup_local_masks = copy.deepcopy(clip.get('local_masks', {}))

        self._matting_backup_bg = {
            'bg_color': clip.get('bg_color', QColor(0, 255, 0)),
            'bg_image_path': clip.get('bg_image_path', None),
            'bg_is_transparent': clip.get('bg_is_transparent', False)
        }

        self._is_matting_dirty = False

        self.target_points = {}
        self.processed_masks = {}

        in_point = int(clip.get('in_point', 0))
        for tid, tdata in clip.get('local_targets', {}).items():
            tdata_copy = copy.deepcopy(tdata)
            physical_ann = tdata_copy['annotation_frame']
            tdata_copy['annotation_frame'] = self._matting_global_start + (physical_ann - in_point)
            self.target_points[tid] = tdata_copy

        for physical_idx, masks in clip.get('local_masks', {}).items():
            global_idx = self._matting_global_start + (physical_idx - in_point)
            self.processed_masks[global_idx] = copy.deepcopy(masks)

        self.current_target_id = -1
        self.next_target_id = max(self.target_points.keys(), default=-1) + 1
        self._refresh_video_objects_list()

        is_same_clip = getattr(self, '_last_built_sandbox_clip_idx', -1) == clip_idx
        if getattr(self, 'clip_sandbox_dir', None) is None:
            is_same_clip = False

        if is_same_clip:
            self._finalize_enter_matting_mode(clip_idx, clip, True)
            return

        if getattr(self, 'video_inference_state', None) is not None:
            try:
                self.video_predictor.reset_state(self.video_inference_state)
            except Exception:
                pass
            self.video_inference_state = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.clip_sandbox_dir = os.path.join(TEMP_BASE_DIR, "ui_sandbox")

        class SandboxBuilderThread(QThread):
            finished_signal = Signal(int)

            def __init__(self, start_idx, end_idx, in_pt, src_dir, dst_dir):
                super().__init__()
                self.start_idx = start_idx
                self.end_idx = end_idx
                self.in_pt = in_pt
                self.src_dir = src_dir
                self.dst_dir = dst_dir

            def run(self):
                import shutil
                shutil.rmtree(self.dst_dir, ignore_errors=True)
                os.makedirs(self.dst_dir, exist_ok=True)
                copied_count = 0
                for g_idx in range(self.start_idx, self.end_idx + 1):
                    l_idx = g_idx - self.start_idx
                    src1 = os.path.join(self.src_dir, f"{g_idx:05d}.jpg")
                    src2 = os.path.join(self.src_dir, f"{g_idx}.jpg")
                    src = src1 if os.path.exists(src1) else (src2 if os.path.exists(src2) else None)

                    if not src:
                        physical_target_frame = self.in_pt + l_idx
                        src = os.path.join(self.src_dir, f"{physical_target_frame:05d}.jpg")
                        if not os.path.exists(src):
                            src = os.path.join(self.src_dir, f"{physical_target_frame}.jpg")

                    dst = os.path.join(self.dst_dir, f"{l_idx:05d}.jpg")

                    if src and os.path.exists(src):
                        shutil.copy2(src, dst)
                        copied_count += 1
                self.finished_signal.emit(copied_count)

        self._sandbox_thread = SandboxBuilderThread(
            self._matting_global_start, self._matting_global_end,
            in_point, self.temp_frame_dir, self.clip_sandbox_dir
        )

        def on_sandbox_finished(copied_count):
            self._sandbox_thread.deleteLater()
            if copied_count == 0:
                QMessageBox.warning(self, "沙盒建立失败", "未能提取到有效的物理帧数据。")
                if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                    self._global_loading_overlay.hide()
                return

            self._last_built_sandbox_clip_idx = clip_idx
            self._finalize_enter_matting_mode(clip_idx, clip, False)

        self._sandbox_thread.finished_signal.connect(on_sandbox_finished)
        self._sandbox_thread.start()

    def _finalize_enter_matting_mode(self, clip_idx, clip, is_same_clip):
        """进入智能抠图沙盒模式并同步 UI"""
        try:
            self._mount_player_to_workspace(self.matting_workspace_wrapper)
            self.vid_editor_stack.setCurrentWidget(self.vid_dedicated_matting_page)

            if hasattr(self, 'video_thumbnail_scrubber'):
                local_thumbs = getattr(self, 'video_thumbnail_paths', [])[self._matting_global_start: self._matting_global_end + 1]
                self.video_thumbnail_scrubber.set_params(
                    clip['frames'], local_thumbs, getattr(self, 'video_width', 0), getattr(self, 'video_height', 0),
                    clip['fps'], getattr(self, 'is_gif_input', False), []
                )

            if hasattr(self, 'time_label_total'):
                self.time_label_total.setText(self._format_time(clip['duration']))

            self.current_frame_index = self._matting_global_start

            if hasattr(self, 'video_display_label'):
                self.video_display_label.set_allow_interaction(True)
                self.video_display_label.allow_zoom_pan = True
                self.video_display_label.update_cursor()

            has_baked = bool(clip.get('baked_preview_dir') and os.path.exists(clip['baked_preview_dir']))
            has_masks = False
            for masks in clip.get('local_masks', {}).values():
                if any(m is not None and np.any(m) for m in masks.values()):
                    has_masks = True
                    break

            self.video_segmentation_finished = has_baked or has_masks

            if self.video_segmentation_finished:
                self.current_target_id = -1
                if hasattr(self, 'video_objects_list'):
                    self.video_objects_list.clearSelection()
                if hasattr(self, 'video_compare_mode_button') and self.video_compare_mode_button:
                    self.video_compare_mode_button.blockSignals(True)
                    self.video_compare_mode_button.setChecked(True)
                    self.video_compare_mode_button.blockSignals(False)
                if hasattr(self, 'video_display_label') and self.video_display_label:
                    self.video_display_label.set_compare_mode(True)
                    self.video_display_label.set_active_object(-1)

            self._display_frame_wrapper(self.current_frame_index)

            # 更新静音勾选框 (基于 mute_all)
            is_muted = clip.get('mute_all', False)
            if hasattr(self, 'matting_mute_checkbox'):
                self.matting_mute_checkbox.blockSignals(True)
                self.matting_mute_checkbox.setChecked(is_muted)
                self.matting_mute_checkbox.blockSignals(False)

            use_direct_matting = hasattr(self, 'vid_direct_matting_checkbox') and self.vid_direct_matting_checkbox.isChecked()
            initializing_predictor = False
            if not use_direct_matting:
                target_dir = getattr(self, 'clip_sandbox_dir', getattr(self, 'temp_frame_dir', None))
                if getattr(self, 'video_path', None) and target_dir:
                    if getattr(self, 'video_inference_state', None) is None:
                        if not getattr(self, 'is_extracting_frames', False) and getattr(self, 'video_predictor_loaded', False):
                            initializing_predictor = True
                            self._initialize_video_predictor_state()
                            return

            if not initializing_predictor:
                from PySide6.QtWidgets import QApplication
                while QApplication.overrideCursor() is not None:
                    QApplication.restoreOverrideCursor()
                if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                    self._global_loading_overlay.hide()

            self._update_video_bg_selector_ui()
            self._update_bg_color_button_style()
            self.update_button_states()

        except Exception as e:
            print(f"[ERROR] Failed to finalize matting mode: {e}")
            import traceback
            from PySide6.QtWidgets import QApplication
            traceback.print_exc()
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            self.update_button_states()

    @Slot()
    def _apply_matting_and_exit(self):
        """
        [性能飞跃版]：应用并退出智能抠图。
        拦截冗余重绘：如果没有使用画笔或新增打点修改蒙版，直接复用刚才自动抠图的成果，0毫秒秒退！
        """
        is_dirty = getattr(self, '_is_matting_dirty', False)

        if hasattr(self, 'video_display_label') and hasattr(self.video_display_label, 'temp_multi_masks'):
            for frame_idx, obj_masks in self.video_display_label.temp_multi_masks.items():
                if obj_masks:
                    is_dirty = True  # 检测到有时序运算残留，说明修改了
                if frame_idx not in self.processed_masks:
                    self.processed_masks[frame_idx] = {}
                for obj_id, mask_np in obj_masks.items():
                    if mask_np is not None:
                        self.processed_masks[frame_idx][obj_id] = mask_np.copy()

        if hasattr(self, 'video_display_label'):
            temp_idx = getattr(self.video_display_label, 'temp_annotation_mask_frame_idx', -1)
            temp_mask = getattr(self.video_display_label, 'temp_annotation_frame_mask', None)
            temp_obj_id = getattr(self.video_display_label, 'temp_annotation_target_id', -1)
            if temp_idx != -1 and temp_mask is not None and temp_obj_id != -1:
                is_dirty = True
                if temp_idx not in self.processed_masks:
                    self.processed_masks[temp_idx] = {}
                self.processed_masks[temp_idx][temp_obj_id] = temp_mask.copy()

        # 将新蒙版归档到各个片段
        self._distribute_global_masks_to_clips()
        self._gather_global_masks_from_clips()

        clip_idx = getattr(self, '_matting_clip_idx', -1)
        if clip_idx == -1 or not hasattr(self, 'virtual_timeline') or clip_idx >= len(self.virtual_timeline):
            self._exit_dedicated_matting_mode()
            return

        # =========================================================================
        # 【拦截枢纽】：没有实质性修改，跳过 BakeSingleClipWorker，极速链结秒退！
        # =========================================================================
        if not is_dirty:
            self._assemble_global_render_dir()
            self._exit_dedicated_matting_mode()
            return

        self.show_global_loading_overlay(_TR("检测到手工修补，正在重组发丝图层..."), 0)

        clip = self.virtual_timeline[clip_idx]
        local_baked_dir = os.path.join(TEMP_BASE_DIR, f"baked_clip_{uuid.uuid4().hex[:8]}")

        self._single_bake_thread = QThread(self)
        self._single_bake_worker = BakeSingleClipWorker(
            raw_frame_dir=self.temp_frame_dir,
            processed_masks=self.processed_masks,
            start_frame=self._matting_global_start,
            end_frame=self._matting_global_end,
            target_w=self.video_width or 1280,
            target_h=self.video_height or 720,
            bg_color=clip.get('bg_color', QColor(0, 255, 0)),
            custom_bg_path=clip.get('bg_image_path', None),
            bg_is_transparent=clip.get('bg_is_transparent', False),
            output_dir=local_baked_dir
        )

        self._single_bake_worker.moveToThread(self._single_bake_thread)
        self._single_bake_worker.progress.connect(
            lambda pct, msg: self.show_global_loading_overlay(msg, pct)
        )
        self._single_bake_worker.finished.connect(self._on_single_clip_bake_finished_wrapper)

        self._single_bake_thread.started.connect(self._single_bake_worker.run)
        self._single_bake_thread.start()

    @Slot()
    def _discard_matting_and_exit(self):
        """[无痕放弃] 丢弃编辑并返回主页面，瞬间链结重组时间线成果。"""
        import copy
        clip_idx = getattr(self, '_matting_clip_idx', -1)
        if clip_idx != -1 and clip_idx < len(self.virtual_timeline):
            clip = self.virtual_timeline[clip_idx]
            clip['local_targets'] = copy.deepcopy(getattr(self, '_matting_backup_local_targets', {}))
            clip['local_masks'] = copy.deepcopy(getattr(self, '_matting_backup_local_masks', {}))

            # 【核心修复4】：把刚才在沙盒里面手贱乱点的背景还原回来
            if hasattr(self, '_matting_backup_bg'):
                clip.update(self._matting_backup_bg)

        before_size = getattr(self, '_undo_stack_size_before_matting', -1)
        if before_size != -1 and hasattr(self, 'video_undo_stack') and self.video_undo_stack:
            self.video_undo_stack = self.video_undo_stack[:before_size]

        # 直接组装，不重新执行重烘焙
        self._assemble_global_render_dir()
        self._exit_dedicated_matting_mode()

    @Slot()
    def _exit_dedicated_matting_mode(self):
        """
        安全退出抠图沙盒，复位滑块并强刷主页面视图。
        """
        if getattr(self, 'is_playing', False):
            self.pause_video()

        self._clear_audio_sources_safe()

        if hasattr(self, 'top_bar_main'):
            self.top_bar_main.show()
        if hasattr(self, 'video_margin_wrapper'):
            self.ratio_container.setParent(self.video_margin_wrapper)

        self._mount_player_to_workspace(self.main_workspace_wrapper)
        self.vid_editor_stack.setCurrentWidget(self.vid_main_editor_page)

        # =========================================================================
        # 【滑块强制复位】：确保返回合并主界面时无滑块分界线干扰
        # =========================================================================
        if hasattr(self, 'video_compare_mode_button') and self.video_compare_mode_button:
            self.video_compare_mode_button.blockSignals(True)
            self.video_compare_mode_button.setChecked(False)
            self.video_compare_mode_button.blockSignals(False)

        if hasattr(self, 'video_display_label') and self.video_display_label:
            self.video_display_label.set_compare_mode(False)  # 关闭滑块显示
            self.video_display_label.set_allow_interaction(False)
            self.video_display_label.allow_zoom_pan = False
            self.video_display_label.set_active_object(-1)
            self.video_display_label.interaction_points_for_display = {}
            self.video_display_label.interaction_boxes_for_display = {}
            self.video_display_label.segmentation_masks_for_display = {}
            if hasattr(self.video_display_label, 'temp_multi_masks'):
                self.video_display_label.temp_multi_masks.clear()
            self.video_display_label.temp_annotation_frame_mask = None
            self.video_display_label.temp_annotation_target_id = -1
            self.video_display_label.temp_annotation_mask_frame_idx = -1
            self.video_display_label.fit_frame_to_view()
            self.video_display_label.update_cursor()

        self._matting_clip_idx = -1
        self._matting_global_start = 0
        self._matting_global_end = 0

        self._gather_global_masks_from_clips()

        if not self.processed_masks:
            self.video_segmentation_finished = False

        self._recalc_global_timeline()
        self._display_frame_wrapper(self.current_frame_index)

        QTimer.singleShot(50, lambda: self._display_frame_wrapper(self.current_frame_index))
        QTimer.singleShot(100, lambda: self._update_video_time_label())

    @Slot(float)
    def _on_bgm_clip_start_changed(self, value):
        """
        [New slot function] Handle response when background music's own trim time changes.
        """
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline'): return

        self.virtual_timeline[idx]['custom_audio_clip_start'] = value

        # Real-time feedback to reset player
        if self._current_playing_clip_idx == idx:
            self._sync_audio_engine_to_current_frame(idx)
            if getattr(self, 'is_playing', False):
                self.bgm_player.play()

    @Slot(bool)
    def toggle_video_compare_mode(self, checked: bool):
        """
        对比模式按钮切换槽函数。
        安全保护：非智能抠图沙盒页面不响应此操作。
        """
        is_matting_page = getattr(self, 'vid_editor_stack', None) and self.vid_editor_stack.currentWidget() == getattr(
            self, 'vid_dedicated_matting_page', None)

        if not is_matting_page:
            # 非抠图沙盒页，强行纠正并弹起按钮
            if hasattr(self, 'video_compare_mode_button') and self.video_compare_mode_button:
                self.video_compare_mode_button.blockSignals(True)
                self.video_compare_mode_button.setChecked(False)
                self.video_compare_mode_button.blockSignals(False)
            if hasattr(self, 'video_display_label') and self.video_display_label:
                self.video_display_label.set_compare_mode(False)
            return

        if hasattr(self, 'video_display_label') and self.video_display_label:
            self.video_display_label.set_compare_mode(checked)
            if hasattr(self, 'update_button_states'):
                self.update_button_states()



class VideoThumbnailScrubber(QWidget):
        frame_selected = Signal(int)
        frame_hovered = Signal(int)
        THUMBNAIL_HEIGHT = 50
        INDIVIDUAL_TIMECODE_AREA_HEIGHT = 18
        THUMBNAIL_MIN_WIDTH = 30
        THUMBNAIL_SPACING = 1
        PLAYHEAD_LINE_COLOR = QColor(79, 70, 229, 220)
        PLAYHEAD_LINE_WIDTH = 1
        PLAYHEAD_TOP_BAR_HEIGHT = 4
        PLAYHEAD_TOP_BAR_WIDTH = 8
        PLAYHEAD_TOP_BAR_COLOR = PLAYHEAD_LINE_COLOR
        PLAYHEAD_TOP_BAR_OFFSET_Y = 1
        PLAYHEAD_BOTTOM_TRIANGLE_HEIGHT = 5
        PLAYHEAD_BOTTOM_TRIANGLE_WIDTH = 8
        HOVER_RECT_COLOR = QColor(79, 70, 229, 40)
        PLACEHOLDER_BG_COLOR = QColor(243, 244, 246)
        PLACEHOLDER_TEXT_COLOR = QColor(156, 163, 175)
        INFO_TEXT_COLOR = QColor(107, 114, 128)
        INFO_TEXT_POINT_SIZE = 8
        INFO_TEXT_PADDING_X = 5
        INFO_TEXT_PADDING_Y = 2
        TIMECODE_TEXT_COLOR_INDIVIDUAL = QColor(156, 163, 175)
        TIMECODE_TEXT_POINT_SIZE_INDIVIDUAL = 7
        TIMECODE_TEXT_MARGIN_BOTTOM_INDIVIDUAL = 2

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMouseTracking(True)
            self.setAutoFillBackground(False)

            self._total_frames = 0
            self._current_frame_idx = -1
            self._hover_frame_idx = -1
            self._thumbnail_paths: list[str | None] = []
            self._thumbnail_cache: dict[int, QPixmap] = {}
            self._thumbnail_width = int(VideoThumbnailScrubber.THUMBNAIL_HEIGHT * (16.0 / 9.0))
            self._is_dragging = False
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            self._scroll_bar = QScrollBar(Qt.Orientation.Horizontal, self)
            self._scroll_bar.valueChanged.connect(self._on_scroll)
            self._scroll_bar.hide()
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.setInterval(30)
            self._resize_timer.timeout.connect(self._update_scroll_bar_and_layout)

            self._time_str_display = ""
            self._frame_str_display = "[0/0]"
            self._fps = VIDEO_DEFAULT_FPS
            self._is_gif = False
            self._gif_frame_durations_ms: list[int] = []
            self._precalculated_timecodes_sec: list[float] = []
            self._update_minimum_height()

        def _get_total_drawable_height(self) -> int:
            return VideoThumbnailScrubber.INDIVIDUAL_TIMECODE_AREA_HEIGHT + VideoThumbnailScrubber.THUMBNAIL_HEIGHT

        def _update_minimum_height(self):
            sb_height = self._scroll_bar.style().pixelMetric(
                QStyle.PixelMetric.PM_ScrollBarExtent) if self._scroll_bar.isVisible() else 0
            if sb_height <= 0 and self._scroll_bar.isVisible(): sb_height = 16
            new_min_height = self._get_total_drawable_height() + sb_height + 2
            if self.minimumHeight() != new_min_height:
                self.setMinimumHeight(new_min_height)
                self.updateGeometry()

        def set_params(self, total_frames: int, thumbnail_paths: list[str | None], frame_width: int, frame_height: int,
                       fps: float, is_gif: bool, gif_frame_durations_ms: list[int]):
            self._total_frames = total_frames
            self._thumbnail_paths = thumbnail_paths if thumbnail_paths else []
            self._thumbnail_cache.clear()
            self._current_frame_idx = 0 if total_frames > 0 else -1
            self._hover_frame_idx = -1
            self._is_dragging = False
            self._fps = fps if fps > 0 else VIDEO_DEFAULT_FPS
            self._is_gif = is_gif
            self._gif_frame_durations_ms = gif_frame_durations_ms if is_gif and gif_frame_durations_ms else []
            self._precalculate_timecodes()

            if frame_height > 0 and frame_width > 0:
                aspect_ratio = float(frame_width) / frame_height
                self._thumbnail_width = max(VideoThumbnailScrubber.THUMBNAIL_MIN_WIDTH,
                                            int(VideoThumbnailScrubber.THUMBNAIL_HEIGHT * aspect_ratio))
            else:
                self._thumbnail_width = max(VideoThumbnailScrubber.THUMBNAIL_MIN_WIDTH,
                                            int(VideoThumbnailScrubber.THUMBNAIL_HEIGHT * (16.0 / 9.0)))

            self._update_scroll_bar_and_layout()
            if self._total_frames == 0: self.set_info_text("", "[0/0]")
            self.update()

        def _precalculate_timecodes(self):
            self._precalculated_timecodes_sec = []
            if self._total_frames == 0: return
            if self._is_gif and self._gif_frame_durations_ms and len(
                    self._gif_frame_durations_ms) == self._total_frames:
                current_time_ms = 0
                for duration_ms in self._gif_frame_durations_ms:
                    self._precalculated_timecodes_sec.append(current_time_ms / 1000.0)
                    current_time_ms += duration_ms
            elif self._fps > 0:
                for i in range(self._total_frames): self._precalculated_timecodes_sec.append(i / self._fps)
            else:
                for i in range(self._total_frames): self._precalculated_timecodes_sec.append(
                    i * (VIDEO_PLAYBACK_INTERVAL_MS / 1000.0))

        def set_current_frame(self, frame_idx: int):
            new_idx = max(0, min(frame_idx, self._total_frames - 1 if self._total_frames > 0 else 0))
            if self._total_frames == 0: new_idx = -1
            if self._current_frame_idx != new_idx: self._current_frame_idx = new_idx
            if new_idx != -1: self._ensure_frame_visible(new_idx); self.update()

        def _ensure_frame_visible(self, frame_idx: int):
            if not self._scroll_bar.isVisible() or self._total_frames == 0: return
            drawing_width = self.width()
            single_thumb_total_width = self._thumbnail_width + VideoThumbnailScrubber.THUMBNAIL_SPACING
            if single_thumb_total_width <= 0: return

            first_visible_idx = self._scroll_bar.value()
            num_thumbnails_can_fit = self._calculate_visible_thumb_count(drawing_width)
            if num_thumbnails_can_fit <= 0: return
            last_visible_idx = first_visible_idx + num_thumbnails_can_fit - 1

            new_scroll_value = self._scroll_bar.value()
            if frame_idx < first_visible_idx:
                new_scroll_value = frame_idx
            elif frame_idx > last_visible_idx:
                target_scroll_value = frame_idx - (num_thumbnails_can_fit // 3)
                new_scroll_value = max(0, min(target_scroll_value, self._scroll_bar.maximum()))

            if self._scroll_bar.value() != new_scroll_value: self._scroll_bar.setValue(new_scroll_value)

        def _on_scroll(self, value: int):
            self.update()

        def _calculate_visible_thumb_count(self, area_width: int) -> int:
            item_width = self._thumbnail_width + VideoThumbnailScrubber.THUMBNAIL_SPACING
            if item_width <= 0: return 0
            return area_width // item_width

        def _update_scroll_bar_and_layout(self):
            drawing_width = self.width()
            current_sb_visible = self._scroll_bar.isVisible()
            new_sb_visible = False
            if self._total_frames > 0 and self._thumbnail_width > 0:
                num_can_fit_in_view = self._calculate_visible_thumb_count(drawing_width)
                if num_can_fit_in_view > 0 and self._total_frames > num_can_fit_in_view:
                    new_sb_visible = True
                    self._scroll_bar.setMinimum(0)
                    self._scroll_bar.setMaximum(max(0, self._total_frames - num_can_fit_in_view))
                    self._scroll_bar.setPageStep(num_can_fit_in_view)
                    self._scroll_bar.setSingleStep(1)
                else:
                    new_sb_visible = False
                    self._scroll_bar.setValue(0)
            else:
                new_sb_visible = False
                self._scroll_bar.setValue(0)

            if current_sb_visible != new_sb_visible:
                self._scroll_bar.setVisible(new_sb_visible)
                self._update_minimum_height()

            if self._scroll_bar.isVisible():
                sb_height = self._scroll_bar.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
                if sb_height <= 0: sb_height = 16
                self._scroll_bar.setGeometry(0, self._get_total_drawable_height() + 1, drawing_width, sb_height)
            self.update()

        def resizeEvent(self, event: QResizeEvent):
            super().resizeEvent(event)
            self._resize_timer.start()

        def _get_thumbnail(self, frame_idx: int) -> QPixmap | None:
            if not (0 <= frame_idx < len(self._thumbnail_paths)): return None
            thumb_path = self._thumbnail_paths[frame_idx]
            if thumb_path is None: return None
            if frame_idx in self._thumbnail_cache: return self._thumbnail_cache[frame_idx]

            if os.path.exists(thumb_path):
                pixmap = QPixmap(thumb_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaledToHeight(VideoThumbnailScrubber.THUMBNAIL_HEIGHT,
                                                          Qt.TransformationMode.SmoothTransformation)
                    if scaled_pixmap.width() > self._thumbnail_width:
                        scaled_pixmap = scaled_pixmap.scaledToWidth(self._thumbnail_width,
                                                                    Qt.TransformationMode.SmoothTransformation)
                    self._thumbnail_cache[frame_idx] = scaled_pixmap
                    return scaled_pixmap
            return None

        def _frame_from_pos(self, pos: QPoint) -> int:
            if self._total_frames == 0 or pos.y() < 0 or pos.y() > self._get_total_drawable_height(): return -1
            first_visible_thumbnail_index = self._scroll_bar.value() if self._scroll_bar.isVisible() else 0
            item_width_with_spacing = self._thumbnail_width + VideoThumbnailScrubber.THUMBNAIL_SPACING
            if item_width_with_spacing <= 0: return -1

            clicked_thumb_visual_index = pos.x() // item_width_with_spacing
            frame_idx = first_visible_thumbnail_index + clicked_thumb_visual_index

            if 0 <= frame_idx < self._total_frames:
                start_x_of_clicked_thumb_in_view = clicked_thumb_visual_index * item_width_with_spacing
                if start_x_of_clicked_thumb_in_view <= pos.x() < start_x_of_clicked_thumb_in_view + self._thumbnail_width:
                    return frame_idx
            return -1

        def set_info_text(self, time_str: str, frame_str: str):
            if self._frame_str_display != frame_str:
                self._frame_str_display = frame_str
                self.update()

        def _format_seconds_to_individual_timecode(self, sec_float: float) -> str:
            total_milliseconds = int(sec_float * 1000)
            milliseconds_part = (total_milliseconds % 1000) // 10
            total_seconds_int = total_milliseconds // 1000
            minutes = total_seconds_int // 60
            seconds = total_seconds_int % 60
            return f"{minutes:02d}:{seconds:02d}.{milliseconds_part:02d}"

        def paintEvent(self, event: QPaintEvent):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#FFFFFF"))
            painter.drawRoundedRect(self.rect(), 12, 12)

            clip_path = QPainterPath()
            clip_path.addRoundedRect(self.rect(), 12, 12)
            painter.setClipPath(clip_path)

            if self._total_frames == 0:
                painter.setPen(VideoThumbnailScrubber.PLACEHOLDER_TEXT_COLOR)
                placeholder_rect = QRect(0, 0, self.width(), self._get_total_drawable_height())
                painter.drawText(placeholder_rect, Qt.AlignmentFlag.AlignCenter, "视频未加载")
                self._draw_info_text(painter,
                                     QRect(0, 0, self.width(), VideoThumbnailScrubber.INDIVIDUAL_TIMECODE_AREA_HEIGHT))
                painter.end()
                return

            timecode_area_y_start = 0
            thumbnail_area_y_start = timecode_area_y_start + VideoThumbnailScrubber.INDIVIDUAL_TIMECODE_AREA_HEIGHT

            first_thumb_to_draw = self._scroll_bar.value() if self._scroll_bar.isVisible() else 0
            num_thumbs_to_draw_estimate = self._calculate_visible_thumb_count(self.width()) + 2

            painter.save()
            painter.setClipRect(QRect(0, 0, self.width(), self._get_total_drawable_height()))

            font_individual_timecode = painter.font()
            font_individual_timecode.setPointSize(VideoThumbnailScrubber.TIMECODE_TEXT_POINT_SIZE_INDIVIDUAL)

            for i in range(num_thumbs_to_draw_estimate):
                actual_frame_idx = first_thumb_to_draw + i
                if actual_frame_idx >= self._total_frames: break

                current_draw_x = (actual_frame_idx - first_thumb_to_draw) * (
                        self._thumbnail_width + VideoThumbnailScrubber.THUMBNAIL_SPACING)
                if current_draw_x + self._thumbnail_width < 0: continue
                if current_draw_x > self.width(): break

                if actual_frame_idx < len(self._precalculated_timecodes_sec):
                    time_sec = self._precalculated_timecodes_sec[actual_frame_idx]
                    timecode_str = self._format_seconds_to_individual_timecode(time_sec)
                    painter.setFont(font_individual_timecode)
                    painter.setPen(VideoThumbnailScrubber.TIMECODE_TEXT_COLOR_INDIVIDUAL)
                    timecode_text_rect = QRect(current_draw_x, timecode_area_y_start, self._thumbnail_width,
                                               VideoThumbnailScrubber.INDIVIDUAL_TIMECODE_AREA_HEIGHT - VideoThumbnailScrubber.TIMECODE_TEXT_MARGIN_BOTTOM_INDIVIDUAL)
                    painter.drawText(timecode_text_rect, Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter,
                                     timecode_str)

                thumbnail_pixmap = self._get_thumbnail(actual_frame_idx)
                thumb_img_rect = QRect(current_draw_x, thumbnail_area_y_start, self._thumbnail_width,
                                       VideoThumbnailScrubber.THUMBNAIL_HEIGHT)
                if thumbnail_pixmap:
                    px = current_draw_x + max(0, (self._thumbnail_width - thumbnail_pixmap.width()) // 2)
                    py = thumbnail_area_y_start + max(0, (
                            VideoThumbnailScrubber.THUMBNAIL_HEIGHT - thumbnail_pixmap.height()) // 2)
                    painter.drawPixmap(px, py, thumbnail_pixmap)
                else:
                    painter.fillRect(thumb_img_rect, VideoThumbnailScrubber.PLACEHOLDER_BG_COLOR)
                    painter.setPen(VideoThumbnailScrubber.PLACEHOLDER_TEXT_COLOR)
                    painter.drawText(thumb_img_rect, Qt.AlignmentFlag.AlignCenter, str(actual_frame_idx + 1))

                if actual_frame_idx == self._hover_frame_idx and not self._is_dragging:
                    full_item_rect_for_hover = QRect(current_draw_x, timecode_area_y_start, self._thumbnail_width,
                                                     self._get_total_drawable_height())
                    painter.fillRect(full_item_rect_for_hover, VideoThumbnailScrubber.HOVER_RECT_COLOR)
            painter.restore()

            self._draw_info_text(painter, QRect(0, timecode_area_y_start, self.width(),
                                                VideoThumbnailScrubber.INDIVIDUAL_TIMECODE_AREA_HEIGHT))

            playhead_visual_offset_y = VideoThumbnailScrubber.INDIVIDUAL_TIMECODE_AREA_HEIGHT
            if self._current_frame_idx >= 0 and self._current_frame_idx < self._total_frames:
                playhead_x_in_view = ((self._current_frame_idx - first_thumb_to_draw) * \
                                      (self._thumbnail_width + VideoThumbnailScrubber.THUMBNAIL_SPACING)) + \
                                     (self._thumbnail_width // 2)

                if -VideoThumbnailScrubber.PLAYHEAD_TOP_BAR_WIDTH < playhead_x_in_view < self.width() + VideoThumbnailScrubber.PLAYHEAD_TOP_BAR_WIDTH:
                    top_bar_actual_y = playhead_visual_offset_y + VideoThumbnailScrubber.PLAYHEAD_TOP_BAR_OFFSET_Y
                    main_line_y_start = top_bar_actual_y + VideoThumbnailScrubber.PLAYHEAD_TOP_BAR_HEIGHT
                    main_line_y_end = playhead_visual_offset_y + VideoThumbnailScrubber.THUMBNAIL_HEIGHT
                    if VideoThumbnailScrubber.PLAYHEAD_BOTTOM_TRIANGLE_HEIGHT > 0: main_line_y_end -= VideoThumbnailScrubber.PLAYHEAD_BOTTOM_TRIANGLE_HEIGHT

                    painter.setPen(
                        QPen(VideoThumbnailScrubber.PLAYHEAD_LINE_COLOR, VideoThumbnailScrubber.PLAYHEAD_LINE_WIDTH))
                    painter.drawLine(playhead_x_in_view, main_line_y_start, playhead_x_in_view, main_line_y_end)

                    top_bar_rect = QRect(playhead_x_in_view - VideoThumbnailScrubber.PLAYHEAD_TOP_BAR_WIDTH // 2,
                                         top_bar_actual_y,
                                         VideoThumbnailScrubber.PLAYHEAD_TOP_BAR_WIDTH,
                                         VideoThumbnailScrubber.PLAYHEAD_TOP_BAR_HEIGHT)
                    painter.fillRect(top_bar_rect, VideoThumbnailScrubber.PLAYHEAD_TOP_BAR_COLOR)

                    if VideoThumbnailScrubber.PLAYHEAD_BOTTOM_TRIANGLE_HEIGHT > 0:
                        triangle_base_y = playhead_visual_offset_y + VideoThumbnailScrubber.THUMBNAIL_HEIGHT
                        triangle_points = [QPoint(playhead_x_in_view, main_line_y_end),
                                           QPoint(
                                               playhead_x_in_view - VideoThumbnailScrubber.PLAYHEAD_BOTTOM_TRIANGLE_WIDTH // 2,
                                               triangle_base_y),
                                           QPoint(
                                               playhead_x_in_view + VideoThumbnailScrubber.PLAYHEAD_BOTTOM_TRIANGLE_WIDTH // 2,
                                               triangle_base_y)]
                        painter.setBrush(QBrush(VideoThumbnailScrubber.PLAYHEAD_TOP_BAR_COLOR))
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawPolygon(triangle_points)
            painter.end()

        def _draw_info_text(self, painter: QPainter, area_rect: QRect):
            painter.save()
            font = painter.font()
            font.setPointSize(VideoThumbnailScrubber.INFO_TEXT_POINT_SIZE)
            painter.setFont(font)
            painter.setPen(VideoThumbnailScrubber.INFO_TEXT_COLOR)
            fm = painter.fontMetrics()
            text_height = fm.height()
            content_y = area_rect.top() + (area_rect.height() - text_height) // 2
            if self._frame_str_display:
                frame_text_width = fm.horizontalAdvance(self._frame_str_display)
                frame_text_rect = QRect(
                    area_rect.right() - frame_text_width - VideoThumbnailScrubber.INFO_TEXT_PADDING_X,
                    content_y, frame_text_width, text_height)
                painter.drawText(frame_text_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                                 self._frame_str_display)
            painter.restore()

        def mousePressEvent(self, event: QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                frame_idx = self._frame_from_pos(event.position().toPoint())
                if frame_idx != -1:
                    self._is_dragging = True
                    self.frame_selected.emit(frame_idx)
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event: QMouseEvent):
            pos = event.position().toPoint()
            frame_idx = self._frame_from_pos(pos)
            if self._is_dragging and (event.buttons() & Qt.MouseButton.LeftButton):
                if frame_idx != -1 and frame_idx != self._current_frame_idx:
                    self.frame_selected.emit(frame_idx)
            else:
                if frame_idx != self._hover_frame_idx:
                    self._hover_frame_idx = frame_idx
                    self.update()
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event: QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                if self._is_dragging:
                    self._is_dragging = False
                    self.update()
                    # 【修复】：鼠标释放时，强制发送当前帧信号，通知主应用解除锁定并对齐音频
                    self.frame_selected.emit(self._current_frame_idx)
            super().mouseReleaseEvent(event)

        def leaveEvent(self, event: QEvent):
            self._hover_frame_idx = -1
            self.update()
            super().leaveEvent(event)



