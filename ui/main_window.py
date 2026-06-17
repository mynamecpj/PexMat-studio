"""
Image & Video Enhancer Toolbox - Main Application Window
========================================================
A comprehensive PySide6-based GUI application for image/video processing,
integrating SAM 2 (Segment Anything Model 2) for matting and Real-ESRGAN for upscaling.
"""

import ctypes
import os
import sys
import platform
import time
import uuid
import shutil
import functools
import traceback
import collections
import math
import random
import copy
from typing import Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ["PATH"] = PROJECT_ROOT + os.pathsep + os.environ.get("PATH", "")

import mpv
import ctypes
from PySide6.QtCore import QMetaObject, Q_ARG

# PyAV import for frame extraction
try:
    import av
except ImportError:
    av = None

# PyTorch and device configuration
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    device = "cpu"

# Hydra import (used in background computations)
try:
    from hydra.core.global_hydra import GlobalHydra
except ImportError:
    GlobalHydra = None

# PySide6 Imports
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QFileDialog, QMessageBox, QSizePolicy, QProgressDialog,
    QGroupBox, QSlider, QCheckBox, QRadioButton, QGridLayout, QSpacerItem,
    QFrame, QSpinBox, QPlainTextEdit, QScrollArea, QStyle, QToolButton,
    QLineEdit, QColorDialog, QToolBar, QAbstractSpinBox, QComboBox,
    QStackedWidget, QListWidget, QListWidgetItem, QGraphicsPixmapItem,
    QGraphicsScene, QDoubleSpinBox, QGraphicsDropShadowEffect,
    QGraphicsBlurEffect, QButtonGroup, QDialog, QProgressBar, QTabWidget,
    QTreeWidget, QTableWidget, QHeaderView, QGraphicsOpacityEffect
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QCursor, QMouseEvent, QResizeEvent,
    QIcon, QCloseEvent, QKeyEvent, QAction, QKeySequence, QActionGroup,
    QPainterPath, QTransform, QMovie, QMoveEvent, QColor, QFont, QImageReader
)
from PySide6.QtCore import (
    Qt, QPoint, QPointF, QRect, QRectF, QSize, QTimer, QEvent, QThread,
    Signal, QObject, Slot, QSizeF, QPropertyAnimation, QEasingCurve,
    QSettings, QParallelAnimationGroup, QAbstractAnimation,
    QSequentialAnimationGroup, QUrl, QEventLoop
)

# Local Project Imports
from config.settings import *
from config.settings import _TR
from core.models.realesrgan_root import REALESRGAN_AVAILABLE
from core.utils import (
    get_asset_path, convert_cv_to_pixmap, imread_unicode, resize_image_to_max_dim,
    upscale_mask_with_guidance, _static_apply_mask_refinements, imwrite_unicode
)
from core.workers import (
    ImageLoaderWorker, EnhanceWorker, SaveWorker, PredictWorker,
    FrameExtractorWorker, VideoSegmentationPropagateWorker,
    VideoMatAnyoneWorker, ModelLoaderWorker, HeadlessLoader,
    BatchMattingWorker, VideoInitStateWorker, VideoSyncHistoryWorker,
    VideoInteractionWorker
)
from ui.components.basic_widgets import RecentProjectItemWidget, ToggleSwitch
from ui.components.cards import AspectRatioPixmapWidget, TransitionCard, AssetThumbnail, RoundedShadowCard, WelcomeImageCard
from ui.components.dialogs import ProgressDialog, ModernExportDialog
from ui.components.panels import AssetPanelFrame, FloatingPanelContainer, SlidingPanelFrame, PanelMixin
from ui.style import apply_stylesheet
from ui.views.batch_view import BatchMattingPage
from ui.views.canvas_view import StitchedImageItem, StitchingCanvas, CanvasViewMixin
from ui.views.image_view import ImageLabel, ImageCompareWidget, ImageViewMixin
from ui.views.video_view import VideoDisplayLabel, VideoThumbnailScrubber, VID_TEXT_PRIMARY, VideoViewMixin, \
    StoryboardItemWidget, BakeSingleClipWorker
from ui.views.welcome_view import WelcomePageWithCards, WelcomeViewMixin

# Optional local model imports (for lazy evaluation checks / execution)
try:
    from core.models.matanyone2.utils.get_default_model import get_matanyone2_model
except ImportError:
    get_matanyone2_model = None

# --- SAM2 Availability Check ---
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    SAM2_IMAGE_PREDICTOR_AVAILABLE = True
    SAM2_VIDEO_PREDICTOR_AVAILABLE = True
except ImportError:
    SAM2ImagePredictor = build_sam2 = None
    SAM2VideoPredictor = build_sam2_video_predictor = None
    SAM2_IMAGE_PREDICTOR_AVAILABLE = False
    SAM2_VIDEO_PREDICTOR_AVAILABLE = False

SAM2_AVAILABLE = SAM2_IMAGE_PREDICTOR_AVAILABLE or SAM2_VIDEO_PREDICTOR_AVAILABLE

# --- Pillow Availability Check ---
try:
    from PIL import Image as PILImage, ImageSequence
    PILLOW_AVAILABLE = True
except ImportError:
    PILImage = ImageSequence = None
    PILLOW_AVAILABLE = False

# --- OpenCV Contrib Check ---
CV2_CONTRIB_AVAILABLE = hasattr(cv2, 'ximgproc')


class RefinementComputeThread(QThread):
    compute_finished = Signal(object, bool, object, bool)

    def __init__(self, mask_np, image_np, refine_params, is_full_rebuild, dirty_rect_info, mat_model, device_str):
        super().__init__(None)
        self.mask_np = mask_np
        self.image_np = image_np
        self.refine_params = refine_params
        self.is_full_rebuild = is_full_rebuild
        self.dirty_rect_info = dirty_rect_info
        self.mat_model = mat_model
        self.device_str = device_str

    def run(self):
        try:
            result_mask = None
            is_cuda_active = (self.device_str == 'cuda')

            with torch.inference_mode():
                if is_cuda_active:
                    with torch.autocast(device_type='cuda', enabled=True):
                        result_mask = self._do_compute()
                else:
                    result_mask = self._do_compute()

            if result_mask is not None:
                result_mask = np.array(result_mask, copy=True, order='C')

            self.compute_finished.emit(result_mask, self.is_full_rebuild, self.dirty_rect_info, True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.compute_finished.emit(None, self.is_full_rebuild, self.dirty_rect_info, False)

    def _do_compute(self):
        # 1. 准备要推理的图像和蒙版（完美兼容画笔的局部更新/脏矩形机制）
        if self.is_full_rebuild:
            target_image = self.image_np
            target_mask = self.mask_np
        else:
            # 如果是画笔涂抹，只提取涂抹区域的图像块，避免形状不匹配报错 (ValueError: operands could not be broadcast)
            x, y, cw, ch = self.dirty_rect_info
            target_image = self.image_np[y:y + ch, x:x + cw]
            target_mask = self.mask_np[y:y + ch, x:x + cw]

        # 2. 如果开启了发丝大模型，走官方的 MatAnyone2 网页版全图/局部直抠逻辑
        if self.mat_model is not None and self.refine_params.get('refine_matteformer_enabled', False):
            from core.models.matanyone2.inference.inference_core import InferenceCore
            from core.models.matanyone2.matanyone2_wrapper import matanyone2

            actual_device = next(self.mat_model.parameters()).device
            matanyone_processor = InferenceCore(self.mat_model, cfg=self.mat_model.cfg)
            matanyone_processor.device = actual_device

            # 官方网页版逻辑：把图像复制两份，伪装成2帧的视频以激活时序特征
            frames = [target_image, target_image]

            # 官方网页版逻辑：构建 template_mask 并乘以 255 (转为 int32)
            mask_uint8 = (target_mask > 0).astype(np.int32)
            template_mask = mask_uint8 * 255

            # 读取内核参数
            erode_kernel = self.refine_params.get('refine_erode', 10)
            dilate_kernel = self.refine_params.get('refine_dilate', 10)
            # 官方网页版的图像抠图利用了 n_warmup 迭代提升精度
            refine_iter = self.refine_params.get('refine_iter', 10)

            # 执行完全等价的推理
            foreground, alpha = matanyone2(
                matanyone_processor,
                frames,
                template_mask,
                r_erode=erode_kernel,
                r_dilate=dilate_kernel,
                n_warmup=refine_iter
            )

            # 提取最后一帧的 Alpha 遮罩
            alpha_out = alpha[-1]
            if alpha_out.ndim == 3:
                alpha_out = alpha_out[:, :, 0]

            # 将 0-255 的数据转为软件UI画笔支持的 0.0 - 1.0 的高精度软蒙版格式返回
            if alpha_out.dtype == np.uint8:
                return alpha_out.astype(np.float32) / 255.0
            elif torch.is_tensor(alpha_out):
                return alpha_out.detach().cpu().numpy().astype(np.float32) / 255.0
            return alpha_out.astype(np.float32) / 255.0

        # 3. 如果没有勾选发丝大模型，则平滑降级走 SAM2 等传统算法，不去动它
        from core.utils import _static_apply_mask_refinements
        return _static_apply_mask_refinements(
            target_mask, target_image, self.refine_params, self.mat_model, self.device_str
        )


class SetImageWorker(QObject):
    """
    Dedicated background worker class for managing the SAM 2 image feature encoding (set_image) lifecycle.
    """
    finished = Signal(bool, str)

    def __init__(self, predictor, img_rgb):
        super().__init__(None)  # Use None parent to guarantee thread ownership migration safety.
        self.predictor = predictor
        self.img_rgb = img_rgb

    def run(self):
        try:
            with torch.inference_mode():
                self.predictor.set_image(self.img_rgb)
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, f"{e}\n{traceback.format_exc()}")


class ColorSwatch(QPushButton):
    """
    Stylized color swatch with built-in physics-like hover scaling effects.
    """

    def __init__(self, color_hex, parent=None):
        super().__init__(parent)
        self.color = QColor(color_hex)
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.is_selected = False
        self.is_hovered = False

    def enterEvent(self, event):
        self.is_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.is_hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Shrink the ellipse bounds slightly to allow room for hover scaling
        circle_rect = QRectF(3, 3, 30, 30)

        if self.is_hovered and not self.is_selected:
            circle_rect = QRectF(2, 2, 32, 32)

        path = QPainterPath()
        path.addEllipse(circle_rect)
        painter.fillPath(path, self.color)

        # Hover state border hints
        if self.is_hovered and not self.is_selected:
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1.5))
            painter.drawEllipse(circle_rect)
        elif not self.is_selected:
            painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
            painter.drawEllipse(circle_rect)

        # Selection state double ring feedback
        if self.is_selected:
            painter.setPen(QPen(QColor("#262626"), 2))
            painter.drawEllipse(3, 3, 30, 30)
            painter.setPen(QPen(QColor("#FFFFFF"), 2.5))
            painter.drawEllipse(1, 1, 34, 34)


class ModernColorDialog(QDialog):
    """
    High-end minimalist dark mode palette with updated layout.
    """

    def __init__(self, initial_color=QColor("#1A73E8"), parent=None, title="选择颜色"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(290, 340)
        self.current_color = QColor(initial_color)

        self.setStyleSheet("""
            QDialog { 
                background-color: #262626; 
                border-radius: 12px; 
            }
            QLabel { 
                color: #E0E0E0; 
                font-family: "Microsoft YaHei", sans-serif; 
                font-size: 13px; 
                font-weight: bold; 
            }
            QPushButton#BtnPrimary {
                background-color: #1A73E8; 
                color: white;
                border-radius: 6px; 
                padding: 7px 18px; 
                font-weight: bold; 
                font-size: 13px;
                border: none;
            }
            QPushButton#BtnPrimary:hover { 
                background-color: #4C8BF5; 
            }
            QPushButton#BtnPrimary:pressed { 
                background-color: #145DBF; 
            }
            QPushButton#BtnSecondary {
                background-color: transparent; 
                color: #9CA3AF;
                border: 1px solid #555555;
                border-radius: 6px; 
                padding: 7px 18px; 
                font-size: 13px;
            }
            QPushButton#BtnSecondary:hover { 
                background-color: rgba(255, 255, 255, 0.08); 
                color: #FFFFFF; 
                border-color: #888888;
            }
            QPushButton#BtnSecondary:pressed { 
                background-color: rgba(255, 255, 255, 0.15); 
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 16)
        main_layout.setSpacing(16)

        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(12)

        preset_colors = [
            "#EF4444", "#F97316", "#F59E0B", "#EAB308", "#84CC16",
            "#22C55E", "#10B981", "#14B8A6", "#06B6D4", "#0EA5E9",
            "#3B82F6", "#6366F1", "#8B5CF6", "#A855F7", "#D946EF",
            "#EC4899", "#F43F5E", "#FFFFFF", "#9CA3AF", "#000000"
        ]

        self.swatches = []
        row, col = 0, 0
        for hex_code in preset_colors:
            swatch = ColorSwatch(hex_code)
            swatch.clicked.connect(lambda checked=False, c=hex_code: self._on_swatch_clicked(c))
            grid_layout.addWidget(swatch, row, col)
            self.swatches.append(swatch)
            col += 1
            if col > 4:
                col = 0
                row += 1

        main_layout.addWidget(grid_widget)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #333333;")
        main_layout.addWidget(sep)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        preview_container = QHBoxLayout()
        preview_container.setSpacing(8)

        self.preview_lbl = QLabel()
        self.preview_lbl.setFixedSize(24, 24)
        self._update_preview()

        hint_lbl = QLabel("当前色")
        hint_lbl.setStyleSheet("color: #888888; font-weight: normal; font-size: 12px;")

        preview_container.addWidget(self.preview_lbl)
        preview_container.addWidget(hint_lbl)

        bottom_layout.addLayout(preview_container)
        bottom_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("BtnSecondary")
        cancel_btn.clicked.connect(self.reject)

        ok_btn = QPushButton("确定")
        ok_btn.setObjectName("BtnPrimary")
        ok_btn.clicked.connect(self.accept)

        bottom_layout.addWidget(cancel_btn)
        bottom_layout.addWidget(ok_btn)

        main_layout.addLayout(bottom_layout)
        self._update_selection()

    def _on_swatch_clicked(self, hex_code):
        self.current_color = QColor(hex_code)
        self._update_preview()
        self._update_selection()

    def _update_preview(self):
        self.preview_lbl.setStyleSheet(
            f"background-color: {self.current_color.name()}; border-radius: 6px; border: 1px solid #404040;"
        )

    def _update_selection(self):
        for swatch in self.swatches:
            is_match = (swatch.color.name().upper() == self.current_color.name().upper())
            if swatch.is_selected != is_match:
                swatch.is_selected = is_match
                swatch.update()

    def currentColor(self):
        return self.current_color

    @classmethod
    def getColor(cls, initial_color, parent, title):
        dialog = cls(initial_color, parent, title)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.currentColor()
        return QColor()


class DarkDialogEventFilter(QObject):
    """
    Global dialog event interceptor and real-time translator.
    Forces system title bars to dark mode, injects modern hover styles,
    and dynamically translates dialog texts.
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Show:
            if isinstance(obj, (QDialog, QMessageBox, QFileDialog, QColorDialog, QProgressDialog)):
                self._apply_dark_titlebar_to_hwnd(int(obj.winId()))

                if type(obj).__name__ == "ModernColorDialog":
                    return super().eventFilter(obj, event)

                # Dynamically translate titles and text for system-generated dialogs
                title = obj.windowTitle()
                if title:
                    obj.setWindowTitle(_TR(title))

                if isinstance(obj, QMessageBox):
                    text = obj.text()
                    if text:
                        obj.setText(_TR(text))
                    info_text = obj.informativeText()
                    if info_text:
                        obj.setInformativeText(_TR(info_text))
                    for btn in obj.buttons():
                        btn_text = btn.text()
                        if btn_text and not btn_text.startswith("&"):
                            btn.setText(_TR(btn_text))

                elif isinstance(obj, QProgressDialog):
                    label_text = obj.labelText()
                    if label_text:
                        obj.setLabelText(_TR(label_text))
                    cancel_btn = obj.findChild(QPushButton)
                    if cancel_btn and cancel_btn.text():
                        cancel_btn.setText(_TR(cancel_btn.text()))

                dialog_qss = """
                    QDialog, QMessageBox, QFileDialog, QProgressDialog { background-color: #262626; color: #E0E0E0; }
                    QLabel { color: #E0E0E0; background-color: transparent; font-size: 13px; }
                    QPushButton { background-color: #333333; color: #E0E0E0; border: 1px solid #404040; border-radius: 6px; padding: 6px 18px; min-width: 64px; font-size: 13px; font-weight: bold; }
                    QPushButton:hover { background-color: rgba(255, 255, 255, 0.08); border: 1px solid #555555; color: #FFFFFF; }
                    QPushButton:pressed { background-color: #1A73E8; border: 1px solid #1A73E8; color: #FFFFFF; }
                """
                obj.setStyleSheet(dialog_qss)

        return super().eventFilter(obj, event)

    def _apply_dark_titlebar_to_hwnd(self, hwnd_int):
        if sys.platform != "win32":
            return
        try:
            build_number = int(platform.version().split('.')[2])
            if build_number >= 17763:
                attribute = 20 if build_number >= 19041 else 19
                rendering_policy = ctypes.c_int(1)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd_int, attribute, ctypes.byref(rendering_policy),
                                                           ctypes.sizeof(rendering_policy))
            if build_number >= 22000:
                bg_color, text_color = ctypes.c_int(0x00262626), ctypes.c_int(0x00E0E0E0)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd_int, 35, ctypes.byref(bg_color),
                                                           ctypes.sizeof(bg_color))
                ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd_int, 36, ctypes.byref(text_color),
                                                           ctypes.sizeof(text_color))
        except Exception:
            pass


class AsyncImageExportThread(QThread):
    finished_signal = Signal(bool, str)

    def __init__(self, qimage, target_size, save_path, fmt):
        super().__init__()
        self.qimage = qimage
        self.target_size = target_size
        self.save_path = save_path
        self.fmt = fmt

    def run(self):
        try:
            # 在后台线程进行高强度的 4K/8K 缩放
            if self.target_size != self.qimage.size():
                export_image = self.qimage.scaled(
                    self.target_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            else:
                export_image = self.qimage

            # 在后台线程进行高强度的磁盘 I/O 写入
            success = export_image.save(self.save_path, self.fmt, 100 if self.fmt == 'JPG' else 9)
            if success:
                self.finished_signal.emit(True, self.save_path)
            else:
                self.finished_signal.emit(False, "图片编码保存失败")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class RAMPreloaderWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(dict)

    def __init__(self, frame_dir, total_frames):
        super().__init__(None)
        self.frame_dir = frame_dir
        self.total_frames = total_frames

    def run(self):
        import cv2
        import os
        import numpy as np

        # 安全防爆内存机制：最高预热 400 帧（约 13 秒的 30fps 视频，占用不到 2GB 内存）
        # 绝大多数被抠图的素材都在这个范围内，确保普通电脑也不会卡死
        preload_count = min(self.total_frames, 400)
        cache_dict = {}

        for i in range(preload_count):
            path = os.path.join(self.frame_dir, f"{i:05d}.jpg")
            if os.path.exists(path):
                # 避免 cv2.imread 遇到中文路径报错，采用底层 numpy 数据流读取
                stream = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
                if img is not None:
                    # 确保存储连续性，极大提升后续 QImage 的渲染速度
                    if not img.flags['C_CONTIGUOUS']:
                        img = np.ascontiguousarray(img)
                    cache_key = f"global_{self.frame_dir}_{i}"
                    cache_dict[cache_key] = img

            # 每处理 5 帧向主界面汇报一次进度，保持界面活泼
            if i % 5 == 0:
                pct = int((i / preload_count) * 100)
                self.progress.emit(pct, f"正在将画幅预热至高速内存(RAM)... {pct}%")

        # 预热完毕，把装满高清图像的字典交回给主线程
        self.finished.emit(cache_dict)


class PreRenderMattedVideoWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(bool, str)

    def __init__(self, temp_frame_dir, processed_masks, total_frames, target_w, target_h,
                 bg_color, custom_bg_path, bg_is_transparent, temp_render_dir, virtual_timeline=None):
        super().__init__(None)
        self.temp_frame_dir = temp_frame_dir
        self.processed_masks = processed_masks
        self.total_frames = total_frames
        self.target_w = target_w
        self.target_h = target_h
        self.bg_color = bg_color
        self.custom_bg_path = custom_bg_path
        self.bg_is_transparent = bg_is_transparent
        self.temp_render_dir = temp_render_dir
        self.virtual_timeline = virtual_timeline
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def _safe_link_or_copy(self, src: str, dst: str):
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.link(src, dst)
        except Exception:
            import shutil
            shutil.copy2(src, dst)

    @Slot()
    def run(self):
        try:
            import os
            import cv2
            import numpy as np
            import shutil
            import time
            import gc

            if os.path.exists(self.temp_render_dir):
                shutil.rmtree(self.temp_render_dir, ignore_errors=True)
            os.makedirs(self.temp_render_dir, exist_ok=True)

            # 1. 映射分段配置并预缓存背景图片
            clip_configs = []
            bg_canvas_cache = {}

            if self.virtual_timeline:
                # 若有时序故事板，则读取独立视频片段的属性
                current_count = 0
                for clip in self.virtual_timeline:
                    clip_configs.append({
                        'start': current_count,
                        'end': current_count + clip['frames'] - 1,
                        'bg_is_transparent': clip.get('bg_is_transparent', False),
                        'bg_image_path': clip.get('bg_image_path', None),
                        'bg_color': clip.get('bg_color', QColor(0, 255, 0))
                    })
                    current_count += clip['frames']

                for idx, cfg in enumerate(clip_configs):
                    path = cfg['bg_image_path']
                    if not cfg['bg_is_transparent'] and path and os.path.exists(path):
                        from core.utils import imread_unicode
                        bg_img = imread_unicode(path, cv2.IMREAD_COLOR)
                        if bg_img is not None:
                            bg_canvas_cache[idx] = cv2.resize(bg_img, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            else:
                # 兼容旧逻辑退化通道：若无故事板，使用传入的全局变量生成单一分段
                clip_configs.append({
                    'start': 0,
                    'end': self.total_frames - 1,
                    'bg_is_transparent': self.bg_is_transparent,
                    'bg_image_path': self.custom_bg_path,
                    'bg_color': self.bg_color
                })
                if not self.bg_is_transparent and self.custom_bg_path and os.path.exists(self.custom_bg_path):
                    from core.utils import imread_unicode
                    bg_img = imread_unicode(self.custom_bg_path, cv2.IMREAD_COLOR)
                    if bg_img is not None:
                        bg_canvas_cache[0] = cv2.resize(bg_img, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

            # 2. 逐帧混合渲染
            for i in range(self.total_frames):
                if self._is_cancelled:
                    self.finished.emit(False, "Cancelled")
                    return

                out_path = os.path.join(self.temp_render_dir, f"{i:05d}.jpg")
                frame_path = os.path.join(self.temp_frame_dir, f"{i:05d}.jpg")
                if not os.path.exists(frame_path):
                    frame_path = os.path.join(self.temp_frame_dir, f"{i}.jpg")

                frame_masks = self.processed_masks.get(i, {})
                has_mask = False

                for mask_raw in frame_masks.values():
                    if mask_raw is not None and np.any(mask_raw):
                        has_mask = True
                        break

                # 寻找当前帧对应的片段配置
                cfg_idx = -1
                active_cfg = None
                for idx, cfg in enumerate(clip_configs):
                    if cfg['start'] <= i <= cfg['end']:
                        active_cfg = cfg
                        cfg_idx = idx
                        break
                if active_cfg is None:
                    active_cfg = {'bg_is_transparent': False, 'bg_image_path': None, 'bg_color': QColor(0, 255, 0)}

                bg_is_transparent = active_cfg['bg_is_transparent']
                bg_color = active_cfg['bg_color']

                # 无遮罩且未修改背景时，执行极速硬链接
                is_default_bg = (not bg_is_transparent and active_cfg['bg_image_path'] is None and bg_color == QColor(0, 255, 0))
                if not has_mask and is_default_bg:
                    self._safe_link_or_copy(frame_path, out_path)
                    if i % 50 == 0 or i == self.total_frames - 1:
                        pct = int((i / self.total_frames) * 100)
                        self.progress.emit(pct, f"正在急速重构流畅轨道... ({i + 1}/{self.total_frames})")
                    continue

                from core.utils import imread_unicode, imwrite_unicode
                frame_cv = imread_unicode(frame_path, cv2.IMREAD_COLOR)
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

                if bg_is_transparent:
                    checker_size = 20
                    y_indices = (np.arange(h) // checker_size) % 2
                    x_indices = (np.arange(w) // checker_size) % 2
                    grid_mask = (y_indices[:, None] == x_indices[None, :])
                    bg_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    bg_canvas[grid_mask] = [40, 40, 40]
                    bg_canvas[~grid_mask] = [60, 60, 60]
                elif cfg_idx in bg_canvas_cache:
                    bg_canvas = bg_canvas_cache[cfg_idx].copy()
                else:
                    bg_bgr = (bg_color.blue(), bg_color.green(), bg_color.red())
                    bg_canvas = np.full((h, w, 3), bg_bgr, dtype=np.uint8)

                alpha_3d = combined_alpha[:, :, np.newaxis]
                blended_frame = frame_cv.astype(np.float32) * alpha_3d + bg_canvas.astype(np.float32) * (1.0 - alpha_3d)
                frame_cv = np.clip(blended_frame, 0.0, 255.0).astype(np.uint8)

                imwrite_unicode(out_path, frame_cv)

                del combined_alpha, alpha_3d, blended_frame
                if not (bg_is_transparent or cfg_idx in bg_canvas_cache):
                    del bg_canvas

                if i % 5 == 0 or i == self.total_frames - 1:
                    pct = int((i / self.total_frames) * 100)
                    self.progress.emit(pct, f"正在进行 Alpha 发丝融合... ({i + 1}/{self.total_frames})")

                if i % 30 == 0:
                    gc.collect()

            self.finished.emit(True, self.temp_render_dir)
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(False, str(e))




class ImageEnhancerApp(QMainWindow, PanelMixin, WelcomeViewMixin, VideoViewMixin, CanvasViewMixin, ImageViewMixin):
    """
    Main application window managing UI routing, state, machine learning models, and threaded operations.
    """
    status_update_request = Signal(str, int)
    log_message = Signal(str)

    __version__ = "1.0.0"

    WELCOME_PAGE_INDEX = 0
    VIDEO_SEG_PAGE_INDEX = 1
    CREATIVE_WORKSHOP_INDEX = 2
    BATCH_MATTING_INDEX = 3

    def __init__(self, image_predictor=None, video_predictor=None, mat_model=None, parent=None):
        super().__init__(parent)

        self._init_infrastructure()
        self._init_models_and_state(image_predictor, video_predictor, mat_model)
        self._init_floating_panels_and_timers()
        self._init_ui()
        self._init_post_startup()

    def _init_infrastructure(self):
        """Initialize styling, logging, global filters, settings, and audio engines."""
        apply_stylesheet(QApplication.instance())
        settings_path = os.path.join(self.get_app_data_path(), "settings.ini")
        self.settings = QSettings(settings_path, QSettings.Format.IniFormat)

        self.setWindowTitle("PexMat-studio")
        self.setGeometry(50, 50, 1800, 950)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self._set_window_icon()
        self._apply_dark_titlebar()
        self.statusBar().hide()

        self.dark_dialog_filter = DarkDialogEventFilter()
        QApplication.instance().installEventFilter(self.dark_dialog_filter)
        QApplication.instance().installEventFilter(self)
        QApplication.instance().applicationStateChanged.connect(self._handle_application_state_change)
        self._ignore_focus_changes = False

        # ==========================================
        # [核心优化 1]：废弃 QMediaPlayer，引入 mpv 作为底层音频和主时钟引擎
        # vo='null' 和 video='no' 表示我们只要 mpv 的精准音频引擎和时钟，画面由我们自己带遮罩渲染
        # ==========================================
        self.mpv_audio = mpv.MPV(vo='null', video='no', gapless_audio='yes', keep_open='yes')
        self.mpv_bgm = mpv.MPV(vo='null', video='no', gapless_audio='yes', keep_open='yes')

        self._is_global_muted = False
        self._current_playing_clip_idx = -1

        # 注册高精度时钟回调：由 mpv 的 C 语言底层驱动，彻底杜绝 QTimer 抖动
        @self.mpv_audio.property_observer('time-pos')
        def _on_mpv_time_update(name, value):
            if value is not None and getattr(self, 'is_playing', False):
                # 线程安全地调用主线程进行画面刷新
                QMetaObject.invokeMethod(self, "_sync_frame_to_audio_clock",
                                         Qt.ConnectionType.QueuedConnection,
                                         Q_ARG(float, value))

    def get_app_data_path(self) -> str:
        """
        Cross-platform application data directory resolution.
        Resolves the appropriate settings folder depending on the OS platform.
        """
        app_name = "ImageVideoToolbox"
        if os.name == 'nt':
            # Windows: APPDATA path
            app_data = os.environ.get('APPDATA')
            if not app_data:
                app_data = os.path.expanduser('~')
            path = os.path.join(app_data, app_name)
        else:
            # Linux/macOS: ~/.config path
            path = os.path.join(os.path.expanduser('~'), '.config', app_name)

        os.makedirs(path, exist_ok=True)
        return path

    @Slot()
    def _toggle_global_mute(self):
        """【MPV版】全局静音控制"""
        self._is_global_muted = not getattr(self, '_is_global_muted', False)

        if hasattr(self, 'mpv_audio'):
            orig_vol = 1.0
            bgm_vol = 1.0
            current_local_mute = False

            if getattr(self, '_current_playing_clip_idx', -1) != -1 and hasattr(self, 'virtual_timeline'):
                clip = self.virtual_timeline[self._current_playing_clip_idx]
                current_local_mute = clip.get('mute_original', False)
                orig_vol = clip.get('original_audio_volume', 1.0)
                bgm_vol = clip.get('custom_audio_volume', 1.0)

            total_mute = self._is_global_muted or current_local_mute

            # MPV 音量范围是 0 到 100
            self.mpv_audio.volume = 0 if total_mute else int(orig_vol * 100)
            self.mpv_bgm.volume = 0 if self._is_global_muted else int(bgm_vol * 100)

        icon_name = "volume-mute-fill.svg" if self._is_global_muted else "volume-up-fill.svg"
        if hasattr(self, 'btn_global_mute'):
            self.btn_global_mute.setIcon(self._create_svg_icon(icon_name, color=VID_TEXT_PRIMARY))
        if hasattr(self, 'btn_crop_global_mute'):
            self.btn_crop_global_mute.setIcon(self._create_svg_icon(icon_name, color=VID_TEXT_PRIMARY))

    def _apply_dark_titlebar(self):
        """Forces the operating system title bar to render in dark mode (Windows only)."""
        if sys.platform == "win32":
            try:
                hwnd = int(self.winId())
                build_number = int(platform.version().split('.')[2])

                if build_number >= 17763:
                    attribute = 20 if build_number >= 19041 else 19
                    rendering_policy = ctypes.c_int(1)
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, attribute, ctypes.byref(rendering_policy), ctypes.sizeof(rendering_policy)
                    )

                if build_number >= 22000:
                    DWMWA_CAPTION_COLOR = 35
                    bg_color = ctypes.c_int(0x00181818)
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(bg_color), ctypes.sizeof(bg_color)
                    )

                    DWMWA_TEXT_COLOR = 36
                    text_color = ctypes.c_int(0x00E0E0E0)
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, DWMWA_TEXT_COLOR, ctypes.byref(text_color), ctypes.sizeof(text_color)
                    )

            except Exception as e:
                self.log_message.emit(f"Failed to apply dark title bar: {e}")

    def _init_models_and_state(self, image_predictor, video_predictor, mat_model):
        """Initialize global application properties and ML predictor state flags."""
        self.image_predictor = image_predictor
        self.image_predictor_loaded = bool(image_predictor)
        self.sam_image_load_failed = not bool(image_predictor)

        self.video_predictor = video_predictor
        self.video_predictor_loaded = bool(video_predictor)
        self.sam_video_load_failed = not bool(video_predictor)

        self.matteformer_model = mat_model
        self.matteformer_loaded = bool(mat_model)

        self.is_loading_model = False
        self.image_set_in_predictor = False
        self.is_predicting = False
        self.sam_prediction_cache = {}
        self._current_predict_cache_key = None
        self._current_prediction_cumulative = False
        self.active_workers = {}
        self.image_loading_tasks = {}
        self._is_in_transition = False
        self._is_in_preview_transition = False
        self.use_blur_animation = True
        self.is_in_quick_segment_flow = False
        self.is_saving = False
        self._last_combine_timestamp = 0
        self._is_color_dialog_open = False
        self.welcome_page_blurred_pixmap = None

        self.video_path = None
        self.temp_frame_dir = None
        self.total_frames = 0
        self.video_fps = VIDEO_DEFAULT_FPS
        self.video_width = 0
        self.video_height = 0
        self.current_frame_index = -1
        self.is_gif_input = False
        self.gif_frame_duration_ms = int(1000 / VIDEO_DEFAULT_FPS)
        self.gif_frame_durations_ms = []
        self.video_thumbnail_paths = []
        self.is_playing = False
        self.target_points = {}
        self.current_target_id = -1
        self.next_target_id = 0
        self.processed_masks = {}
        self.video_segmentation_running = False
        self.video_segmentation_finished = False
        self.video_segmentation_saved = False
        self.video_inference_state = None
        self.video_save_bg_color = DEFAULT_VIDEO_BG_COLOR
        self.current_propagate_video_progress_dialog = None
        self.is_extracting_frames = False

        self.is_in_segmentation_overlay_mode = False
        self.item_being_segmented = None
        self.item_being_enhanced = None
        self.preserved_seg_controls_state = None
        self.user_assets_config_path = os.path.join(self.get_app_data_path(), "user_assets.cfg")
        self.stitch_asset_grid_widget = None
        self.stitch_asset_grid_layout = None
        self.current_mask_color_name = next(iter(MASK_COLORS))
        self.selected_mask_color = MASK_COLORS.get(DEFAULT_MASK_COLOR_NAME, QColor(60, 120, 220))
        self.color_buttons = {}
        self.selected_color_button_name = DEFAULT_MASK_COLOR_NAME
        self.segmentation_working_resolution_mode = "1280px"
        self.segmentation_custom_max_dim = 1280
        self.segment_image_path = None
        self.is_enhancing = False
        self.last_enhancement_scale = 4

        self._pending_refinement_values = {}
        self._refinement_update_timer = QTimer(self)
        self._refinement_update_timer.setSingleShot(True)
        self._refinement_update_timer.setInterval(150)
        self._refinement_update_timer.timeout.connect(self._apply_pending_refinements)

    def _init_ui(self):
        """Construct primary UI widgets, layout stacks, and default actions."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.root_layout = QHBoxLayout(self.central_widget)
        self.root_layout.setContentsMargins(10, 10, 10, 10)
        self.root_layout.setSpacing(5)

        self.stacked_widget = QStackedWidget()
        self.root_layout.addWidget(self.stacked_widget, 1)

        self._create_internal_actions()
        self._setup_global_actions()

        self.setup_welcome_page()
        self.setup_video_segment_page()
        self.setup_creative_workshop_page()

        self.batch_matting_page = BatchMattingPage(self)
        self.stacked_widget.addWidget(self.batch_matting_page)

        self._create_rotate_cursor()

    def _init_post_startup(self):
        """Execute post-initialization tasks after the main window is exposed."""
        self.stacked_widget.setCurrentIndex(self.WELCOME_PAGE_INDEX)
        self._load_settings()
        self.update_ui_for_page_change(self.WELCOME_PAGE_INDEX)
        self.update_button_states()

        QTimer.singleShot(0, lambda: (
            self.resizeEvent(QResizeEvent(self.size(), self.size())),
            setattr(self, '_last_main_window_pos', self.pos()),
            self.populate_recent_projects(),
            self._start_default_enforcer()
        ))

    def _start_default_enforcer(self):
        """Start background daemon timer to verify default UI state variables."""
        self._enforce_timer = QTimer(self)
        self._enforce_timer.setInterval(100)
        self._enforce_timer.timeout.connect(self._enforce_defaults_tick)
        self._enforce_timer.start()

    def _create_rotate_cursor(self):
        """
        Draws and initializes a custom rotation cursor with visual angle hints
        used during item rotation interactions in the canvas view.
        """
        rotate_pixmap = QPixmap(32, 32)
        rotate_pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(rotate_pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        center = QPointF(16, 16)
        radius = 11
        pen = QPen(QColor("#505050"), 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)

        arc_path = QPainterPath()
        rect_for_arc = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        start_angle_deg, span_angle_deg = 45, -270
        arc_path.arcMoveTo(rect_for_arc, start_angle_deg)
        arc_path.arcTo(rect_for_arc, start_angle_deg, span_angle_deg)
        p.drawPath(arc_path)

        end_point_on_arc = arc_path.pointAtPercent(1.0)
        tangent_angle_at_end = arc_path.angleAtPercent(1.0)

        arrow_path_cursor = QPainterPath()
        arrow_path_cursor.moveTo(0, 0)
        arrow_path_cursor.lineTo(-8.0, -4.0)
        arrow_path_cursor.lineTo(-8.0, 4.0)
        arrow_path_cursor.closeSubpath()

        p.save()
        p.translate(end_point_on_arc)
        p.rotate(-tangent_angle_at_end)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#505050"))
        p.drawPath(arrow_path_cursor)
        p.restore()
        p.end()

        self.rotate_cursor = QCursor(rotate_pixmap, 16, 16)

    @Slot(int, int)
    def on_stitch_canvas_resized(self, width: int, height: int):
        """
        Slot connected to the stitching canvas resizing signals.
        Synchronizes external canvas dimension changes back into the UI spinboxes.
        """
        if hasattr(self, 'stitch_canvas_width_spin') and hasattr(self, 'stitch_canvas_height_spin'):
            self.stitch_canvas_width_spin.blockSignals(True)
            self.stitch_canvas_height_spin.blockSignals(True)

            self.stitch_canvas_width_spin.setValue(width)
            self.stitch_canvas_height_spin.setValue(height)

            self.stitch_canvas_width_spin.blockSignals(False)
            self.stitch_canvas_height_spin.blockSignals(False)

    def _enforce_defaults_tick(self):
        """Regularly verify checkbox/layout dependencies and stop once default configs are successfully verified."""
        sam_ok = getattr(self, '_cumulative_sam_forced', False)
        vid_ok = getattr(self, '_vid_matteformer_forced', False)

        if not sam_ok and hasattr(self, 'cumulative_sam_checkbox'):
            self.cumulative_sam_checkbox.setChecked(True)
            self._cumulative_sam_forced = True
            sam_ok = True

        if not vid_ok and hasattr(self, 'vid_matteformer_checkbox'):
            self.vid_matteformer_checkbox.setChecked(True)
            try:
                self.vid_matteformer_checkbox.toggled.connect(self.update_button_states)
            except Exception:
                pass
            self._vid_matteformer_forced = True
            vid_ok = True

        if sam_ok and vid_ok:
            self._enforce_timer.stop()
            self._enforce_timer.deleteLater()

    def get_current_device(self):
        """Resolve current hardware compute target dynamically."""
        use_gpu = self.settings.value("hardware/use_gpu", True, type=bool)
        if use_gpu:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
        return torch.device("cpu")

    def show_refinement_progress_overlay(self, message: str):
        """Renders an animated micro-interaction progress overlay over the current work area."""
        if not hasattr(self, 'refinement_overlay_widget') or self.refinement_overlay_widget is None:
            self.refinement_overlay_widget = QWidget(self.workspace_stack)
            self.refinement_overlay_widget.setObjectName("RefinementOverlay")

        if hasattr(self, 'refinement_status_label') and self.refinement_status_label:
            self.refinement_status_label.hide()
        if hasattr(self, 'loading_spinner_label') and self.loading_spinner_label:
            self.loading_spinner_label.hide()

        self.refinement_overlay_widget.setStyleSheet("background: rgba(0, 0, 0, 0.45); border-radius: 15px;")

        if hasattr(self, '_diagonal_pet_overlay') and self._diagonal_pet_overlay is not None:
            self._diagonal_pet_overlay.stop()
            self._diagonal_pet_overlay.deleteLater()
            self._diagonal_pet_overlay = None

        class DiagonalPetOverlay(QWidget):
            def __init__(self, parent_widget=None):
                super().__init__(parent_widget)
                self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                self._offset = 0.0
                self._timer = QTimer(self)
                self._timer.timeout.connect(self._animate)

                self._font = QFont("Segoe UI Emoji", 34)
                self._text_font = QFont("Microsoft YaHei", 18, QFont.Weight.Bold)
                self.message_text = "正在处理..."

                self._pets = []
                self._generate_pets(count=30)

            def _generate_pets(self, count):
                self._pets.clear()
                for _ in range(count):
                    self._pets.append({
                        'char': random.choice(["🐱", "🐰", "✨", "⏳", "🚀"]),
                        'x_ratio': random.uniform(-0.1, 1.1),
                        'y_ratio': random.uniform(-0.1, 1.1),
                        'fade_phase': random.uniform(0, math.pi * 2),
                        'fade_speed': random.uniform(0.5, 0.9),
                        'speed_x': random.uniform(0.0006, 0.0012),
                        'speed_y': random.uniform(0.0006, 0.0012)
                    })

            def start(self):
                self.show()
                self._timer.start(30)

            def stop(self):
                self._timer.stop()
                self.hide()

            def _animate(self):
                self._offset += 0.02
                for pet in self._pets:
                    pet['x_ratio'] += pet['speed_x']
                    pet['y_ratio'] += pet['speed_y']
                    if pet['x_ratio'] > 1.15 or pet['y_ratio'] > 1.15:
                        if random.random() > 0.5:
                            pet['x_ratio'] = random.uniform(-0.15, -0.05)
                            pet['y_ratio'] = random.uniform(-0.1, 0.8)
                        else:
                            pet['x_ratio'] = random.uniform(-0.1, 0.8)
                            pet['y_ratio'] = random.uniform(-0.15, -0.05)
                self.update()

            def paintEvent(self, event):
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                w = self.width()
                h = self.height()

                painter.setFont(self._font)
                for pet in self._pets:
                    sine_val = math.sin(self._offset * pet['fade_speed'] + pet['fade_phase'])
                    if sine_val > 0:
                        alpha_val = int(sine_val * 160)
                        abs_x = w * pet['x_ratio']
                        abs_y = h * pet['y_ratio']
                        painter.setPen(QColor(255, 255, 255, alpha_val))
                        painter.drawText(QRectF(abs_x - 30, abs_y - 30, 60, 60), Qt.AlignmentFlag.AlignCenter, pet['char'])

                painter.setFont(self._text_font)
                text_alpha = int(180 + 75 * math.sin(self._offset * 4))
                painter.setPen(QColor(255, 255, 255, text_alpha))
                painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, self.message_text)
                painter.end()

        self._diagonal_pet_overlay = DiagonalPetOverlay(self.refinement_overlay_widget)
        self._diagonal_pet_overlay.message_text = message

        if hasattr(self, 'segmentation_overlay_label'):
            self.refinement_overlay_widget.setGeometry(self.segmentation_overlay_label.geometry())
            self._diagonal_pet_overlay.setGeometry(self.refinement_overlay_widget.rect())

        self.refinement_overlay_widget.show()
        self.refinement_overlay_widget.raise_()
        self._diagonal_pet_overlay.start()

        QApplication.processEvents()

    def show_global_loading_overlay(self, message: str, percentage: int = -1):
        """Displays a floating progress card in the center of the window (no full-screen background)."""
        # 为了兼容代码中各处的 .hide() 调用，保留原变量名 _global_loading_overlay
        if not hasattr(self, '_global_loading_overlay') or not self._global_loading_overlay:
            # 放弃全屏透明背景，直接把它当作一个纯粹的居中小卡片
            self._global_loading_overlay = QWidget(self)
            self._global_loading_overlay.setObjectName("GlobalLoadingOverlay")

            # 卡片尺寸策略，允许上下自由膨胀以适应多行文字
            self._global_loading_overlay.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,
                                                       QSizePolicy.Policy.MinimumExpanding)
            self._global_loading_overlay.setMinimumWidth(360)

            # 直接把原先深色磨砂卡片的样式赋给它
            self._global_loading_overlay.setStyleSheet("""
                QWidget#GlobalLoadingOverlay {
                    background-color: #1C1C1E;
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-radius: 16px;
                }
            """)

            # 拦截卡片自身的鼠标事件，防止点穿卡片区域
            self._global_loading_overlay.mousePressEvent = lambda event: event.accept()
            self._global_loading_overlay.mouseReleaseEvent = lambda event: event.accept()
            self._global_loading_overlay.mouseMoveEvent = lambda event: event.accept()

            # 高级阴影效果，让它看起来是悬浮的
            shadow = QGraphicsDropShadowEffect(self._global_loading_overlay)
            shadow.setBlurRadius(45)
            shadow.setColor(QColor(0, 0, 0, 220))
            shadow.setOffset(0, 12)
            self._global_loading_overlay.setGraphicsEffect(shadow)

            # 内部布局
            card_layout = QVBoxLayout(self._global_loading_overlay)
            card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.setSpacing(16)
            card_layout.setContentsMargins(24, 24, 24, 24)

            self._global_loading_text = QLabel()
            self._global_loading_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._global_loading_text.setWordWrap(True)
            self._global_loading_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
            self._global_loading_text.setMinimumHeight(40)

            # 彻底移除了导致文字被切割的 line-height
            self._global_loading_text.setStyleSheet("""
                QLabel {
                    color: #F3F4F6;
                    font-size: 14px;
                    font-weight: bold;
                    letter-spacing: 0.5px;
                    background: transparent;
                    border: none;
                }
            """)

            self._overlay_spinner = QProgressBar()
            self._overlay_spinner.setTextVisible(False)
            self._overlay_spinner.setFixedHeight(5)
            self._overlay_spinner.setFixedWidth(280)
            self._overlay_spinner.setStyleSheet("""
                QProgressBar {
                    background-color: #2C2C2E;
                    border: none;
                    border-radius: 3px;
                }
                QProgressBar::chunk {
                    background-color: #0A84FF;
                    border-radius: 3px;
                }
            """)

            card_layout.addWidget(self._global_loading_text, 0, Qt.AlignmentFlag.AlignHCenter)
            card_layout.addWidget(self._overlay_spinner, 0, Qt.AlignmentFlag.AlignHCenter)

        # --- 刷新进度与文字 ---
        if percentage < 0:
            self._overlay_spinner.setRange(0, 0)
        else:
            self._overlay_spinner.setRange(0, 100)
            self._overlay_spinner.setValue(percentage)

        self._global_loading_text.setText(message)

        # --- 强制更新尺寸并动态居中 ---
        self._global_loading_text.adjustSize()
        self._global_loading_overlay.adjustSize()

        if self._global_loading_overlay.layout():
            self._global_loading_overlay.layout().activate()

        # 核心修改：由于取消了全屏遮罩布局，改为利用数学计算，将卡片绝对居中移动
        center_x = (self.width() - self._global_loading_overlay.width()) // 2
        center_y = (self.height() - self._global_loading_overlay.height()) // 2
        self._global_loading_overlay.move(center_x, center_y)

        self._global_loading_overlay.show()
        self._global_loading_overlay.raise_()

        # 现在仅仅重绘这个小尺寸的不透明卡片，彻底消除了大面积半透明导致的闪烁撕裂感
        self._global_loading_overlay.repaint()
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    @Slot()
    def _update_pet_text(self):
        """Update loading indicator labels during computation."""
        if hasattr(self, '_pet_label') and hasattr(self, '_dot_count'):
            self._dot_count = (self._dot_count + 1) % 4
            dots = "." * self._dot_count
            spaces = " " * (3 - self._dot_count)
            pet_str = "😸🐇" if self._dot_count % 2 == 0 else "🐱🐰"
            self._pet_label.setText(f"{pet_str} 极速解算中{dots}{spaces}")

    @Slot()
    def hide_refinement_progress_overlay(self):
        """Stops animation timers and hides refinement status UI overlays."""
        if hasattr(self, '_diagonal_pet_overlay') and self._diagonal_pet_overlay is not None:
            self._diagonal_pet_overlay.stop()

        if hasattr(self, 'refinement_overlay_widget') and self.refinement_overlay_widget is not None:
            self.refinement_overlay_widget.hide()

    @Slot(bool)
    def _on_work_resolution_mode_changed(self, checked):
        """Handles working resolution ratio toggle event, adjusting boundaries."""
        if not checked:
            return

        sender = self.sender()
        new_mode = ""

        if sender == getattr(self, 'res_original_radio', None):
            new_mode = "original"
        elif sender == getattr(self, 'res_512_radio', None):
            new_mode = "512px"
        elif sender == getattr(self, 'res_768_radio', None):
            new_mode = "768px"
        elif sender == getattr(self, 'res_1280_radio', None):
            new_mode = "1280px"
        elif sender == getattr(self, 'res_1920_radio', None):
            new_mode = "1920px"
        elif sender == getattr(self, 'res_custom_radio', None):
            new_mode = "custom"

        if new_mode and getattr(self, 'segmentation_working_resolution_mode', '') == new_mode:
            self._update_work_resolution_controls_state()
            return

        if new_mode:
            self.segmentation_working_resolution_mode = new_mode
            self._update_work_resolution_controls_state()
            self.log_message.emit(f"Working resolution changed to: {new_mode}")

    @Slot()
    def _apply_and_reload_segmentation_image_with_new_resolution(self):
        """Remaps the active mask to the newly selected working resolution limit."""
        source_data = getattr(self, 'current_seg_source_image_cv', None)
        if source_data is None:
            return

        if not hasattr(self, 'segmentation_overlay_label'):
            return

        img_label = self.segmentation_overlay_label
        self._save_segmentation_controls_state()

        if self.segmentation_working_resolution_mode == "custom":
            self.segmentation_custom_max_dim = self.custom_max_dim_spinbox.value()

        existing_mask = img_label._cached_refined_mask.copy() if img_label._cached_refined_mask is not None else (
            img_label.current_mask.copy() if img_label.current_mask is not None else None
        )

        self.show_global_loading_overlay("正在重建全新分辨率下的 AI 特征图，请稍候...", -1)

        try:
            self._load_image_for_segmentation(
                source_image_data=source_data,
                existing_mask_to_scale=existing_mask,
                is_resolution_change=True,
                target_label=img_label
            )
        except Exception:
            traceback.print_exc()
        finally:
            QTimer.singleShot(0, self._restore_segmentation_controls_state)

    def _restore_segmentation_controls_state(self):
        if not hasattr(self, 'preserved_seg_controls_state') or self.preserved_seg_controls_state is None:
            return

        state = self.preserved_seg_controls_state
        widgets_to_manage = [
            self.res_original_radio, self.res_1280_radio, self.res_1920_radio, self.res_custom_radio,
            self.custom_max_dim_spinbox, self.cumulative_sam_checkbox, self.brush_slider,
            self.paint_render_mode_checkbox, self.smooth_slider, self.feather_slider, self.shift_slider,
            self.matteformer_checkbox, self.seg_sam_mode_tool, self.seg_paint_mode_tool
        ]

        for widget in widgets_to_manage:
            if hasattr(self, widget.objectName()):
                widget.blockSignals(True)

        try:
            self.segmentation_working_resolution_mode = state.get('resolution_mode', '1280px')
            self._update_work_resolution_controls_state()
            self.custom_max_dim_spinbox.setValue(state.get('custom_dim', 1280))
            self.cumulative_sam_checkbox.setChecked(state.get('cumulative_sam', True))
            self.brush_slider.setValue(state.get('brush_size', 15))
            self.paint_render_mode_checkbox.setChecked(state.get('paint_render_mode', True))
            self.smooth_slider.setValue(state.get('smooth', DEFAULT_REFINE_SMOOTH))
            self.feather_slider.setValue(state.get('feather', DEFAULT_REFINE_FEATHER))
            self.shift_slider.setValue(state.get('shift', int(DEFAULT_REFINE_SHIFT_SLIDER)))
            self.matteformer_checkbox.setChecked(state.get('refine_matteformer_enabled', False))

            is_sam_mode = state.get('interaction_mode', 'sam') == 'sam'
            self.seg_sam_mode_tool.setChecked(is_sam_mode)
            self.seg_paint_mode_tool.setChecked(not is_sam_mode)

            if hasattr(self, 'segmentation_overlay_label'):
                self.segmentation_overlay_label.set_interaction_mode('sam' if is_sam_mode else 'paint')

            self.selected_color_button_name = state.get('mask_color_name', DEFAULT_MASK_COLOR_NAME)
            self.selected_mask_color = MASK_COLORS.get(self.selected_color_button_name,
                                                       MASK_COLORS[DEFAULT_MASK_COLOR_NAME])

            if hasattr(self, '_update_color_preview_button_style'):
                self._update_color_preview_button_style()

        finally:
            for widget in widgets_to_manage:
                if hasattr(self, widget.objectName()):
                    widget.blockSignals(False)

        active_label = getattr(self, 'segmentation_overlay_label', None)
        if active_label:
            params_to_sync = {
                'refine_smooth': self.smooth_slider.value(),
                'refine_feather': self.feather_slider.value(),
                'refine_shift': int(round(self.shift_slider.value() / SHIFT_SLIDER_FACTOR)),
                'refine_matteformer_enabled': self.matteformer_checkbox.isChecked()
            }
            active_label.set_refinement_params_batch(params_to_sync)

        self.update_brush_size(self.brush_slider.value())
        self.on_paint_render_mode_changed(self.paint_render_mode_checkbox.isChecked())
        self._update_segmentation_settings_panel_visibility()
        self.update_button_states()
        self.preserved_seg_controls_state = None

    @Slot(str, str)
    def start_async_image_load(self, file_path, item_id):
        """Asynchronously loads full images for canvas rendering elements."""
        if item_id in self.image_loading_tasks:
            return

        thread = QThread(self)
        worker = ImageLoaderWorker(file_path, item_id)
        worker.moveToThread(thread)

        worker.image_loaded.connect(self._handle_async_image_loaded)
        worker.image_loaded.connect(thread.quit)

        def make_loader_cleanup(t, w, iid):
            def cleanup():
                try:
                    w.moveToThread(QApplication.instance().thread())
                    w.deleteLater()
                except Exception:
                    pass
                try:
                    t.deleteLater()
                except Exception:
                    pass
            return cleanup

        thread.finished.connect(make_loader_cleanup(thread, worker, item_id))
        thread.started.connect(worker.run)
        thread.start()

        self.image_loading_tasks[item_id] = (thread, worker)

    @Slot(QPixmap, str, str, str)
    def _handle_async_image_loaded(self, pixmap, file_path, item_id, error_message):
        if item_id in self.image_loading_tasks:
            del self.image_loading_tasks[item_id]

        if not error_message and not pixmap.isNull():
            if hasattr(self, 'stitching_canvas'):
                self.stitching_canvas.update_item_pixmap(item_id, pixmap)
        else:
            self.log_message.emit(f"Failed to load image for item ID '{item_id}': {error_message}")
            if hasattr(self, 'stitching_canvas'):
                self.stitching_canvas.handle_item_load_error(item_id)

    @Slot()
    def save_segmentation_from_overlay(self):
        """图像导出：支持分辨率选择，使用异步线程防止4K卡死"""
        if not self.is_in_segmentation_overlay_mode or not hasattr(self, 'segmentation_overlay_label'):
            return

        source_label = self.segmentation_overlay_label
        if source_label.current_mask is None or source_label.original_cv_image_full_res is None:
            QMessageBox.warning(self, _TR("保存错误"), _TR("没有有效的抠图结果。"))
            return

        final_mask, final_pixmap = self._get_final_segmentation_result(source_label)
        if final_mask is None or final_pixmap is None:
            return

        base_name = os.path.basename(self.segment_image_path) if self.segment_image_path else "segmented_image.png"

        # 【核心修改】：同名检测循环机制
        while True:
            dialog = ModernExportDialog('image', final_pixmap.size(), base_name, parent=self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            params = dialog.get_export_params()
            save_path = params['path']

            # 判断文件是否已存在
            if os.path.exists(save_path):
                reply = QMessageBox.question(
                    self, _TR("确认覆盖"),
                    _TR("文件 '{}' 已存在。\n\n您确定要覆盖此文件吗？").format(os.path.basename(save_path)),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    break  # 允许覆盖，跳出循环执行导出
                else:
                    # 不覆盖，则将刚才输入的名称作为下一次弹出的默认名称
                    base_name = os.path.basename(save_path)
                    continue  # 重新循环，弹出设置窗口
            else:
                break  # 不存在同名文件，直接导出

        # 调出系统处理中遮罩，冻结用户操作
        self.show_global_loading_overlay(_TR("正在进行超高分辨率缩放与编码..."), 0)
        self.is_saving = True
        self.update_button_states()

        # 启动异步线程处理缩放和保存
        self.export_thread = AsyncImageExportThread(final_pixmap.toImage(), params['size'], params['path'],
                                                    params['format'].upper())
        self.export_thread.finished_signal.connect(self._on_async_image_export_finished)
        self.export_thread.finished_signal.connect(self.export_thread.deleteLater)
        self.export_thread.start()

    @Slot()
    def save_enhanced_item_result(self):
        """画布内选中素材的增强结果保存：纯异步防卡死"""
        if not hasattr(self, 'image_compare_widget_enhance'):
            return

        enhanced_pixmap = self.image_compare_widget_enhance.get_enhanced_pixmap()

        if enhanced_pixmap is None or enhanced_pixmap.isNull():
            QMessageBox.warning(self, _TR("信息"), _TR("没有可保存的增强结果。请先应用增强。"))
            return

        if self.is_saving:
            QMessageBox.information(self, _TR("忙碌"), _TR("其他保存操作正在进行中。"))
            return

        base = "enhanced_output"
        if self.item_being_enhanced and self.item_being_enhanced.original_image_path:
            base = os.path.splitext(os.path.basename(self.item_being_enhanced.original_image_path))[0]

        scale_str = self.enhance_scale_combo.currentText()
        default_name = f"{base}_enhanced_{scale_str}.png"

        save_path, _ = QFileDialog.getSaveFileName(self, _TR("保存增强图像 (PNG)"), default_name, "PNG 图像 (*.png)")

        if save_path:
            if not save_path.lower().endswith('.png'):
                save_path += '.png'

            # 呼出全局系统处理中遮罩，冻结用户操作
            self.show_global_loading_overlay(_TR("正在进行超高分辨率图像编码写盘..."), 0)
            self.is_saving = True
            self.update_button_states()

            # 【核心修复】：延迟100毫秒提取图像数据并丢进异步后台线程
            def _do_export():
                try:
                    # 巨幅图像的 toImage() 会瞬间吃满单核，必须放在遮罩完全渲染之后
                    qimage_to_save = enhanced_pixmap.toImage()

                    self.export_thread = AsyncImageExportThread(
                        qimage_to_save,
                        qimage_to_save.size(),
                        save_path,
                        'PNG'
                    )
                    self.export_thread.finished_signal.connect(self._on_async_image_export_finished)
                    self.export_thread.finished_signal.connect(self.export_thread.deleteLater)
                    self.export_thread.start()
                except Exception as e:
                    self._on_async_image_export_finished(False, str(e))

            QTimer.singleShot(100, _do_export)

    def _save_segmentation_controls_state(self):
        if not hasattr(self, 'cumulative_sam_checkbox'):
            self.preserved_seg_controls_state = None
            return

        state = {
            'resolution_mode': self.segmentation_working_resolution_mode,
            'custom_dim': self.custom_max_dim_spinbox.value() if hasattr(self, 'custom_max_dim_spinbox') else 1280,
            'cumulative_sam': self.cumulative_sam_checkbox.isChecked(),
            'brush_size': self.brush_slider.value(),
            'paint_render_mode': self.paint_render_mode_checkbox.isChecked(),
            'smooth': self.smooth_slider.value(),
            'feather': self.feather_slider.value(),
            'shift': self.shift_slider.value(),
            'refine_matteformer_enabled': self.matteformer_checkbox.isChecked(),
            'mask_color_name': self.selected_color_button_name,
            'interaction_mode': self.segmentation_overlay_label.interaction_mode if hasattr(self, 'segmentation_overlay_label') else 'sam'
        }
        self.preserved_seg_controls_state = state

    @Slot(bool)
    def _on_segmentation_mode_tool_toggled(self, checked: bool):
        seg_label = getattr(self, 'segmentation_overlay_label', None)
        if not seg_label:
            return

        new_mode = 'sam' if checked else 'paint'
        seg_label.set_interaction_mode(new_mode)
        self._update_segmentation_settings_panel_visibility()
        self.update_button_states()

    def _update_segmentation_settings_panel_visibility(self):
        """Disables layout components rather than hiding them to avoid layout jumping."""
        if not hasattr(self, 'seg_sam_mode_tool'):
            return

        is_sam_mode = self.seg_sam_mode_tool.isChecked()

        if hasattr(self, 'sam_options_card'):
            self.sam_options_card.setEnabled(is_sam_mode)
            opacity = 1.0 if is_sam_mode else 0.4
            eff1 = QGraphicsOpacityEffect(self)
            eff1.setOpacity(opacity)
            self.sam_options_card.setGraphicsEffect(eff1)

        if hasattr(self, 'paint_options_card'):
            self.paint_options_card.setEnabled(not is_sam_mode)
            opacity = 1.0 if not is_sam_mode else 0.4
            eff2 = QGraphicsOpacityEffect(self)
            eff2.setOpacity(opacity)
            self.paint_options_card.setGraphicsEffect(eff2)

    @Slot()
    def load_stitch_image_action(self):
        if not hasattr(self, 'stitching_canvas'):
            return

        supported_formats = "图像文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;PNG 图像 (*.png);;所有文件 (*)"
        files, _ = QFileDialog.getOpenFileNames(self, "选择要添加到画布的图像 (可多选)", "", supported_formats)

        if files:
            self.stitching_canvas._process_dropped_files(files)

    def _enter_segmentation_mode(self, item_to_segment=None):
        """Enters interactive segmentation view and setups background workspace parameters."""
        self.item_being_segmented = item_to_segment
        if not self.item_being_segmented:
            QMessageBox.warning(self, "错误", "请先在画布上选中一个素材。")
            return

        if not getattr(self, 'image_predictor_loaded', False):
            QMessageBox.warning(self, "模型未就绪", "图像分割模型尚未加载完成，请稍候。")
            return

        self.show_global_loading_overlay("正在进入抠图界面...", -1)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

        try:
            cv_image_full_res = None
            source_pixmap = self.item_being_segmented.pixmap

            if source_pixmap.isNull():
                raise ValueError("选中的素材图像数据无效。")

            existing_mask = getattr(self.item_being_segmented, 'segmentation_mask_np', None)
            original_path = getattr(self.item_being_segmented, 'original_image_path', None)

            use_original = False
            if existing_mask is not None and original_path and os.path.exists(original_path):
                temp_img = imread_unicode(original_path, cv2.IMREAD_COLOR)
                if temp_img is not None:
                    if existing_mask.shape[:2] == temp_img.shape[:2]:
                        use_original = True
                        cv_image_full_res = temp_img

            if not use_original:
                source_qimage = source_pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
                ptr = source_qimage.constBits()
                cv_image_bgra = np.array(ptr).reshape(source_qimage.height(), source_qimage.width(), 4).copy()

                alpha = cv_image_bgra[:, :, 3:4] / 255.0
                cv_image_full_res = (cv_image_bgra[:, :, :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)

                if existing_mask is not None and existing_mask.shape[:2] != (source_pixmap.height(), source_pixmap.width()):
                    if np.issubdtype(existing_mask.dtype, np.floating):
                        aligned_mask = cv2.resize(existing_mask, (source_pixmap.width(), source_pixmap.height()),
                                                  interpolation=cv2.INTER_LINEAR)
                    else:
                        aligned_mask = cv2.resize(existing_mask.astype(np.uint8),
                                                  (source_pixmap.width(), source_pixmap.height()),
                                                  interpolation=cv2.INTER_NEAREST)
                    existing_mask = aligned_mask
                    self.item_being_segmented.segmentation_mask_np = existing_mask

            self.is_in_segmentation_overlay_mode = True
            self.current_seg_source_image_cv = cv_image_full_res
            self.segment_image_path = original_path

            if hasattr(self, 'bottom_pill_toolbar'):
                self.bottom_pill_toolbar.hide()

            if hasattr(self, 'segmentation_overlay_label'):
                self.segmentation_overlay_label.clear_all()
                self.segmentation_overlay_label.is_overlay_mode = True

            if hasattr(self, 'workspace_stack'):
                self.workspace_stack.setCurrentWidget(self.segmentation_overlay_label)

            if hasattr(self, 'right_properties_stack'):
                self.right_properties_stack.setCurrentWidget(self.segment_props_widget)

            self._load_image_for_segmentation(
                source_image_data=self.current_seg_source_image_cv,
                existing_mask_to_scale=existing_mask,
                is_resolution_change=True,
                target_label=getattr(self, 'segmentation_overlay_label', None)
            )

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"无法进入抠图界面: {e}")
            self.is_in_segmentation_overlay_mode = False
            self.update_button_states()
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()

    def _get_final_segmentation_result(self, source_label: 'ImageLabel') -> Tuple[Optional[np.ndarray], Optional[QPixmap]]:
        """Projects current working resolution edits back onto the high-resolution source image."""
        if source_label.original_cv_image_full_res is None:
            return None, None

        working_mask = source_label.get_cached_refined_mask()
        if working_mask is None:
            working_mask = source_label.current_mask.astype(np.float32) if source_label.current_mask is not None else None

        if working_mask is None:
            return None, None

        h_orig, w_orig = source_label.original_cv_image_full_res.shape[:2]
        high_res_guide = source_label.original_cv_image_full_res

        subject_strategy = "product"
        if hasattr(self, '_editing_batch_card') and self._editing_batch_card is not None:
            subject_strategy = getattr(self._editing_batch_card, 'strategy', "product")
        elif hasattr(self, 'target_strategy_type'):
            subject_strategy = getattr(self, 'target_strategy_type', "product")

        if working_mask.shape[:2] != (h_orig, w_orig):
            final_mask_full_res = upscale_mask_with_guidance(
                low_res_mask=working_mask,
                high_res_guide=high_res_guide,
                subject_type=subject_strategy
            )
        else:
            final_mask_full_res = np.clip(working_mask, 0.0, 1.0)

        alpha_channel_u8 = (final_mask_full_res * 255.0).clip(0, 255).astype(np.uint8)

        if len(high_res_guide.shape) == 3 and high_res_guide.shape[2] == 4:
            bgr_image = cv2.cvtColor(high_res_guide, cv2.COLOR_BGRA2BGR)
        else:
            bgr_image = high_res_guide

        bgra_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)
        bgra_image[:, :, 3] = alpha_channel_u8

        return final_mask_full_res, convert_cv_to_pixmap(bgra_image)

    def _apply_segmentation_and_exit_mode(self):
        """Applies refined mask transformations to the item object, cropping boundaries gracefully."""
        if not self.is_in_segmentation_overlay_mode or not self.item_being_segmented:
            self._exit_segmentation_mode()
            return

        final_mask_full_res, final_cutout_pixmap = self._get_final_segmentation_result(self.segmentation_overlay_label)

        batch_card = getattr(self, '_editing_batch_card', None)
        if batch_card is not None:
            if final_mask_full_res is not None:
                orig_cv_img = self.segmentation_overlay_label.original_cv_image_full_res
                if orig_cv_img is not None:
                    bgra_cv = cv2.cvtColor(orig_cv_img, cv2.COLOR_BGR2BGRA)
                    bgra_cv[:, :, 3] = (final_mask_full_res * 255).clip(0, 255).astype(np.uint8)
                    batch_card.update_result(bgra_cv, final_mask_full_res > 0.5, "")
            self._exit_segmentation_mode()
            return

        if final_mask_full_res is not None and final_cutout_pixmap is not None:
            item = self.item_being_segmented

            mask_u8_for_contour = (final_mask_full_res > 0.1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(np.concatenate(contours))
                cropped_pixmap = final_cutout_pixmap.copy(x_crop, y_crop, w_crop, h_crop)
                cropped_mask = final_mask_full_res[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
            else:
                x_crop, y_crop, w_crop, h_crop = 0, 0, final_cutout_pixmap.width(), final_cutout_pixmap.height()
                cropped_pixmap = final_cutout_pixmap
                cropped_mask = final_mask_full_res

            orig_pos = item.pos
            orig_size = item.size
            orig_rot = item.rotation

            h_orig, w_orig = final_mask_full_res.shape[:2]

            scale_x = orig_size.width() / w_orig
            scale_y = orig_size.height() / h_orig

            offset_canvas = QPointF(x_crop * scale_x, y_crop * scale_y)

            transform = QTransform()
            transform.rotate(orig_rot)
            rotated_offset = transform.map(offset_canvas)

            item.pixmap = cropped_pixmap
            item.size = QSizeF(w_crop * scale_x, h_crop * scale_y)
            item.pos = orig_pos + rotated_offset
            item.segmentation_mask_np = cropped_mask
            item.has_alpha_channel = True

            self.stitching_canvas.update()
            self.stitching_canvas.selection_changed.emit()

        self._exit_segmentation_mode()

    @Slot()
    def _exit_segmentation_mode(self):
        """Resets panel configurations and exits image matting view mode."""
        self.is_in_segmentation_overlay_mode = False

        if hasattr(self, 'segmentation_overlay_label'):
            self.segmentation_overlay_label.is_overlay_mode = False
            self.segmentation_overlay_label.set_mask_preview_mode(False)

        if hasattr(self, 'seg_preview_tool'):
            self.seg_preview_tool.setChecked(False)

        if hasattr(self, 'bottom_pill_toolbar'):
            self.bottom_pill_toolbar.show()

        if hasattr(self, 'workspace_stack') and hasattr(self, 'stitching_canvas'):
            self.workspace_stack.setCurrentWidget(self.stitching_canvas)

        if hasattr(self, 'right_properties_stack') and hasattr(self, 'canvas_props_widget'):
            self.right_properties_stack.setCurrentWidget(self.canvas_props_widget)

        if hasattr(self, 'stitching_canvas'):
            self.stitching_canvas.update()
            self.stitching_canvas.setFocus()

        if getattr(self, 'is_in_quick_segment_flow', False):
            self.is_in_quick_segment_flow = False
            self.item_being_segmented = None
            if not getattr(self, '_is_in_transition', False):
                self.switch_page_with_slide(self.WELCOME_PAGE_INDEX)
            self.update_button_states()
            return

        batch_card = getattr(self, '_editing_batch_card', None)
        if batch_card is not None:
            if self.item_being_segmented in self.stitching_canvas.items:
                self.stitching_canvas.items.remove(self.item_being_segmented)
            self.stitching_canvas.selected_items.clear()
            self.stitching_canvas.update()

            self.item_being_segmented = None
            self._editing_batch_card = None

            if not getattr(self, '_is_in_transition', False):
                self.switch_page(getattr(self, 'BATCH_MATTING_INDEX', 3))

            self.update_button_states()
            return

        self.item_being_segmented = None
        if getattr(self, 'is_in_quick_segment_flow', False):
            self.is_in_quick_segment_flow = False

        self.update_button_states()

    def _on_workshop_content_container_resize(self, event: QResizeEvent):
        container_rect = self.workshop_content_container.rect()
        if hasattr(self, 'stitching_canvas'):
            self.stitching_canvas.setGeometry(container_rect)
        if hasattr(self, 'segmentation_overlay_label'):
            self.segmentation_overlay_label.setGeometry(container_rect)
        QWidget.resizeEvent(self.workshop_content_container, event)

    def create_card_widget(self, title_text: str):
        """Constructs an aligned rounded container acting as a modern UI card component."""
        card_container = QWidget()
        card_container.setStyleSheet("""
            QWidget { 
                background-color: #262626; 
                border-radius: 10px; 
            }
        """)
        card_layout = QVBoxLayout(card_container)
        card_layout.setContentsMargins(12, 12, 12, 16)
        card_layout.setSpacing(12)

        title_label = QLabel(title_text)
        title_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF; 
                font-size: 13px; 
                font-weight: bold; 
                background: transparent;
            }
        """)

        content_area = QWidget()
        content_area.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        card_layout.addWidget(title_label)
        card_layout.addWidget(content_area)

        return card_container, content_layout

    @Slot(Qt.ApplicationState)
    def _handle_application_state_change(self, state: Qt.ApplicationState):
        if hasattr(self, '_ignore_focus_changes') and self._ignore_focus_changes:
            return

        right_panel = getattr(self, 'floating_panel_container', None)
        left_panel = getattr(self, 'asset_library_panel_floating', None)

        if not right_panel or not left_panel:
            return

        if state == Qt.ApplicationInactive:
            if right_panel.isVisible():
                self._was_right_panel_visible_on_deactivate = True
                right_panel.hide()
            if left_panel.isVisible():
                self._was_left_panel_visible_on_deactivate = True
                left_panel.hide()

        elif state == Qt.ApplicationActive:
            if self._was_right_panel_visible_on_deactivate:
                right_panel.show()
                right_panel.raise_()
            if self._was_left_panel_visible_on_deactivate:
                left_panel.show()
                left_panel.raise_()

            self._was_right_panel_visible_on_deactivate = False
            self._was_left_panel_visible_on_deactivate = False

    def eventFilter(self, watched_object: QObject, event: QEvent) -> bool:
        if event.type() not in [QEvent.Type.KeyPress, QEvent.Type.KeyRelease]:
            return super().eventFilter(watched_object, event)

        key_event = QKeyEvent(event)
        key = key_event.key()
        is_input_focused = isinstance(QApplication.focusWidget(), (QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit))
        current_page_idx = self.stacked_widget.currentIndex()

        if key == Qt.Key.Key_Alt:
            if current_page_idx == self.CREATIVE_WORKSHOP_INDEX and self.is_in_segmentation_overlay_mode:
                active_label = getattr(self, 'segmentation_overlay_label', None)
                if active_label and active_label.original_cv_image is not None:
                    if event.type() == QEvent.Type.KeyPress and not key_event.isAutoRepeat():
                        self.start_mask_preview()
                        return True
                    elif event.type() == QEvent.Type.KeyRelease and not key_event.isAutoRepeat():
                        self.stop_mask_preview()
                        return True

        if event.type() == QEvent.Type.KeyPress and key == Qt.Key.Key_H and not is_input_focused:
            if not self._is_in_transition and current_page_idx != self.WELCOME_PAGE_INDEX:
                self.switch_page_with_slide(self.WELCOME_PAGE_INDEX)
                return True

        if event.type() == QEvent.Type.KeyPress and key == Qt.Key.Key_Space and not is_input_focused:
            if current_page_idx == self.VIDEO_SEG_PAGE_INDEX:
                play_button = getattr(self, 'play_pause_button', None)
                if play_button and play_button.isEnabled():
                    play_button.click()
                    return True

        if not is_input_focused:
            if key_event.matches(QKeySequence.StandardKey.Undo) and self.undo_action.isEnabled():
                self.undo_action.trigger()
                return True
            if key_event.matches(QKeySequence.StandardKey.Redo) and self.redo_action.isEnabled():
                self.redo_action.trigger()
                return True
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.delete_action.isEnabled():
                self.delete_action.trigger()
                return True

        return super().eventFilter(watched_object, event)

    def moveEvent(self, event: QMoveEvent):
        super().moveEvent(event)
        current_main_pos = event.pos()

        if not hasattr(self, '_last_main_window_pos') or not self._last_main_window_pos:
            self._last_main_window_pos = current_main_pos
            return

        delta = current_main_pos - self._last_main_window_pos

        if not delta.isNull():
            if hasattr(self, 'floating_panel_container') and self.floating_panel_container and self.floating_panel_container.isVisible() and not self.floating_panel_container.isMinimized():
                try:
                    _ = self.floating_panel_container.metaObject()
                    self.floating_panel_container.move(self.floating_panel_container.pos() + delta)
                except RuntimeError:
                    pass

            if hasattr(self, 'asset_library_panel_floating') and self.asset_library_panel_floating and self.asset_library_panel_floating.isVisible() and not self.asset_library_panel_floating.isMinimized():
                try:
                    _ = self.asset_library_panel_floating.metaObject()
                    self.asset_library_panel_floating.move(self.asset_library_panel_floating.pos() + delta)
                except RuntimeError:
                    pass

        self._last_main_window_pos = current_main_pos

    def _create_welcome_card_final(self, title: str, asset_name: str, target_page_index: int):
        """Instantiates elevated dynamic cards on the Welcome view page."""
        outer_container = QWidget()
        outer_container.setCursor(Qt.CursorShape.PointingHandCursor)
        outer_container.setObjectName(f"welcome_card_{target_page_index}")
        outer_container.setProperty("is_pressed_down", False)
        outer_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        outer_layout = QVBoxLayout(outer_container)
        outer_layout.setContentsMargins(25, 50, 25, 25)

        floating_card = RoundedShadowCard(outer_container)
        shadow_effect = QGraphicsDropShadowEffect(floating_card)
        floating_card.setGraphicsEffect(shadow_effect)

        card_internal_layout = QVBoxLayout(floating_card)
        card_internal_layout.setContentsMargins(0, 0, 0, 0)

        asset_path = get_asset_path(os.path.join("welcome_assets", asset_name))
        media_player_widget = WelcomeImageCard(asset_path)
        media_player_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card_internal_layout.addWidget(media_player_widget)

        title_label = QLabel(title, outer_container)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignHCenter)
        title_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        title_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                color: #FFFFFF; 
                font-size: 16pt;
                font-weight: bold;
                border: none;
                qproperty-alignment: 'AlignCenter';
            }
        """)

        outer_layout.addWidget(floating_card)
        title_label.raise_()

        animation_group = QParallelAnimationGroup(outer_container)
        geom_animation_card = QPropertyAnimation(floating_card, b"geometry")
        shadow_color_animation = QPropertyAnimation(shadow_effect, b"color")
        shadow_blur_animation = QPropertyAnimation(shadow_effect, b"blurRadius")
        shadow_offset_animation = QPropertyAnimation(shadow_effect, b"offset")

        for anim in [geom_animation_card, shadow_color_animation, shadow_blur_animation, shadow_offset_animation]:
            anim.setDuration(250)
            anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
            animation_group.addAnimation(anim)

        def sync_title_position(card_geometry: QRect):
            title_height = 40
            title_y = (50 - title_height) / 2
            title_label.setGeometry(0, int(title_y), int(outer_container.width()), title_height)

        def get_target_hover_geom():
            orig_geom = floating_card.property("original_geometry") or floating_card.geometry()
            target_scale = 1.08
            w, h = orig_geom.width(), orig_geom.height()
            new_w, new_h = w * target_scale, h * target_scale
            center_x, center_y = orig_geom.x() + w / 2, orig_geom.y() + h / 2
            return QRect(int(center_x - new_w / 2), int(center_y - new_h / 2), int(new_w), int(new_h))

        def set_initial_state():
            if not outer_container.isVisible():
                return
            animation_group.stop()
            initial_geom = floating_card.geometry()
            floating_card.setProperty("original_geometry", initial_geom)
            sync_title_position(initial_geom)
            shadow_effect.setColor(QColor(0, 0, 0, 0))
            shadow_effect.setBlurRadius(0)
            shadow_effect.setOffset(0, 0)

        QTimer.singleShot(0, set_initial_state)

        def enterEvent(event):
            if outer_container.property("is_pressed_down"):
                return
            animation_group.stop()
            geom_animation_card.setStartValue(floating_card.geometry())
            geom_animation_card.setEndValue(get_target_hover_geom())
            shadow_color_animation.setStartValue(shadow_effect.color())
            shadow_color_animation.setEndValue(QColor(0, 0, 0, 75))
            shadow_blur_animation.setStartValue(shadow_effect.blurRadius())
            shadow_blur_animation.setEndValue(70)
            shadow_offset_animation.setStartValue(shadow_effect.offset())
            shadow_offset_animation.setEndValue(QPointF(0, 20))
            animation_group.start()
            media_player_widget.start_media()
            QWidget.enterEvent(outer_container, event)

        def leaveEvent(event):
            animation_group.stop()
            orig_geom = floating_card.property("original_geometry") or floating_card.geometry()
            geom_animation_card.setStartValue(floating_card.geometry())
            geom_animation_card.setEndValue(orig_geom)
            shadow_color_animation.setStartValue(shadow_effect.color())
            shadow_color_animation.setEndValue(QColor(0, 0, 0, 0))
            shadow_blur_animation.setStartValue(shadow_effect.blurRadius())
            shadow_blur_animation.setEndValue(0)
            shadow_offset_animation.setStartValue(shadow_effect.offset())
            shadow_offset_animation.setEndValue(QPointF(0, 0))
            animation_group.start()
            media_player_widget.stop_media()
            QWidget.leaveEvent(outer_container, event)

        def mouseMoveEvent(event):
            QWidget.mouseMoveEvent(outer_container, event)

        def card_mouse_press_event(event):
            if event.button() == Qt.MouseButton.LeftButton:
                animation_group.stop()
                outer_container.setProperty("is_pressed_down", True)
                press_animation = QPropertyAnimation(floating_card, b"geometry", outer_container)
                press_animation.setDuration(120)
                press_animation.setEasingCurve(QEasingCurve.Type.OutQuad)
                current_geom = floating_card.geometry()
                target_w = current_geom.width() * 0.95
                target_h = current_geom.height() * 0.95
                center = current_geom.center()
                press_animation.setStartValue(current_geom)
                press_animation.setEndValue(
                    QRect(int(center.x() - target_w / 2), int(center.y() - target_h / 2), int(target_w), int(target_h))
                )
                press_animation.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
            QWidget.mousePressEvent(outer_container, event)

        def card_mouse_release_event(event):
            was_pressed = outer_container.property("is_pressed_down")
            outer_container.setProperty("is_pressed_down", False)
            if event.button() == Qt.MouseButton.LeftButton and was_pressed:
                if outer_container.rect().contains(event.position().toPoint()):
                    animation_group.stop()
                    media_player_widget.stop_media()
                    self.switch_page_with_slide(target_page_index, clicked_widget=outer_container,
                                                core_card_widget=floating_card)
            QWidget.mouseReleaseEvent(outer_container, event)

        def container_resize_event(event):
            QTimer.singleShot(0, set_initial_state)
            QWidget.resizeEvent(outer_container, event)

        outer_container.enterEvent = enterEvent
        outer_container.leaveEvent = leaveEvent
        outer_container.mouseMoveEvent = mouseMoveEvent
        outer_container.mousePressEvent = card_mouse_press_event
        outer_container.mouseReleaseEvent = card_mouse_release_event
        outer_container.resizeEvent = container_resize_event

        outer_container.setProperty("core_content_widget", floating_card)
        return outer_container, floating_card

    def _create_inspector_button(self, svg_name, text, tooltip="", icon_color=None):
        """Constructs an action item for the right-hand Inspector tab panel."""
        action = QAction(text, self)
        default_color = icon_color if icon_color else globals().get('C_ICON_COLOR_TOOL', "#606060")
        action.setIcon(self._create_svg_icon(svg_name, size=24, color=default_color))
        action.setToolTip(tooltip)
        action.setCheckable(True)
        return action

    def toggle_asset_library(self, checked: bool):
        button_to_sync = getattr(self, 'workshop_asset_library_button', None)
        try:
            if not hasattr(self, 'asset_panel_animation_floating') or not hasattr(self, 'asset_library_panel_floating'):
                if button_to_sync:
                    button_to_sync.setChecked(not checked)
                return

            if not hasattr(self, 'workshop_main_top_bar'):
                if button_to_sync:
                    button_to_sync.setChecked(not checked)
                return

            self.asset_panel_animation_floating.stop()
            panel = self.asset_library_panel_floating
            panel_ideal_width = 220

            button_bar_global_pos = self.workshop_main_top_bar.mapToGlobal(QPoint(0, 0))
            button_bar_global_bottom_y = button_bar_global_pos.y() + self.workshop_main_top_bar.height()
            main_window_global_pos = self.mapToGlobal(QPoint(0, 0))
            main_window_height = self.height()

            panel_target_y_global = button_bar_global_bottom_y + 8
            available_height_for_panel = (main_window_global_pos.y() + main_window_height) - panel_target_y_global - 10
            panel_final_height = max(250, available_height_for_panel)

            target_rect_global_visible = QRect(int(main_window_global_pos.x() + 10), int(panel_target_y_global),
                                               int(panel_ideal_width), int(panel_final_height))
            target_rect_global_hidden = QRect(int(main_window_global_pos.x() - panel_ideal_width),
                                              int(panel_target_y_global), int(panel_ideal_width),
                                              int(panel_final_height))

            current_geom = panel.geometry()
            if checked:
                if current_geom == target_rect_global_visible and panel.isVisible():
                    return
                panel.setGeometry(current_geom if panel.isVisible() else target_rect_global_hidden)
                panel.show()
                panel.raise_()
                self.asset_panel_animation_floating.setStartValue(panel.geometry())
                self.asset_panel_animation_floating.setEndValue(target_rect_global_visible)
            else:
                if current_geom == target_rect_global_hidden and not panel.isVisible():
                    return
                self.asset_panel_animation_floating.setStartValue(current_geom)
                self.asset_panel_animation_floating.setEndValue(target_rect_global_hidden)
                try:
                    self.asset_panel_animation_floating.finished.disconnect(panel.hide)
                except (TypeError, RuntimeError):
                    pass
                self.asset_panel_animation_floating.finished.connect(panel.hide, Qt.ConnectionType.SingleShotConnection)

            self.asset_panel_animation_floating.start()
        finally:
            if button_to_sync and button_to_sync.isChecked() != checked:
                button_to_sync.blockSignals(True)
                button_to_sync.setChecked(checked)
                button_to_sync.blockSignals(False)

    @Slot()
    def combine_selected_items_action(self):
        """Combines multiple selected canvas items into a single, flattened layout layer."""
        current_time_ms = time.time() * 1000
        if hasattr(self, '_last_combine_timestamp') and (current_time_ms - self._last_combine_timestamp < 500):
            return

        if not self.stitching_canvas or len(self.stitching_canvas.selected_items) < 2:
            QMessageBox.information(self, "操作无效", "请至少选择两个素材进行合并。")
            return

        self._last_combine_timestamp = current_time_ms

        try:
            self.show_status_message("正在合并素材...", 0)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            selected_items_to_combine = list(self.stitching_canvas.selected_items)

            total_bounds = QRectF()
            for item in selected_items_to_combine:
                item_bounds = item.get_transformed_bounding_rect()
                if total_bounds.isNull():
                    total_bounds = item_bounds
                else:
                    total_bounds = total_bounds.united(item_bounds)

            if total_bounds.isEmpty() or not total_bounds.isValid():
                raise ValueError("无法计算选中素材的有效边界。")

            target_image = QImage(total_bounds.size().toSize(), QImage.Format.Format_ARGB32_Premultiplied)
            target_image.fill(Qt.transparent)

            painter = QPainter(target_image)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            sorted_items_to_draw = [item for item in self.stitching_canvas.items if item in selected_items_to_combine]

            for item in sorted_items_to_draw:
                painter.save()
                relative_pos = item.pos - total_bounds.topLeft()
                transform = QTransform()
                transform.translate(relative_pos.x() + item.size.width() / 2, relative_pos.y() + item.size.height() / 2)
                transform.rotate(item.rotation)
                transform.translate(-item.size.width() / 2, -item.size.height() / 2)
                painter.setTransform(transform)
                painter.drawPixmap(QRectF(QPointF(0, 0), item.size), item.pixmap, item.pixmap.rect())
                painter.restore()
            painter.end()

            new_pixmap = QPixmap.fromImage(target_image)
            temp_dir = os.path.join(TEMP_BASE_DIR, "combined")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"combined_{uuid.uuid4().hex[:8]}.png")
            if not new_pixmap.save(temp_path, "PNG"):
                raise IOError("无法将合并后的图像保存到临时文件。")

            new_item_name = "合并素材_" + str(uuid.uuid4())[:4]
            new_item = StitchedImageItem(new_pixmap, new_item_name, original_path=temp_path)
            new_item.pos = total_bounds.topLeft()
            new_item.size = QSizeF(new_pixmap.size())
            new_item.source_items = selected_items_to_combine

            self.stitching_canvas.items = [item for item in self.stitching_canvas.items if
                                           item not in selected_items_to_combine]
            self.stitching_canvas.items.append(new_item)
            self.stitching_canvas.selected_items = [new_item]
            self.stitching_canvas.layers_changed.emit()
            self.stitching_canvas.selection_changed.emit()
            self.stitching_canvas.update()

            self.show_status_message("素材合并成功！", 3000)

        except Exception as e:
            self.log_message.emit(f"合并素材时出错: {e}")
            QMessageBox.critical(self, "合并失败", f"合并素材时发生错误：\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
            self.update_button_states()

    @Slot()
    def uncombine_selected_item_action(self):
        """Splits a merged canvas item back into its original separate components."""
        if not self.stitching_canvas:
            return

        if len(self.stitching_canvas.selected_items) != 1:
            QMessageBox.information(self, "操作无效", "请只选择一个已合并的素材项来取消合并。")
            return

        item_to_uncombine = self.stitching_canvas.get_primary_selected_item()

        if not item_to_uncombine or not item_to_uncombine.source_items:
            QMessageBox.information(self, "操作无效", "当前选中的素材不是通过合并创建的，无法取消合并。")
            return

        self.show_status_message("正在取消合并...", 0)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            original_items = item_to_uncombine.source_items
            self.stitching_canvas.items.remove(item_to_uncombine)
            self.stitching_canvas.items.extend(original_items)
            self.stitching_canvas.selected_items = original_items

            self.stitching_canvas.layers_changed.emit()
            self.stitching_canvas.selection_changed.emit()
            self.stitching_canvas.update()

            self.show_status_message("取消合并成功！", 3000)

        except Exception as e:
            self.log_message.emit(f"取消合并时出错: {e}")
            QMessageBox.critical(self, "取消合并失败", f"取消合并时发生错误：\n{e}")
            if item_to_uncombine not in self.stitching_canvas.items:
                self.stitching_canvas.items.append(item_to_uncombine)
        finally:
            QApplication.restoreOverrideCursor()

    def _animate_trigger_button(self, button: QToolButton, expand: bool):
        """Applies dynamic micro-interactions to top bar icon buttons."""
        if not button:
            return

        original_icon_size = button.iconSize()
        if not hasattr(button, '_button_animation'):
            button._button_animation = QPropertyAnimation(button, b"iconSize")
            button._button_animation.setDuration(200)
            button._button_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        animation = button._button_animation
        animation.stop()

        scale_factor = 1.2 if expand else 1.0
        intermediate_size = QSize(int(original_icon_size.width() * scale_factor),
                                  int(original_icon_size.height() * scale_factor))

        seq_animation = QSequentialAnimationGroup(button)
        anim_part1 = QPropertyAnimation(button, b"iconSize")
        anim_part1.setDuration(100)
        anim_part1.setStartValue(original_icon_size)
        anim_part1.setEndValue(intermediate_size)
        anim_part1.setEasingCurve(QEasingCurve.Type.OutQuad)

        anim_part2 = QPropertyAnimation(button, b"iconSize")
        anim_part2.setDuration(100)
        anim_part2.setStartValue(intermediate_size)
        anim_part2.setEndValue(original_icon_size)
        anim_part2.setEasingCurve(QEasingCurve.Type.InQuad)

        seq_animation.addAnimation(anim_part1)
        seq_animation.addAnimation(anim_part2)

        seq_animation.finished.connect(seq_animation.deleteLater)
        seq_animation.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _create_internal_actions(self):
        self.nav_action_group = QActionGroup(self)
        self.nav_action_group.setExclusive(True)

        actions_data = [
            ("视频抠图", self.VIDEO_SEG_PAGE_INDEX),
            ("创意工坊", self.CREATIVE_WORKSHOP_INDEX)
        ]

        for text, page_index in actions_data:
            action = QAction(text, self)
            action.setData(page_index)
            self.nav_action_group.addAction(action)

            action.triggered.connect(
                lambda checked, index=page_index: self.switch_page_with_slide(index) if checked else None
            )

            if page_index == self.VIDEO_SEG_PAGE_INDEX:
                self.nav_video_seg_action = action
            elif page_index == self.CREATIVE_WORKSHOP_INDEX:
                self.nav_creative_workshop_action = action

    def _animate_right_sidebar(self, show: bool, on_finished_callback=None):
        settings_panel = self.right_settings_container
        animation = self.right_settings_animation

        target_width = 0
        is_currently_visible = settings_panel.width() > 0
        current_actual_width = settings_panel.width()

        if show:
            current_page_index = self.stacked_widget.currentIndex()
            active_inspector_action = None
            nav_action_for_current_page = self.nav_action_group.checkedAction()
            inspector_button_group_to_check = None

            if nav_action_for_current_page:
                if nav_action_for_current_page == getattr(self, 'nav_enhance_action', None) and hasattr(self, 'enhance_button_group'):
                    inspector_button_group_to_check = self.enhance_button_group
                elif nav_action_for_current_page == getattr(self, 'nav_segment_action', None) and hasattr(self, 'seg_button_group'):
                    inspector_button_group_to_check = self.seg_button_group
                elif nav_action_for_current_page == self.nav_video_seg_action and hasattr(self, 'vid_button_group'):
                    inspector_button_group_to_check = self.vid_button_group
                elif nav_action_for_current_page == getattr(self, 'nav_stitching_action', None) and hasattr(self, 'stitch_button_group'):
                    inspector_button_group_to_check = self.stitch_button_group

                if not inspector_button_group_to_check and nav_action_for_current_page.property("indices"):
                    inspector_idx = nav_action_for_current_page.property("indices")['inspector']
                    current_inspector_bar = self.all_pages_inspector_stack.widget(inspector_idx)
                    if current_inspector_bar and isinstance(current_inspector_bar, QToolBar):
                        inspector_button_group_to_check = current_inspector_bar.property("button_group")

            if inspector_button_group_to_check:
                active_inspector_action = inspector_button_group_to_check.checkedAction()

            if active_inspector_action:
                if current_page_index == self.VIDEO_SEG_PAGE_INDEX:
                    target_width = 400
                else:
                    target_width = 380

                if nav_action_for_current_page and nav_action_for_current_page.property("indices"):
                    indices = nav_action_for_current_page.property("indices")
                    settings_stack_for_page = self.all_pages_settings_stack.widget(indices['settings'])
                    if settings_stack_for_page:
                        settings_stack_for_page.setCurrentIndex(active_inspector_action.data())
            else:
                target_width = 0
        else:
            target_width = 0

        if (show and is_currently_visible and current_actual_width == target_width) or \
                (not show and not is_currently_visible and target_width == 0):
            if on_finished_callback:
                QTimer.singleShot(0, on_finished_callback)
            return

        if animation.state() == QPropertyAnimation.State.Running:
            animation.stop()

        try:
            animation.finished.disconnect()
        except (TypeError, RuntimeError):
            pass

        animation.setStartValue(current_actual_width)
        animation.setEndValue(target_width)

        if on_finished_callback:
            animation.finished.connect(on_finished_callback)
            if target_width == 0:
                animation.finished.connect(
                    lambda: animation.finished.disconnect(on_finished_callback) if animation.signalsBlocked() else None,
                    Qt.ConnectionType.SingleShotConnection
                )

        animation.start()

    def _get_nav_button_for_page_index(self, page_index: int) -> Optional[QToolButton]:
        if not hasattr(self, 'nav_action_group'):
            return None
        for action in self.nav_action_group.actions():
            if action.data() == page_index:
                widget = self.nav_toolbar.widgetForAction(action)
                if isinstance(widget, QToolButton):
                    return widget
        return None

    def _prepare_sidebar_for_state(self, page_index: int, should_be_active: bool):
        target_width = 0
        inspector_bar = None
        settings_stack = None
        nav_action = self._get_nav_action_for_page(page_index)

        if nav_action and nav_action.property("indices"):
            indices = nav_action.property("indices")
            inspector_bar = self.all_pages_inspector_stack.widget(indices['inspector'])
            settings_stack = self.all_pages_settings_stack.widget(indices['settings'])

        if should_be_active and inspector_bar and settings_stack:
            if page_index == self.VIDEO_SEG_PAGE_INDEX:
                target_width = 400
            else:
                target_width = 380

            self.all_pages_inspector_stack.setCurrentWidget(inspector_bar)
            self.all_pages_settings_stack.setCurrentWidget(settings_stack)

            button_group = inspector_bar.property("button_group")
            if button_group:
                button_group.blockSignals(True)
                first_action = next((a for a in button_group.actions() if a.isCheckable()), None)
                if first_action:
                    first_action.setChecked(True)
                    settings_stack.setCurrentIndex(first_action.data())
                button_group.blockSignals(False)

        if hasattr(self, 'right_settings_container'):
            self.right_settings_container.setFixedWidth(target_width)
            self.right_settings_container.setVisible(should_be_active)

    def _get_nav_action_for_page(self, page_index: int) -> Optional[QAction]:
        for action in self.nav_action_group.actions():
            if action.data() == page_index:
                return action
        return None

    def _execute_actual_page_switch_animation(self, target_page_index: int, clicked_widget: QWidget = None,
                                              core_card_widget: QWidget = None, on_finished_callback: callable = None):
        from_index = self.stacked_widget.currentIndex()
        to_widget = self.stacked_widget.widget(target_page_index)

        is_expanding = (from_index == self.WELCOME_PAGE_INDEX and target_page_index != self.WELCOME_PAGE_INDEX)
        is_collapsing = (from_index != self.WELCOME_PAGE_INDEX and target_page_index == self.WELCOME_PAGE_INDEX)

        stage_rect = self.stacked_widget.geometry()
        animation_overlay = QWidget(self)
        animation_overlay.setGeometry(stage_rect)
        animation_overlay.show()
        animation_overlay.raise_()

        animation_group = QParallelAnimationGroup(self)
        _local_target_core_card_for_cleanup = None

        TOTAL_DURATION = 650
        EASING_CURVE = QEasingCurve.Type.InOutCubic
        should_use_blur_effect = self.use_blur_animation

        if is_expanding and clicked_widget and core_card_widget:
            initial_card_pixmap = core_card_widget.grab()
            core_card_widget.hide()
            QApplication.processEvents()

            welcome_page_pixmap = self.welcome_page.grab(self.welcome_page.rect())

            card_start_pos_global = clicked_widget.mapToGlobal(core_card_widget.geometry().topLeft())
            card_start_rect = QRect(animation_overlay.mapFromGlobal(card_start_pos_global),
                                    core_card_widget.geometry().size())
            page_final_rect_in_overlay = animation_overlay.rect()

            to_widget.setGeometry(page_final_rect_in_overlay)
            target_page_pixmap = QPixmap(page_final_rect_in_overlay.size())
            target_page_pixmap.fill(Qt.transparent)
            to_widget.render(target_page_pixmap)
            self.stacked_widget.hide()

            background_actor = AspectRatioPixmapWidget(welcome_page_pixmap, animation_overlay)
            background_actor.setGeometry(animation_overlay.rect())
            background_actor.show()

            foreground_actor = TransitionCard(initial_card_pixmap, animation_overlay)
            foreground_actor.set_target_pixmap(target_page_pixmap)
            foreground_actor.setGeometry(card_start_rect)
            card_radius = getattr(core_card_widget, 'radius', 20.0)
            foreground_actor.radius = card_radius
            foreground_actor.initialOpacity = 1.0
            foreground_actor.targetOpacity = 0.0
            foreground_actor.show()

            geom_anim = QPropertyAnimation(foreground_actor, b"geometry", animation_group)
            geom_anim.setDuration(TOTAL_DURATION)
            geom_anim.setStartValue(card_start_rect)
            geom_anim.setEndValue(page_final_rect_in_overlay)
            geom_anim.setEasingCurve(EASING_CURVE)

            radius_anim = QPropertyAnimation(foreground_actor, b"radius", animation_group)
            radius_anim.setDuration(TOTAL_DURATION)
            radius_anim.setStartValue(card_radius)
            radius_anim.setEndValue(0.0)
            radius_anim.setEasingCurve(EASING_CURVE)

            mid_point_swap_sequence = QSequentialAnimationGroup(animation_group)
            HALF_DURATION = TOTAL_DURATION // 2

            mid_point_swap_sequence.addPause(HALF_DURATION)

            swap_animations = QParallelAnimationGroup()
            instant_fade_out_card = QPropertyAnimation(foreground_actor, b"initialOpacity")
            instant_fade_out_card.setDuration(1)
            instant_fade_out_card.setStartValue(1.0)
            instant_fade_out_card.setEndValue(0.0)
            swap_animations.addAnimation(instant_fade_out_card)

            instant_fade_in_page = QPropertyAnimation(foreground_actor, b"targetOpacity")
            instant_fade_in_page.setDuration(1)
            instant_fade_in_page.setStartValue(0.0)
            instant_fade_in_page.setEndValue(1.0)
            swap_animations.addAnimation(instant_fade_in_page)

            mid_point_swap_sequence.addAnimation(swap_animations)

            bg_geom_anim = QPropertyAnimation(background_actor, b"geometry", animation_group)
            bg_geom_anim.setDuration(TOTAL_DURATION)
            start_bg_rect = animation_overlay.rect()
            scale_factor = 1.05
            end_bg_w = start_bg_rect.width() * scale_factor
            end_bg_h = start_bg_rect.height() * scale_factor
            end_bg_x = (start_bg_rect.width() - end_bg_w) / 2
            end_bg_y = (start_bg_rect.height() - end_bg_h) / 2
            end_bg_rect_zoomed = QRect(int(end_bg_x), int(end_bg_y), int(end_bg_w), int(end_bg_h))
            bg_geom_anim.setStartValue(start_bg_rect)
            bg_geom_anim.setEndValue(end_bg_rect_zoomed)
            bg_geom_anim.setEasingCurve(EASING_CURVE)

            if should_use_blur_effect:
                blur_effect = QGraphicsBlurEffect(background_actor)
                blur_effect.setBlurRadius(0)
                background_actor.setGraphicsEffect(blur_effect)
                blur_anim = QPropertyAnimation(blur_effect, b"blurRadius", animation_group)
                blur_anim.setDuration(int(TOTAL_DURATION * 0.8))
                blur_anim.setStartValue(0.0)
                blur_anim.setEndValue(25.0)
                blur_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        elif is_collapsing and clicked_widget and core_card_widget:
            _local_target_core_card_for_cleanup = core_card_widget
            _local_target_core_card_for_cleanup.hide()

            source_content_widget = None
            if from_index == self.VIDEO_SEG_PAGE_INDEX:
                source_content_widget = getattr(self, 'video_display_label', None)
            elif from_index == self.CREATIVE_WORKSHOP_INDEX:
                source_content_widget = getattr(self, 'stitching_canvas', None)

            if not source_content_widget:
                animation_overlay.deleteLater()
                self.switch_page(target_page_index)
                self._is_in_transition = False
                self.unsetCursor()
                if on_finished_callback:
                    on_finished_callback()
                return

            initial_page_pixmap = source_content_widget.grab()
            source_content_global_pos = source_content_widget.mapToGlobal(QPoint(0, 0))
            page_start_rect_in_overlay = QRect(animation_overlay.mapFromGlobal(source_content_global_pos),
                                               source_content_widget.size())

            target_card_pixmap = _local_target_core_card_for_cleanup.grab()
            welcome_page_pixmap = self.welcome_page.grab()
            card_target_pos_global = clicked_widget.mapToGlobal(core_card_widget.geometry().topLeft())
            card_target_rect_in_overlay = QRect(animation_overlay.mapFromGlobal(card_target_pos_global),
                                                core_card_widget.geometry().size())
            self.stacked_widget.hide()

            background_actor = AspectRatioPixmapWidget(welcome_page_pixmap, animation_overlay)
            background_actor.show()

            foreground_actor = TransitionCard(initial_page_pixmap, animation_overlay)
            foreground_actor.set_target_pixmap(target_card_pixmap)
            foreground_actor.setGeometry(page_start_rect_in_overlay)
            foreground_actor.radius = 0.0
            foreground_actor.initialOpacity = 1.0
            foreground_actor.targetOpacity = 0.0
            foreground_actor.show()

            card_radius = getattr(_local_target_core_card_for_cleanup, 'radius', 20.0)

            geom_anim = QPropertyAnimation(foreground_actor, b"geometry", animation_group)
            geom_anim.setDuration(TOTAL_DURATION)
            geom_anim.setStartValue(page_start_rect_in_overlay)
            geom_anim.setEndValue(card_target_rect_in_overlay)
            geom_anim.setEasingCurve(EASING_CURVE)

            radius_anim = QPropertyAnimation(foreground_actor, b"radius", animation_group)
            radius_anim.setDuration(TOTAL_DURATION)
            radius_anim.setStartValue(0.0)
            radius_anim.setEndValue(card_radius)
            radius_anim.setEasingCurve(EASING_CURVE)

            mid_point_swap_sequence_collapse = QSequentialAnimationGroup(animation_group)
            HALF_DURATION = TOTAL_DURATION // 2

            mid_point_swap_sequence_collapse.addPause(HALF_DURATION)

            swap_animations_collapse = QParallelAnimationGroup()
            instant_fade_out_page = QPropertyAnimation(foreground_actor, b"initialOpacity")
            instant_fade_out_page.setDuration(1)
            instant_fade_out_page.setStartValue(1.0)
            instant_fade_out_page.setEndValue(0.0)
            swap_animations_collapse.addAnimation(instant_fade_out_page)

            instant_fade_in_card = QPropertyAnimation(foreground_actor, b"targetOpacity")
            instant_fade_in_card.setDuration(1)
            instant_fade_in_card.setStartValue(0.0)
            instant_fade_in_card.setEndValue(1.0)
            swap_animations_collapse.addAnimation(instant_fade_in_card)

            mid_point_swap_sequence_collapse.addAnimation(swap_animations_collapse)

            bg_geom_anim = QPropertyAnimation(background_actor, b"geometry", animation_group)
            bg_geom_anim.setDuration(TOTAL_DURATION)
            end_bg_rect = animation_overlay.rect()
            scale_factor = 1.05
            start_bg_w = end_bg_rect.width() * scale_factor
            start_bg_h = end_bg_rect.height() * scale_factor
            start_bg_x = (end_bg_rect.width() - start_bg_w) / 2
            start_bg_y = (end_bg_rect.height() - start_bg_h) / 2
            start_bg_rect_zoomed = QRect(int(start_bg_x), int(start_bg_y), int(start_bg_w), int(start_bg_h))
            bg_geom_anim.setStartValue(start_bg_rect_zoomed)
            bg_geom_anim.setEndValue(end_bg_rect)
            bg_geom_anim.setEasingCurve(EASING_CURVE)

            if should_use_blur_effect:
                blur_effect = QGraphicsBlurEffect(background_actor)
                blur_effect.setBlurRadius(25.0)
                background_actor.setGraphicsEffect(blur_effect)
                blur_radius_anim = QPropertyAnimation(blur_effect, b"blurRadius", animation_group)
                blur_radius_anim.setDuration(int(TOTAL_DURATION * 0.8))
                blur_radius_anim.setStartValue(25.0)
                blur_radius_anim.setEndValue(0.0)
                blur_radius_anim.setEasingCurve(QEasingCurve.Type.InCubic)

        else:
            animation_overlay.deleteLater()
            self.switch_page(target_page_index)
            self._is_in_transition = False
            self.unsetCursor()
            if on_finished_callback:
                on_finished_callback()
            return

        def on_page_switch_animation_finished(overlay_to_delete: QWidget):
            nonlocal core_card_widget, _local_target_core_card_for_cleanup
            target_card_to_show = core_card_widget if is_expanding else _local_target_core_card_for_cleanup

            self.stacked_widget.setCurrentIndex(target_page_index)
            self.update_ui_for_page_change(target_page_index)
            self.stacked_widget.show()

            if overlay_to_delete and isinstance(overlay_to_delete, QWidget):
                overlay_to_delete.deleteLater()

            if target_card_to_show:
                target_card_to_show.show()

            self._is_in_transition = False
            self.unsetCursor()
            if on_finished_callback:
                on_finished_callback()

        animation_group.finished.connect(lambda: on_page_switch_animation_finished(animation_overlay))
        animation_group.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def switch_page_with_slide(self, target_page_index: int, clicked_widget: QWidget = None,
                               core_card_widget: QWidget = None, on_finished_callback: callable = None):
        if self._is_in_transition or self.stacked_widget.currentIndex() == target_page_index:
            return

        from_index = self.stacked_widget.currentIndex()

        if target_page_index == self.WELCOME_PAGE_INDEX and clicked_widget is None:
            if from_index == self.VIDEO_SEG_PAGE_INDEX:
                clicked_widget = self.welcome_page.card1_container
            elif from_index == self.CREATIVE_WORKSHOP_INDEX:
                clicked_widget = self.welcome_page.card2_container

            if clicked_widget:
                core_card_widget = clicked_widget.property("core_content_widget")

        self._is_in_transition = True
        self.setCursor(Qt.CursorShape.WaitCursor)

        @Slot()
        def _start_final_animation():
            self._execute_actual_page_switch_animation(
                target_page_index, clicked_widget, core_card_widget, on_finished_callback
            )

        @Slot()
        def _check_and_close_right_panel():
            right_panel_is_open = self.floating_panel_container and not self.floating_panel_container.isHidden()
            if right_panel_is_open:
                self.floating_panel_container.closed.connect(_check_and_close_left_panel,
                                                             Qt.ConnectionType.SingleShotConnection)
                self.hide_floating_panel()
            else:
                _check_and_close_left_panel()

        @Slot()
        def _check_and_close_left_panel():
            try:
                if self.floating_panel_container:
                    self.floating_panel_container.closed.disconnect(_check_and_close_left_panel)
            except (TypeError, RuntimeError):
                pass

            left_panel_is_open = hasattr(self, 'asset_library_panel_floating') and \
                                 self.asset_library_panel_floating and \
                                 not self.asset_library_panel_floating.isHidden()

            if left_panel_is_open:
                if hasattr(self, 'asset_panel_animation_floating') and self.asset_panel_animation_floating:
                    self.asset_panel_animation_floating.finished.connect(_start_final_animation,
                                                                         Qt.ConnectionType.SingleShotConnection)
                    self.toggle_asset_library(False)
                else:
                    if hasattr(self, 'asset_library_panel_floating'):
                        self.asset_library_panel_floating.hide()
                    QTimer.singleShot(50, _start_final_animation)
            else:
                _start_final_animation()

        _check_and_close_right_panel()

    def switch_page(self, index):
        if self._is_in_transition or self.stacked_widget.currentIndex() == index:
            return

        self.hide_floating_panel()
        if hasattr(self, 'asset_library_panel_floating') and self.asset_library_panel_floating.isVisible():
            self.toggle_asset_library(False)

        self.stacked_widget.setCurrentIndex(index)
        self.update_ui_for_page_change(index)

    def update_ui_for_page_change(self, index: int):
        if self.floating_panel_container and self.floating_panel_container.isVisible():
            self.hide_floating_panel()

        if hasattr(self, 'asset_library_panel_floating') and self.asset_library_panel_floating and self.asset_library_panel_floating.isVisible():
            self.toggle_asset_library(False)

        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)
        self.delete_action.setEnabled(False)

        if self.is_playing:
            self.pause_video()

        if index == self.CREATIVE_WORKSHOP_INDEX:
            asset_panel_should_be_visible = self.settings.value("stitching/asset_panel_visible", False, type=bool)
            if asset_panel_should_be_visible and hasattr(self, 'workshop_asset_library_button'):
                self.workshop_asset_library_button.setChecked(True)
                QTimer.singleShot(50, lambda: self.toggle_asset_library(True))

            if self.is_in_segmentation_overlay_mode:
                self._exit_segmentation_mode()

            self.delete_action.setEnabled(True)

            if hasattr(self, 'stitching_canvas') and self.stitching_canvas and not self.stitching_canvas._initial_fit_done:
                QTimer.singleShot(0, lambda: self.stitching_canvas.fit_canvas_to_view(zoom_to_fit=True))

        elif index == self.VIDEO_SEG_PAGE_INDEX:
            self.update_video_preview_all_targets(self.current_frame_index)

        self.update_button_states()

    def _setup_global_actions(self):
        self.undo_action = QAction("撤销", self)
        self.undo_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.undo_action.triggered.connect(self.trigger_undo)

        self.redo_action = QAction("重做", self)
        self.redo_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.redo_action.triggered.connect(self.trigger_redo)

        self.delete_action = QAction("删除", self)
        self.delete_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.delete_action.triggered.connect(self.trigger_delete)

        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)
        self.delete_action.setEnabled(False)

    def trigger_undo(self):
        current_page_idx = self.stacked_widget.currentIndex()
        if current_page_idx == self.CREATIVE_WORKSHOP_INDEX and getattr(self, 'is_in_segmentation_overlay_mode', False):
            if hasattr(self, 'segmentation_overlay_label') and self.segmentation_overlay_label and self.segmentation_overlay_label.isEnabled():
                self.segmentation_overlay_label.undo_last_action()
        elif current_page_idx == self.VIDEO_SEG_PAGE_INDEX:
            self.undo_video_action()

    def trigger_redo(self):
        current_page_idx = self.stacked_widget.currentIndex()
        if current_page_idx == self.CREATIVE_WORKSHOP_INDEX and getattr(self, 'is_in_segmentation_overlay_mode', False):
            if hasattr(self, 'segmentation_overlay_label') and self.segmentation_overlay_label and self.segmentation_overlay_label.isEnabled():
                self.segmentation_overlay_label.redo_last_action()
        elif current_page_idx == self.VIDEO_SEG_PAGE_INDEX:
            self.redo_video_action()

    def trigger_delete(self):
        current_page_idx = self.stacked_widget.currentIndex()
        if current_page_idx == self.CREATIVE_WORKSHOP_INDEX and not getattr(self, 'is_in_segmentation_overlay_mode', False):
            if hasattr(self, 'stitching_canvas') and self.stitching_canvas:
                self.stitching_canvas.delete_selected_item()
        elif current_page_idx == self.VIDEO_SEG_PAGE_INDEX:
            self.clear_current_video_target()

    def _set_window_icon(self):
        icon_path_resolved = get_asset_path(APP_ICON_FILENAME)
        if os.path.exists(icon_path_resolved):
            self.setWindowIcon(QIcon(icon_path_resolved))
        else:
            self.setWindowIcon(self._create_svg_icon("assets/icons/toolbox.svg", size=32))

    def _create_svg_icon(self, svg_filename, size=24, color=None, for_home_button=False):
        """Loads and processes SVG icon files, rendering them dynamically based on active high-DPI scaling targets."""
        icon_path = get_asset_path(os.path.join("icons", svg_filename))

        if not os.path.exists(icon_path):
            print(f"Warning: SVG icon target not found -> {icon_path}")
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            return QIcon(pixmap)

        try:
            device_pixel_ratio = QApplication.primaryScreen().devicePixelRatio()
        except AttributeError:
            device_pixel_ratio = 1.0

        physical_size = int(size * device_pixel_ratio)
        actual_size_logical = int(size * 0.6) if for_home_button else int(size * 0.75)
        actual_size_physical = int(actual_size_logical * device_pixel_ratio)

        raw_icon = QIcon(icon_path)
        raw_pixmap = raw_icon.pixmap(QSize(actual_size_physical, actual_size_physical))

        if not color:
            raw_pixmap.setDevicePixelRatio(device_pixel_ratio)
            return QIcon(raw_pixmap)

        colored_pixmap = QPixmap(physical_size, physical_size)
        colored_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(colored_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        x = (physical_size - actual_size_physical) // 2
        y = (physical_size - actual_size_physical) // 2

        if for_home_button:
            y -= int(1 * device_pixel_ratio)

        painter.drawPixmap(x, y, raw_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)

        target_color = QColor(color) if isinstance(color, (str, QColor)) else QColor("#191919")
        painter.fillRect(colored_pixmap.rect(), target_color)
        painter.end()

        colored_pixmap.setDevicePixelRatio(device_pixel_ratio)
        return QIcon(colored_pixmap)

    def _create_tool_button(self, icon_source=None, text="", tooltip="", checkable=False, object_name=None,
                            svg_name=None, icon_color=None):
        """Constructs and returns flat-styled tool buttons with hover highlights."""
        button = QToolButton()

        if svg_name:
            default_color = icon_color if icon_color else "#606060"
            button.setIcon(self._create_svg_icon(svg_name, size=24, color=default_color))
        elif isinstance(icon_source, QStyle.StandardPixmap):
            button.setIcon(self.style().standardIcon(icon_source))
        elif isinstance(icon_source, str) and os.path.exists(icon_source):
            button.setIcon(QIcon(icon_source))

        if text:
            button.setText(text)
            button.setToolTip(f"{text} ({tooltip})" if tooltip else text)
        else:
            button.setToolTip(tooltip)

        button.setCheckable(checkable)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setIconSize(QSize(24, 24))

        if object_name:
            button.setObjectName(object_name)
        else:
            button.setObjectName("IconOnlyToolButton")

        return button

    @Slot()
    def _on_stitch_bg_mode_changed(self):
        if not all(hasattr(self, name) for name in ['stitch_bg_transparent_radio', 'stitching_canvas', 'stitch_bg_color_button']):
            return

        is_transparent = self.stitch_bg_transparent_radio.isChecked()
        self.stitch_bg_color_button.setEnabled(not is_transparent)

        if not hasattr(self, 'stitch_solid_bg_color'):
            self.stitch_solid_bg_color = QColor(Qt.GlobalColor.white)

        self.stitching_canvas.update_background(is_transparent=is_transparent, color=self.stitch_solid_bg_color)

    @Slot()
    def _select_mask_preview_color(self):
        if hasattr(self, '_is_color_dialog_open') and self._is_color_dialog_open:
            if hasattr(self, 'color_dialog_instance') and self.color_dialog_instance:
                self.color_dialog_instance.raise_()
                self.color_dialog_instance.activateWindow()
            return

        active_label = getattr(self, 'segmentation_overlay_label', None)
        if not active_label or active_label._is_previewing_mask:
            return

        self.color_dialog_instance = ModernColorDialog(self.selected_mask_color, self, title="选择蒙版预览颜色")
        self.color_dialog_instance.accepted.connect(self._handle_color_accepted)
        self.color_dialog_instance.finished.connect(self._on_color_dialog_finished)

        self._is_color_dialog_open = True
        self.color_dialog_instance.open()

    @Slot()
    def _handle_color_accepted(self):
        if not hasattr(self, 'color_dialog_instance') or not self.color_dialog_instance:
            return

        new_color = self.color_dialog_instance.currentColor()
        if not new_color.isValid():
            return

        active_label = getattr(self, 'segmentation_overlay_label', None)
        if not active_label:
            return

        self.selected_mask_color = new_color
        self._update_color_preview_button_style()
        active_label._clear_all_caches()
        active_label.update_display()

    def _update_color_preview_button_style(self):
        if not hasattr(self, 'select_mask_color_button'):
            return

        color = self.selected_mask_color
        text_color = "white" if color.lightnessF() < 0.5 else "black"

        self.select_mask_color_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color.name()};
                color: {text_color};
                border: 1px solid #888888;
                border-radius: {C_BORDER_RADIUS_SM};
                font-weight: bold;
                padding: 5px;
            }}
            QPushButton:hover {{
                border: 2px solid {C_PRIMARY};
            }}
        """)

    def _update_color_button_styles(self):
        if not hasattr(self, 'segment_overlay') or not self.segment_overlay or not hasattr(self, 'color_buttons'):
            return

        current_color_name = self.segment_overlay.mask_color_name
        button_diameter = next(iter(self.color_buttons.values())).width()
        border_radius = button_diameter // 2

        for name, btn in self.color_buttons.items():
            color_val = MASK_COLORS.get(name)
            if not color_val:
                continue

            color_hex = QColor(color_val).name()
            is_selected = (name == current_color_name)
            border_style = "border: 2px solid white;" if is_selected else "border: 2px solid transparent;"

            stylesheet = f"""
                QPushButton {{
                    background-color: {color_hex};
                    border-radius: {border_radius}px;
                    {border_style}
                }}
                QPushButton:hover {{
                    border: 2px solid #CCCCCC;
                }}
                QPushButton:pressed {{
                    background-color: {QColor(color_val).darker(120).name()};
                }}
            """
            btn.setStyleSheet(stylesheet)

    @Slot()
    def save_selected_stitched_item(self):
        """升级版：保存选中的单个素材（异步 UI 刷新防卡死）"""
        if not hasattr(self, 'stitching_canvas') or not self.stitching_canvas:
            return

        primary_item = self.stitching_canvas.get_primary_selected_item()
        if not primary_item:
            QMessageBox.warning(self, _TR("无法保存"), _TR("请先在画布上单选一个素材。"))
            return

        if self.is_saving:
            QMessageBox.information(self, _TR("忙碌"), _TR("其他保存操作正在进行中。"))
            return

        base_name = primary_item.name
        suffix = ""
        if getattr(primary_item, 'is_enhanced', False):
            suffix += "_enhanced"
        if getattr(primary_item, 'has_alpha_channel', False):
            suffix += "_matted"

        default_name = f"{base_name}{suffix}.png"
        orig_size = primary_item.pixmap.size()

        while True:
            dialog = ModernExportDialog('image', orig_size, default_name, parent=self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            params = dialog.get_export_params()
            save_path = params['path']

            if os.path.exists(save_path):
                reply = QMessageBox.question(
                    self, _TR("确认覆盖"),
                    _TR("文件 '{}' 已存在。\n\n您确定要覆盖此文件吗？").format(os.path.basename(save_path)),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    break
                else:
                    default_name = os.path.basename(save_path)
                    continue
            else:
                break

        # 先调出 UI 遮罩，并更新状态
        self.show_global_loading_overlay(_TR("正在进行超高分辨率缩放与编码..."), 0)
        self.is_saving = True
        self.update_button_states()

        # 【核心修复】：将极度耗时的 QImage 提取和线程启动包裹在闭包里
        def _do_export():
            try:
                # 这一步如果是几千万像素会卡死瞬间
                qimage_to_save = primary_item.pixmap.toImage()

                self.export_thread = AsyncImageExportThread(
                    qimage_to_save,
                    params['size'],
                    params['path'],
                    params['format'].upper()
                )
                self.export_thread.finished_signal.connect(self._on_async_image_export_finished)
                self.export_thread.finished_signal.connect(self.export_thread.deleteLater)
                self.export_thread.start()
            except Exception as e:
                self._on_async_image_export_finished(False, str(e))

        # 延迟 100 毫秒执行，让 Qt 的事件循环有足够的时间把遮罩和文字完美渲染出来！
        QTimer.singleShot(100, _do_export)

    @Slot()
    def start_quick_enhance_flow(self):
        """Launches the fast enhancement workflow sequence."""
        if self.stacked_widget.currentIndex() == self.CREATIVE_WORKSHOP_INDEX:
            self._enter_enhance_mode_for_quick_flow()
        else:
            target_card_container = self.welcome_page.card2_container
            if not target_card_container:
                QMessageBox.critical(self, "错误", "无法找到创意工坊功能卡片，无法启动快速高清。")
                return

            core_card_widget = target_card_container.property("core_content_widget")

            def after_transition_callback():
                self._enter_enhance_mode_for_quick_flow()

            self.switch_page_with_slide(
                self.CREATIVE_WORKSHOP_INDEX,
                clicked_widget=target_card_container,
                core_card_widget=core_card_widget,
                on_finished_callback=after_transition_callback
            )

    def _enter_enhance_mode_for_quick_flow(self):
        if hasattr(self, 'stitching_canvas') and self.stitching_canvas:
            self.stitching_canvas.clear_all(confirm=False)
        else:
            QMessageBox.critical(self, "错误", "创意工坊画布尚未初始化。")
            return

        fpath, _ = QFileDialog.getOpenFileName(
            self, "选择一张图片进行高清增强", "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.jfif)"
        )

        if not fpath:
            return

        self.stitching_canvas._process_dropped_files([fpath])
        QTimer.singleShot(100, self._open_tools_panel_for_quick_flow)

    def _open_tools_panel_for_quick_flow(self):
        if not (self.stitching_canvas and self.stitching_canvas.get_primary_selected_item()):
            QTimer.singleShot(100, self._open_tools_panel_for_quick_flow)
            return

        panel_key = "workshop_item_tools"
        tools_button = self._find_button_by_panel_key(panel_key)
        if not tools_button:
            return

        content_widget = self.page_specific_panels.get(panel_key)
        if not content_widget:
            content_widget = self._create_panel_content(panel_key)
            if not content_widget:
                return

        self._on_top_panel_button_clicked(tools_button, panel_key)

        if self.floating_panel_container.isHidden():
            self._show_actual_panel(tools_button, content_widget)

        if not tools_button.isChecked():
            tools_button.setChecked(True)

    def _start_worker(self, worker_id, worker_instance, run_method_name, *args):
        """
        Unified thread instantiator.
        [Safety Upgrade]: Forces thread-priority downgrades on CPU tasks and decouples parent-child
        node links to avoid heap corruption or GC issues during cross-thread object destruction.
        """
        worker_id_cn = WORKER_ID_TO_CN.get(worker_id, worker_id) if hasattr(self, 'WORKER_ID_TO_CN') else worker_id
        if worker_id in getattr(self, 'active_workers', {}):
            self.show_status_message(f"任务“{worker_id_cn}”已在运行中。", 3000)
            return None, None

        thread = QThread(self)
        worker = worker_instance

        try:
            worker.setParent(None)
        except Exception:
            pass

        worker.moveToThread(thread)
        worker.error.connect(self._on_worker_error)

        if hasattr(worker, 'progress'):
            worker.progress.connect(self._on_worker_progress)
        if hasattr(worker, 'log_message'):
            worker.log_message.connect(lambda msg, w_id=worker_id_cn: print(f"工作线程日志[{w_id}]: {msg}"))

        if worker_id == "load_image_model":
            worker.finished.connect(self._handle_load_image_model_completion)
        elif worker_id == "load_video_model":
            worker.finished.connect(self._handle_load_video_model_completion)
        elif worker_id == "enhance_canvas_item":
            worker.finished.connect(self._handle_enhance_for_canvas_item_completion)
        elif worker_id == "predict":
            worker.finished.connect(self._handle_predict_completion)
        elif worker_id == "extract":
            worker.finished.connect(self._handle_extract_completion)
        elif worker_id == "propagate_video_v1991":
            worker.finished.connect(self._handle_video_propagation_v1991_completion)
        elif worker_id == "save_stitched":
            worker.finished.connect(self._handle_save_stitched_completion)
        elif worker_id == "save_video":
            worker.finished.connect(self._handle_save_video_completion)
        elif worker_id == "init_video_state":
            worker.finished.connect(self._handle_init_video_state_completion)
        elif worker_id == "video_interact":
            worker.finished.connect(self._handle_video_interaction_completion)

        thread.started.connect(lambda w=worker, m=run_method_name, a=args: getattr(w, m)(*a))

        if hasattr(worker, 'finished'):
            worker.finished.connect(thread.quit)

        def make_safe_cleanup(t, w, wid):
            def cleanup():
                try:
                    w.moveToThread(QApplication.instance().thread())
                    w.deleteLater()
                except Exception:
                    pass
                try:
                    t.deleteLater()
                except Exception:
                    pass
                self._remove_active_worker(wid)
            return cleanup

        thread.finished.connect(make_safe_cleanup(thread, worker, worker_id))

        if not hasattr(self, 'active_workers'):
            self.active_workers = {}
        self.active_workers[worker_id] = (thread, worker)

        thread.setPriority(QThread.Priority.LowPriority)
        thread.start()

        self.show_status_message(f"任务“{worker_id_cn}”已开始...", 0)
        self.update_button_states()
        return thread, worker

    def _load_assets_into_grid(self, target_grid_layout: QGridLayout, num_columns: int = 2):
        """Updates elements currently populating local project libraries."""
        while target_grid_layout.count():
            child = target_grid_layout.takeAt(0)
            if child and child.widget():
                child.widget().deleteLater()
            elif child:
                del child

        asset_count = 0
        preset_asset_dir_to_load = ""
        try:
            base_path = sys._MEIPASS if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS') else os.getcwd()
            preset_asset_dir_to_load = os.path.join(base_path, "assets")
        except Exception:
            preset_asset_dir_to_load = "assets"

        if os.path.exists(preset_asset_dir_to_load):
            supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
            asset_files = [f for f in os.listdir(preset_asset_dir_to_load) if f.lower().endswith(supported_formats)]
            row = 0
            col = 0
            for filename in sorted(asset_files):
                filepath = os.path.join(preset_asset_dir_to_load, filename)
                thumbnail = AssetThumbnail(filepath, self)
                thumbnail.delete_requested.connect(self.delete_user_asset)
                target_grid_layout.addWidget(thumbnail, row, col)
                asset_count += 1
                col += 1
                if col >= num_columns:
                    col = 0
                    row += 1

        if os.path.exists(self.user_assets_config_path):
            try:
                with open(self.user_assets_config_path, 'r', encoding='utf-8') as f:
                    user_paths = [line.strip() for line in f if line.strip()]

                if target_grid_layout.count() == 0:
                    row = 0
                    col = 0

                for path in user_paths:
                    if os.path.exists(path):
                        is_duplicate = False
                        for i in range(target_grid_layout.count()):
                            widget_item = target_grid_layout.itemAt(i)
                            if widget_item:
                                widget = widget_item.widget()
                                if isinstance(widget, AssetThumbnail) and os.path.normpath(
                                        widget.image_path) == os.path.normpath(path):
                                    is_duplicate = True
                                    break
                        if not is_duplicate:
                            thumbnail = AssetThumbnail(path, self)
                            thumbnail.delete_requested.connect(self.delete_user_asset)
                            target_grid_layout.addWidget(thumbnail, row, col)
                            asset_count += 1
                            col += 1
                            if col >= num_columns:
                                col = 0
                                row += 1
            except Exception as e:
                self.log_message.emit(f"从配置文件加载用户素材时出错: {e}")

        if asset_count == 0:
            info_label = QLabel("素材库为空。\n点击下方“添加素材”按钮\n或确保 'assets' 文件夹存在。")
            info_label.setWordWrap(True)
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            target_grid_layout.addWidget(info_label, 0, 0, 1, num_columns)

    @Slot()
    def start_quick_segment_flow(self):
        """Starts image segmentation workflow on the active image file."""
        self.is_in_quick_segment_flow = True

        target_card_container = self.welcome_page.card2_container
        if not target_card_container:
            QMessageBox.critical(self, "错误", "无法找到创意工坊功能卡片，无法启动快速抠图。")
            self.is_in_quick_segment_flow = False
            return

        core_card_widget = target_card_container.property("core_content_widget")

        def after_transition_callback():
            self._enter_segmentation_mode_for_quick_flow()

        self.switch_page_with_slide(
            self.CREATIVE_WORKSHOP_INDEX,
            clicked_widget=target_card_container,
            core_card_widget=core_card_widget,
            on_finished_callback=after_transition_callback
        )

    def _enter_segmentation_mode_for_quick_flow(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "选择一张图片进行快速抠图", "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.jfif *.gif)"
        )

        if not fpath:
            self._exit_segmentation_mode()
            return

        try:
            pixmap = QPixmap(fpath)
            if pixmap.isNull():
                raise ValueError("无法加载图片为QPixmap")

            name = os.path.splitext(os.path.basename(fpath))[0]
            temp_item = StitchedImageItem(pixmap, name, original_path=fpath)
            self._enter_segmentation_mode(item_to_segment=temp_item)

        except Exception as e:
            QMessageBox.critical(self, "加载错误", f"加载图片进行快速抠图时出错:\n{e}")
            self._exit_segmentation_mode()

    def _handle_quick_action(self, action_type):
        if action_type == "open_file":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "打开文件", "",
                "所有支持的文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.mp4 *.mov *.avi *.mkv *.gif);;图像文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;视频文件 (*.mp4 *.mov *.avi *.mkv *.gif)"
            )
            if not file_path:
                return

            self.add_to_recent_projects(file_path)

            ext = os.path.splitext(file_path)[1].lower()
            if ext in SUPPORTED_VIDEO_FORMATS:
                target_page_index = self.VIDEO_SEG_PAGE_INDEX
                target_card_container = self.welcome_page.card1_container
                load_action = lambda: self._load_video_for_segmentation(file_path)
            else:
                target_page_index = self.CREATIVE_WORKSHOP_INDEX
                target_card_container = self.welcome_page.card2_container
                load_action = lambda: self.stitching_canvas._process_dropped_files([file_path])

            if target_card_container:
                core_card_widget = target_card_container.property("core_content_widget")
                self.switch_page_with_slide(target_page_index, clicked_widget=target_card_container,
                                            core_card_widget=core_card_widget)
                QTimer.singleShot(700, load_action)

        elif action_type == "from_clipboard":
            clipboard = QApplication.clipboard()
            mime_data = clipboard.mimeData()

            if mime_data.hasUrls():
                urls = mime_data.urls()
                if urls:
                    file_path = urls[0].toLocalFile()
                    self.add_to_recent_projects(file_path)

                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in SUPPORTED_VIDEO_FORMATS:
                        target_page_index = self.VIDEO_SEG_PAGE_INDEX
                        target_card_container = self.welcome_page.card1_container
                        load_action = lambda: self._load_video_for_segmentation(file_path)
                    elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'):
                        target_page_index = self.CREATIVE_WORKSHOP_INDEX
                        target_card_container = self.welcome_page.card2_container
                        load_action = lambda: self.stitching_canvas._process_dropped_files([file_path])
                    else:
                        QMessageBox.information(self, "提示", "剪贴板中的文件类型不受支持。")
                        return

                    if target_card_container:
                        core_card_widget = target_card_container.property("core_content_widget")
                        self.switch_page_with_slide(target_page_index, clicked_widget=target_card_container,
                                                    core_card_widget=core_card_widget)
                        QTimer.singleShot(700, load_action)
                    return

            if mime_data.hasImage():
                workshop_card_container = self.welcome_page.card2_container
                core_card_widget = workshop_card_container.property("core_content_widget")
                self.switch_page_with_slide(self.CREATIVE_WORKSHOP_INDEX, clicked_widget=workshop_card_container,
                                            core_card_widget=core_card_widget)
                QTimer.singleShot(700, self._load_from_clipboard_action)
                return

            QMessageBox.information(self, "提示", "剪贴板中没有检测到有效的图像或支持的文件。")

    def populate_recent_projects(self):
        if not hasattr(self, 'welcome_page') or not hasattr(self.welcome_page, 'recent_projects_list'):
            return

        list_widget = self.welcome_page.recent_projects_list
        list_widget.clear()

        recent_files = self.settings.value("recent_files", [], type=list)

        if not recent_files:
            item = QListWidgetItem("暂无历史记录")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            list_widget.addItem(item)
            return

        for file_path in recent_files:
            if os.path.exists(file_path):
                item_widget = RecentProjectItemWidget(file_path)
                list_item = QListWidgetItem(list_widget)
                list_item.setSizeHint(item_widget.sizeHint())
                list_item.setData(Qt.ItemDataRole.UserRole, file_path)
                list_widget.addItem(list_item)
                list_widget.setItemWidget(list_item, item_widget)

    def add_to_recent_projects(self, file_path):
        if not file_path:
            return

        recent_files = self.settings.value("recent_files", [], type=list)
        norm_path = os.path.normpath(file_path)

        if norm_path in [os.path.normpath(p) for p in recent_files]:
            recent_files = [p for p in recent_files if os.path.normpath(p) != norm_path]

        recent_files.insert(0, file_path)
        recent_files = recent_files[:15]

        self.settings.setValue("recent_files", recent_files)

        if self.stacked_widget.currentIndex() == getattr(self, 'WELCOME_PAGE_INDEX', 0):
            self.populate_recent_projects()

    @Slot(QListWidgetItem)
    def _load_recent_project(self, item: QListWidgetItem):
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "文件无效", f"无法找到文件:\n{file_path}\n\n该记录将被移除。")
            self.remove_from_recent_projects(file_path)
            return

        self.add_to_recent_projects(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        if ext in SUPPORTED_VIDEO_FORMATS:
            target_page_index = self.VIDEO_SEG_PAGE_INDEX
            target_card_container = self.welcome_page.card1_container
            load_action = lambda: self._load_video_for_segmentation(file_path)
        else:
            target_page_index = self.CREATIVE_WORKSHOP_INDEX
            target_card_container = self.welcome_page.card2_container
            load_action = lambda: self.stitching_canvas._process_dropped_files([file_path])

        if target_card_container:
            core_card_widget = target_card_container.property("core_content_widget")
            self.switch_page_with_slide(
                target_page_index,
                clicked_widget=target_card_container,
                core_card_widget=core_card_widget,
                on_finished_callback=load_action
            )
        else:
            self.switch_page(target_page_index)
            QTimer.singleShot(50, load_action)

    @Slot()
    def _load_from_clipboard_action(self):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        pixmap_to_add = None

        if mime_data.hasImage():
            image = clipboard.image()
            if not image.isNull():
                pixmap_to_add = QPixmap.fromImage(image)

        if pixmap_to_add is None and mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                file_path = urls[0].toLocalFile()
                supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
                if file_path.lower().endswith(supported_formats):
                    if hasattr(self, 'stitching_canvas'):
                        self.add_to_recent_projects(file_path)
                        self.stitching_canvas._process_dropped_files([file_path])
                        self.show_status_message("已从剪贴板添加图像文件。", 3000)
                    return

        if pixmap_to_add is None:
            pixmap_variant = clipboard.pixmap()
            if isinstance(pixmap_variant, QPixmap) and not pixmap_variant.isNull():
                pixmap_to_add = pixmap_variant

        if pixmap_to_add is not None:
            if not hasattr(self, 'stitching_canvas'):
                QMessageBox.warning(self, "错误", "创意工坊画布尚未初始化。")
                return

            temp_dir = os.path.join(TEMP_BASE_DIR, "clipboard")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"clipboard_{uuid.uuid4().hex[:8]}.png")

            try:
                if pixmap_to_add.save(temp_path, "PNG"):
                    self.add_to_recent_projects(temp_path)
                    self.stitching_canvas._process_dropped_files([temp_path])
                    self.show_status_message("已从剪贴板添加图像。", 3000)
                else:
                    raise IOError("保存剪贴板图像到临时文件失败。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"处理剪贴板图像时出错: {e}")
        else:
            QMessageBox.information(self, "提示", "剪贴板中没有检测到有效的图像或图像文件。")

    def _load_settings(self):
        """Loads animation parameters, UI configuration and GPU state configs from physical settings ini."""
        self.use_blur_animation = self.settings.value("animation/use_blur", True, type=bool)
        self.use_gpu_acceleration = self.settings.value("hardware/use_gpu", True, type=bool)

        if hasattr(self, 'animation_toggle_switch') and isinstance(self.animation_toggle_switch, ToggleSwitch):
            self.animation_toggle_switch.setChecked(self.use_blur_animation)

        if hasattr(self, 'hw_accel_toggle_switch') and isinstance(self.hw_accel_toggle_switch, ToggleSwitch):
            self.hw_accel_toggle_switch.blockSignals(True)
            self.hw_accel_toggle_switch.setChecked(self.use_gpu_acceleration)
            self.hw_accel_toggle_switch.blockSignals(False)

        asset_panel_visible = self.settings.value("stitching/asset_panel_visible", False, type=bool)
        if hasattr(self, 'stitch_asset_library_toggle_button') and self.stitch_asset_library_toggle_button:
            if self.stitch_asset_library_toggle_button.isChecked() != asset_panel_visible:
                self.stitch_asset_library_toggle_button.blockSignals(True)
                self.stitch_asset_library_toggle_button.setChecked(asset_panel_visible)
                self.stitch_asset_library_toggle_button.blockSignals(False)

    def _save_settings(self):
        """Writes current config properties back to settings ini file."""
        self.settings.setValue("animation/use_blur", self.use_blur_animation)

        if hasattr(self, 'hw_accel_toggle_switch'):
            self.settings.setValue("hardware/use_gpu", self.use_gpu_acceleration)

        if hasattr(self, 'stitch_asset_library_toggle_button') and self.stitch_asset_library_toggle_button:
            self.settings.setValue("stitching/asset_panel_visible", self.stitch_asset_library_toggle_button.isChecked())

        if self.settings.contains("recent_files"):
            self.settings.sync()

    def _create_hw_accel_toggle_switch(self) -> QWidget:
        """Instantiates hardware acceleration UI switches."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        has_hardware = torch.cuda.is_available() or (
                    hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
        label_text = "GPU 加速" if has_hardware else "GPU 加速 (无硬件)"

        label = QLabel(label_text)
        label.setStyleSheet("color: #6D6D72; font-size: 11pt; font-weight: bold; background-color: transparent;")

        self.hw_accel_toggle_switch = ToggleSwitch(self)

        if not has_hardware:
            self.hw_accel_toggle_switch.setChecked(False)
            self.hw_accel_toggle_switch.setEnabled(False)

        layout.addWidget(label)
        layout.addWidget(self.hw_accel_toggle_switch)

        self.hw_accel_toggle_switch.toggled.connect(self._on_hw_accel_toggle_changed)
        return container

    @Slot()
    def start_batch_auto_segment_flow(self):
        """Launches batch model segmentation interface workflows."""
        if getattr(self, 'is_predicting', False) or getattr(self, 'is_enhancing', False):
            QMessageBox.information(self, _TR("忙碌"), _TR("其他 AI 任务正在运行中，请稍后再试。"))
            return

        files, _ = QFileDialog.getOpenFileNames(
            self, _TR("选择多张图像进行批量一键抠图"), "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff *.jfif)"
        )
        if not files:
            return

        out_dir = QFileDialog.getExistingDirectory(self, _TR("选择保存批量抠图结果的文件夹"))
        if not out_dir:
            return

        # 彻底移除老的 QProgressDialog，只呼出全局现代遮罩
        self.show_global_loading_overlay(f"准备开始批量处理 {len(files)} 张图像...", 0)

        worker = BatchMattingWorker()
        worker_id = "batch_matting"

        thread, w = self._start_worker(worker_id, worker, "run_batch", files, out_dir)
        w.finished.connect(self._handle_batch_matting_completion)

    @Slot(int, bool, str)
    def _handle_batch_matting_completion(self, success_count, success, error_message):
        # 【核心修复】：关闭并隐藏现代全局遮罩层
        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.hide()

        # 清除可能残留的老进度条对象
        if hasattr(self, 'batch_progress_dialog') and self.batch_progress_dialog:
            try:
                self.batch_progress_dialog.close()
                self.batch_progress_dialog.deleteLater()
            except RuntimeError:
                pass
            self.batch_progress_dialog = None

        if success:
            QMessageBox.information(self, _TR("批量抠图完成"),
                                    f"{_TR('批量处理结束！')}\n{_TR('成功生成')} {success_count} {_TR('张透明图像。')}")
            self.show_status_message(f"批量抠图成功处理 {success_count} 张。", 5000)
        else:
            if "取消" in error_message or "cancel" in error_message.lower():
                self.show_status_message(f"批量抠图已中止，已成功处理 {success_count} 张。", 3000)
            else:
                QMessageBox.critical(self, _TR("批量抠图失败"),
                                     f"{_TR('处理过程中断:')}\n{error_message}\n{_TR('已成功输出')} {success_count} {_TR('张。')}")
                self.show_status_message("批量抠图发生错误。", 5000)

        if hasattr(self, 'batch_matting_page'):
            self.batch_matting_page.on_processing_finished()

    @Slot(bool)
    def _on_hw_accel_toggle_changed(self, checked: bool):
        """
        Handles state change of the GPU hardware acceleration switch.
        Safely flushes VRAM, resets active worker thread structures,
        and reloads model components in the background.
        """
        import gc
        self.use_gpu_acceleration = checked
        self._save_settings()

        new_device = self.get_current_device()
        mode_name = "GPU" if new_device.type in ['cuda', 'mps'] else "CPU"

        self.show_global_loading_overlay(f"正在切换为 {mode_name} 模式\n正在重置硬件显存与底层模型，请稍候。")
        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.repaint()
        QApplication.processEvents()

        global device
        device = new_device
        if 'core.workers' in sys.modules:
            sys.modules['core.workers'].device = new_device

        if getattr(self, 'video_predictor', None) and getattr(self, 'video_inference_state', None):
            try:
                self.video_predictor.reset_state(self.video_inference_state)
            except Exception:
                pass

        self.video_inference_state = None

        if hasattr(self, 'sam_prediction_cache'):
            self.sam_prediction_cache.clear()

        self.image_predictor = None
        self.image_predictor_loaded = False
        self.sam_image_load_failed = False
        self.video_predictor = None
        self.video_predictor_loaded = False
        self.sam_video_load_failed = False
        self.matteformer_model = None
        self.matteformer_loaded = False

        try:
            PredictWorker.clear_models_cache()
        except Exception:
            pass

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        self.is_loading_model = True
        self.update_button_states()

        self._reload_worker = HeadlessLoader()
        self._reload_worker.loading_complete.connect(self._on_headless_reload_complete)
        self._reload_worker.start()

    @Slot()
    def _on_global_extraction_finished(self):
        """
        视频合并/提取完成中枢。
        【核心修复】：彻底阻断冗余的全局重新烘焙！如果因剪裁丢失了蒙版烘焙缓存，在此处拦截并启动极速单片段渲染修复。
        """
        if getattr(self, 'current_extract_progress', None):
            try:
                self.current_extract_progress.setValue(100)
                self.current_extract_progress.close()
                self.current_extract_progress.deleteLater()
            except Exception:
                pass
            self.current_extract_progress = None

        extractor = self.sender()
        if extractor is None:
            extractor = getattr(self, '_timeline_extractor', None)

        if extractor is None:
            return

        success = getattr(extractor, 'success', False)
        error_msg = getattr(extractor, 'error_msg', "")

        if extractor != getattr(self, '_timeline_extractor', None):
            extractor.deleteLater()
            return

        self._timeline_extractor.deleteLater()
        self._timeline_extractor = None

        if success:
            self.is_extracting_frames = False
            self.video_inference_state = None

            # 重新生成最新的全局高清帧路径列表，防止智能抠图的 Scrubber 缩略图错位
            self.video_thumbnail_paths = [
                os.path.join(self.temp_frame_dir, f"{i:05d}.jpg")
                for i in range(self.total_frames)
            ]

            has_existing_masks = bool(getattr(self, 'processed_masks', {}))

            # =========================================================================
            # 【核心修复】：如果是因为剪辑带有蒙版的片段丢失了缓存，则拦截提取流程并先触发单片段重烘焙！
            # =========================================================================
            pending_rebake_idx = getattr(self, '_pending_crop_rebake_clip_idx', -1)
            if pending_rebake_idx != -1:
                self._pending_crop_rebake_clip_idx = -1
                self._skip_rebake_on_next_extraction = False  # 清理免烘焙通行证，交由下方处理
                self._rebake_clip_after_crop(pending_rebake_idx)
                return

            # 捕获我们在故事板增减/移动视频时设置的免烘焙通行证
            skip_heavy_rebake = getattr(self, '_skip_rebake_on_next_extraction', False)

            if has_existing_masks:
                if skip_heavy_rebake:
                    # 【性能飞跃点】：不调用重新烘焙，直接利用底层硬链接重组各片段的历史烘焙产物！
                    self._skip_rebake_on_next_extraction = False
                    self.show_global_loading_overlay("正在极速链结历史流畅播放轨道...", 90)

                    # 调用系统级极速链结组合，几乎 0 毫秒耗时
                    assemble_success = self._assemble_global_render_dir()
                    self._handle_pre_render_completion_after_extraction(assemble_success, "高速链结完成")
                    return
                else:
                    # 只有在发生破坏性修改（如更改了全局背景、或因非预期情况丢失缓存）时，才触发兜底的全局重烘焙
                    base_name = "rendered_timeline"
                    session_id = str(uuid.uuid4())[:8]
                    self.temp_render_dir = os.path.join(TEMP_BASE_DIR, f"render_{base_name}_{session_id}")

                    self.show_global_loading_overlay("时间线结构更新，正在重组片段渲染轨道...", 0)

                    self._prerender_thread = QThread(self)

                    bg_color = getattr(self, 'video_save_bg_color', QColor(0, 255, 0))
                    custom_bg_path = getattr(self, 'video_save_bg_image_path', None)
                    bg_is_transparent = getattr(self, 'video_save_bg_is_transparent', False)

                    self._prerender_worker = PreRenderMattedVideoWorker(
                        temp_frame_dir=self.temp_frame_dir,
                        processed_masks=self.processed_masks,
                        total_frames=self.total_frames,
                        target_w=self.video_width or 1280,
                        target_h=self.video_height or 720,
                        bg_color=bg_color,
                        custom_bg_path=custom_bg_path,
                        bg_is_transparent=bg_is_transparent,
                        temp_render_dir=self.temp_render_dir,
                        virtual_timeline=self.virtual_timeline
                    )

                    self._prerender_worker.moveToThread(self._prerender_thread)
                    self._prerender_worker.progress.connect(
                        lambda pct, msg: self.show_global_loading_overlay(msg, pct)
                    )
                    self._prerender_worker.finished.connect(self._handle_pre_render_completion_after_extraction)

                    self._prerender_thread.started.connect(self._prerender_worker.run)
                    self._prerender_thread.start()
                    return

            # 如果当前工程根本没有任何抠图蒙版，直接进行 UI 收尾即可
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            self._finalize_post_extraction_ui()

        else:
            self.is_extracting_frames = False
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()

            self._pending_mode_switch = None
            QMessageBox.critical(self, "视频流合并失败",
                                 f"无法完成物理帧同步，请检查视频文件是否受损。\n\n错误信息: {error_msg}")
            if hasattr(self, 'update_button_states'):
                self.update_button_states()

    def _rebake_clip_after_crop(self, clip_idx):
        """
        剪裁完成后专用的局部重烘焙流程，仅渲染当前刚剪裁过的片段
        """
        clip = self.virtual_timeline[clip_idx]
        local_baked_dir = os.path.join(TEMP_BASE_DIR, f"baked_clip_{uuid.uuid4().hex[:8]}")
        global_start = sum(v['frames'] for v in self.virtual_timeline[:clip_idx])
        global_end = global_start + clip['frames'] - 1

        self.show_global_loading_overlay("正在重构剪裁后的发丝图层...", 0)

        self._single_bake_thread = QThread(self)
        self._single_bake_worker = BakeSingleClipWorker(
            raw_frame_dir=self.temp_frame_dir,
            processed_masks=self.processed_masks,
            start_frame=global_start,
            end_frame=global_end,
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

        def on_rebuild_complete(success, result_msg):
            self._single_bake_thread.quit()
            self._single_bake_thread.wait()
            self._single_bake_thread.deleteLater()
            self._single_bake_worker.deleteLater()

            if success:
                clip['baked_preview_dir'] = result_msg
                # 重组渲染目录，并流转进入 UI 完美恢复与锁定解锁
                self._assemble_global_render_dir()
                self._handle_pre_render_completion_after_extraction(True, "局部重构完成")
            else:
                self._handle_pre_render_completion_after_extraction(False, result_msg)

        self._single_bake_worker.finished.connect(on_rebuild_complete)
        self._single_bake_thread.started.connect(self._single_bake_worker.run)
        self._single_bake_thread.start()

    @Slot(bool, str)
    def _handle_pre_render_completion_after_extraction(self, success, result_msg):
        """
        剪辑/合并后快速烘焙完毕（或极速硬链结完毕）的统一收尾槽函数。
        """
        if hasattr(self, '_prerender_thread'):
            self._prerender_thread.quit()
            self._prerender_thread.wait()
            self._prerender_thread.deleteLater()
            del self._prerender_thread
        if hasattr(self, '_prerender_worker'):
            self._prerender_worker.deleteLater()
            del self._prerender_worker

        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.hide()

        if success:
            # 标记抠图就绪，通知播放器自动挂载高速预渲染 JPG 硬盘缓存，杜绝 CPU 卡顿！
            self.video_segmentation_finished = True

            # 清空内存预读池，防止读到老版本帧
            if hasattr(self, '_video_frame_cache'):
                self._video_frame_cache.clear()

            self._finalize_post_extraction_ui()
        else:
            self.video_segmentation_finished = False
            QMessageBox.warning(self, "重建轨道失败",
                                f"时间线合并后未能建立流畅播放轨道: {result_msg}\n这可能导致接下来的播放出现卡顿。")
            self._finalize_post_extraction_ui()

    def _finalize_post_extraction_ui(self):
        """
        公共辅助方法：统一调度抽取/合并完成后的 UI 刷新与视口寻道恢复
        """
        pending_mode = getattr(self, '_pending_mode_switch', None)
        if pending_mode == "matting":
            self._pending_mode_switch = None
            QTimer.singleShot(100, self._enter_dedicated_matting_mode)
        elif pending_mode == "crop":
            self._pending_mode_switch = None
            QTimer.singleShot(100, self._enter_dedicated_crop_mode)
        else:
            if getattr(self, 'current_frame_index', -1) != -1:
                QTimer.singleShot(50, lambda: self._display_frame_wrapper(self.current_frame_index))

        if hasattr(self, 'video_display_label'):
            is_matting_page = getattr(self, 'vid_editor_stack', None) and \
                              self.vid_editor_stack.currentWidget() == getattr(self, 'vid_dedicated_matting_page', None)
            self.video_display_label.set_allow_interaction(is_matting_page)
            self.video_display_label.update_cursor()

        if hasattr(self, 'update_button_states'):
            self.update_button_states()



    @Slot(dict)
    def _on_ram_preload_finished(self, preloaded_dict):
        """
        内存预热已废弃，桩函数防错。
        """
        pass

    def _create_animation_toggle_switch(self) -> QWidget:
        """Helper to create and return the animation interface configuration widget."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        label = QLabel("虚化动画")
        label.setStyleSheet("color: #6D6D72; font-size: 11pt; font-weight: bold; background-color: transparent;")

        self.animation_toggle_switch = ToggleSwitch(self)

        layout.addWidget(label)
        layout.addWidget(self.animation_toggle_switch)

        self.animation_toggle_switch.toggled.connect(self._on_animation_toggle_changed)
        return container

    @Slot(bool)
    def _on_animation_toggle_changed(self, checked: bool):
        self.use_blur_animation = checked
        self._save_settings()

    @Slot(int)
    def _on_tile_mode_changed(self, index: int):
        """Handles switching between upscaler tiling options to adapt for VRAM profiles."""
        if not hasattr(self, 'tile_mode_combo') or not hasattr(self, 'custom_tile_size_spinbox'):
            return

        selected_text = self.tile_mode_combo.itemText(index).strip()
        is_custom = selected_text in ["自定义", "Custom"]

        self.custom_tile_size_spinbox.setVisible(is_custom)
        self.custom_tile_size_spinbox.setEnabled(is_custom)

        if selected_text in ["自动 (推荐)", "Auto (Rec)", "Auto"]:
            self.tile_mode_combo.setToolTip("自动根据图像分辨率选择合适的分块大小，平衡速度与显存。")
        elif selected_text in ["自定义", "Custom"]:
            self.tile_mode_combo.setToolTip("手动指定分块大小（越小越省显存，但处理更慢且可能有接缝）。")
        elif selected_text in ["大图模式", "Large Image"]:
            self.tile_mode_combo.setToolTip("强制使用 128 分块，适合极高分辨率（极低显存占用）。")
        elif selected_text in ["中图模式", "Medium Image"]:
            self.tile_mode_combo.setToolTip("强制使用 256 分块，适合 2K-4K 图像（较低显存占用）。")
        elif selected_text in ["小图模式", "Small Image"]:
            self.tile_mode_combo.setToolTip("强制使用 512 分块，适合 1080P 以下（较高显存占用）。")
        else:
            self.tile_mode_combo.setToolTip("选择分块处理模式以平衡显存占用和处理速度。")

    @Slot()
    def _enhance_selected_item_action(self):
        """Launches the upscaling enhancement thread worker on the currently highlighted canvas item."""
        if not self.stitching_canvas or not self.stitching_canvas.get_primary_selected_item():
            return

        self.item_being_enhanced = self.stitching_canvas.get_primary_selected_item()

        try:
            temp_dir = os.path.join(TEMP_BASE_DIR, "enhance_cache")
        except Exception:
            temp_dir = os.path.join(os.getcwd(), "temp", "enhance_cache")

        os.makedirs(temp_dir, exist_ok=True)
        temp_input_path = os.path.join(temp_dir, f"temp_enhance_input_{uuid.uuid4().hex[:8]}.png")

        if not self.item_being_enhanced.pixmap.save(temp_input_path, "PNG"):
            QMessageBox.critical(self, _TR("错误"), _TR("无法提取当前图像数据进行增强。"))
            return

        if hasattr(self, 'image_compare_widget_enhance'):
            original_pixmap = self.item_being_enhanced.pixmap
            if not original_pixmap.isNull():
                self.image_compare_widget_enhance.set_images(original=original_pixmap, enhanced=None)
            else:
                self.image_compare_widget_enhance.clear_content()

        try:
            scale = int(self.enhance_scale_combo.currentText().replace('x', ''))
        except Exception:
            scale = 4

        selected_tile_mode_text = self.tile_mode_combo.currentText().strip()
        tile_value = 0

        if selected_tile_mode_text in ["自动 (推荐)", "Auto (Rec)", "Auto"]:
            try:
                pixmap_for_size = self.item_being_enhanced.pixmap
                img_w, img_h = pixmap_for_size.width(), pixmap_for_size.height()
                longest_side = max(img_w, img_h)
                if longest_side > 2000:
                    tile_value = 128
                elif longest_side > 1080:
                    tile_value = 256
                else:
                    tile_value = 512
            except Exception:
                tile_value = 0
        elif selected_tile_mode_text in ["自定义", "Custom"]:
            if hasattr(self, 'custom_tile_size_spinbox'):
                tile_value = self.custom_tile_size_spinbox.value()
        elif selected_tile_mode_text in ["大图模式", "Large Image"]:
            tile_value = 128
        elif selected_tile_mode_text in ["中图模式", "Medium Image"]:
            tile_value = 256
        elif selected_tile_mode_text in ["小图模式", "Small Image"]:
            tile_value = 512
        else:
            tile_value = 0

        tile_pad_value = 10
        model_name = self.enhance_model_combo.currentText()

        try:
            model_filename = ENHANCE_MODELS.get(DEFAULT_ENHANCE_MODEL_NAME)
            if "动漫" in model_name or "Anime" in model_name:
                model_filename = ENHANCE_MODELS.get("动漫")
            elif "通用" in model_name or "General" in model_name:
                model_filename = ENHANCE_MODELS.get("通用")
        except Exception:
            model_filename = None

        if not model_filename:
            QMessageBox.critical(self, _TR("错误"), _TR("选择的增强模型无效。"))
            return

        resolved_enhance_model_path = os.path.join(
            sys._MEIPASS if getattr(sys, 'frozen', False) else os.getcwd(), model_filename
        )

        self.is_enhancing = True
        self.update_button_states()

        # 【核心修复1】：呼出全局遮罩，防止用户乱点导致假死
        self.show_global_loading_overlay(_TR(f"正在进行 AI 超清放大，请勿操作画布..."), 0)

        # 【核心修复2】：启动心跳定时器，强制主线程刷新，彻底解决“系统未响应”
        if hasattr(self, '_enhance_heartbeat_timer') and self._enhance_heartbeat_timer:
            try:
                self._enhance_heartbeat_timer.stop()
                self._enhance_heartbeat_timer.deleteLater()
            except Exception:
                pass
        self._enhance_heartbeat_timer = QTimer(self)
        self._enhance_heartbeat_timer.setInterval(50)
        self._enhance_heartbeat_timer.timeout.connect(self._pump_gui_heartbeat)
        self._enhance_heartbeat_timer.start()

        worker = EnhanceWorker()
        self._start_worker(
            "enhance_canvas_item", worker, "run_enhance",
            temp_input_path,
            resolved_enhance_model_path,
            ENHANCE_FIXED_DENOISE, scale, self.get_current_device(),
            tile_value, tile_pad_value
        )

    @Slot(object, bool, str)
    def jump_to_refine_from_batch(self, card, mask_bool):
        """
        Seamless refinement routing method.
        Imports an item from the batch view into the Creative Workshop editor
        and applies its current preview mask for detailed canvas manipulation.
        """
        self._editing_batch_card = card
        self.switch_page(getattr(self, 'CREATIVE_WORKSHOP_INDEX', 2))

        cv_img = imread_unicode(card.file_path, cv2.IMREAD_COLOR)
        if cv_img is None:
            return

        reader = QImageReader(card.file_path)
        size = reader.size()
        real_w = size.width() if size.isValid() else cv_img.shape[1]
        real_h = size.height() if size.isValid() else cv_img.shape[0]

        canvas_w = self.stitching_canvas.canvas_size.width()
        canvas_h = self.stitching_canvas.canvas_size.height()
        target_w, target_h = canvas_w * 0.7, canvas_h * 0.7

        scale_ratio = 1.0
        if real_w > 0 and real_h > 0:
            scale_w = target_w / real_w
            scale_h = target_h / real_h
            scale_ratio = min(scale_w, scale_h)

        if real_w * scale_ratio > real_w and real_h * scale_ratio > real_h:
            scale_ratio = 1.0

        final_w = max(10, int(real_w * scale_ratio))
        final_h = max(10, int(real_h * scale_ratio))

        fast_preview = QPixmap(final_w, final_h)
        fast_preview.fill(QColor(40, 40, 40, 180))

        item = StitchedImageItem(fast_preview, os.path.basename(card.file_path), card.file_path)
        item.size = QSizeF(final_w, final_h)
        item.pos = self.stitching_canvas.canvas_pos + QPointF(canvas_w / 2, canvas_h / 2) - QPointF(final_w / 2,
                                                                                                    final_h / 2)

        self.start_async_image_load(card.file_path, item.id)

        if mask_bool is not None:
            if getattr(card, 'bgra_img_cv', None) is not None:
                alpha_u8 = card.bgra_img_cv[:, :, 3]
                float_mask = alpha_u8.astype(np.float32) / 255.0
                item.segmentation_mask_np = float_mask
            else:
                mask_u8 = mask_bool.astype(np.uint8) * 255
                mask_soft = cv2.GaussianBlur(mask_u8, (5, 5), 0)
                item.segmentation_mask_np = mask_soft.astype(np.float32) / 255.0
            item.has_alpha_channel = False
        else:
            item.has_alpha_channel = False
            item.segmentation_mask_np = None

        self.stitching_canvas.items.append(item)
        self.stitching_canvas.selected_items = [item]
        self.stitching_canvas.layers_changed.emit()
        self.stitching_canvas.selection_changed.emit()
        self.stitching_canvas.update()

        self._enter_segmentation_mode(item)

        if mask_bool is not None:
            if hasattr(self, 'segmentation_overlay_label'):
                self.segmentation_overlay_label.set_mask_preview_mode(False)
                self.segmentation_overlay_label.set_interaction_mode('paint')
            if hasattr(self, 'seg_preview_tool'):
                self.seg_preview_tool.setChecked(False)
            if hasattr(self, 'seg_paint_mode_tool'):
                self.seg_paint_mode_tool.setChecked(True)

        self.update_button_states()

    def _execute_batch_matting(self, file_paths, output_dir, ui_cards_list):
        """Initiates parallelized background processing tasks for batch views."""
        # 彻底移除老的 QProgressDialog，只呼出全局现代遮罩
        self.show_global_loading_overlay(_TR("正在高速提取主图像素..."), 0)

        worker = BatchMattingWorker()

        def on_single_result(idx, path, bgra, mask, err):
            if idx < len(ui_cards_list):
                ui_cards_list[idx].update_result(bgra, mask, err)

        worker.single_result_ready.connect(on_single_result)

        thread, w = self._start_worker("batch_matting", worker, "run_batch", file_paths, output_dir)
        w.finished.connect(self._on_batch_matting_done)

    @Slot(int, bool, str)
    def _on_batch_matting_done(self, success_count, success, err_msg):
        # 【核心修复】：关闭并隐藏现代全局遮罩层
        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.hide()

        # 清除可能残留的老进度条对象
        if hasattr(self, 'batch_progress') and self.batch_progress:
            try:
                self.batch_progress.close()
                self.batch_progress.deleteLater()
            except RuntimeError:
                pass
            self.batch_progress = None

        if success:
            QMessageBox.information(self, _TR("计算完成"),
                                    f"{_TR('已成功抠出')} {success_count} {_TR('张图像！')}\n{_TR('您可以点击卡片进行对比或精修，确认无误后点击顶部【导出全部】保存到电脑。')}")

        if hasattr(self, 'batch_matting_page'):
            self.batch_matting_page.on_processing_finished()

    @Slot(object, bool, str)
    def _handle_enhance_for_canvas_item_completion(self, result_cv_image, success, error_message):
        """画布素材增强回调：处理 4K 巨图拼接防止卡顿假死"""
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        # 【核心修复3】：运算完毕，停止并销毁心跳定时器
        if hasattr(self, '_enhance_heartbeat_timer') and self._enhance_heartbeat_timer:
            try:
                self._enhance_heartbeat_timer.stop()
                self._enhance_heartbeat_timer.deleteLater()
            except Exception:
                pass
            self._enhance_heartbeat_timer = None

        if hasattr(self, 'current_enhance_canvas_item_progress') and getattr(self,
                                                                             'current_enhance_canvas_item_progress'):
            try:
                self.current_enhance_canvas_item_progress.close()
            except Exception:
                pass

        if success and result_cv_image is not None and self.item_being_enhanced:
            item = self.item_being_enhanced

            # 【核心修复4】：维持遮罩，并延迟处理巨幅图表转换
            self.show_global_loading_overlay(_TR("正在将超清材质投射到物理画布..."), 0)

            def _do_update_canvas():
                try:
                    # 极度耗时的同步主线程图像映射
                    enhanced_pixmap = convert_cv_to_pixmap(result_cv_image)

                    if enhanced_pixmap.isNull():
                        QMessageBox.warning(self, _TR("增强失败"), _TR("无法将增强结果转换为有效图像。"))
                    else:
                        if hasattr(self,
                                   'image_compare_widget_enhance') and self.image_compare_widget_enhance.original_pixmap:
                            self.image_compare_widget_enhance.set_images(
                                original=self.image_compare_widget_enhance.original_pixmap,
                                enhanced=enhanced_pixmap
                            )

                        item.pixmap = enhanced_pixmap
                        new_raw_size = QSizeF(item.pixmap.size())

                        canvas_w = self.stitching_canvas.canvas_size.width()
                        canvas_h = self.stitching_canvas.canvas_size.height()
                        scale_factor = 1.0

                        if new_raw_size.width() > canvas_w or new_raw_size.height() > canvas_h:
                            scale_w = canvas_w / new_raw_size.width() if new_raw_size.width() > 0 else 1
                            scale_h = canvas_h / new_raw_size.height() if new_raw_size.height() > 0 else 1
                            scale_factor = min(scale_w, scale_h) * 0.95

                        old_center = item.get_transformed_bounding_rect().center()
                        item.size = new_raw_size * scale_factor
                        item.pos = old_center - QPointF(item.size.width() / 2, item.size.height() / 2)
                        item.rotation = 0
                        item.is_enhanced = True

                        self.stitching_canvas.update()
                        if hasattr(self, 'stitching_canvas'):
                            self.stitching_canvas.selection_changed.emit()
                        self.show_status_message(f"素材 '{item.name}' 高清增强完成。", 3000)
                finally:
                    if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                        self._global_loading_overlay.hide()
                    self.is_enhancing = False
                    self.item_being_enhanced = None
                    self.update_button_states()

            # 闭包注入 Qt 事件循环延迟 100ms
            QTimer.singleShot(100, _do_update_canvas)

        else:
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            if not success:
                QMessageBox.warning(self, _TR("增强失败"), f"{_TR('增强素材失败:')} {error_message}")
                if hasattr(self, 'image_compare_widget_enhance'):
                    self.image_compare_widget_enhance.clear_content()

            self.is_enhancing = False
            self.item_being_enhanced = None
            self.update_button_states()

    def _connect_dynamic_panel_signals(self):
        """Binds dynamic panel properties and tool actions with appropriate execution handlers."""

        def safe_bind(widget_name, signal_name, slot):
            widget = getattr(self, widget_name, None)
            if widget is not None and hasattr(widget, signal_name):
                signal = getattr(widget, signal_name)
                try:
                    signal.disconnect()
                except Exception:
                    pass
                signal.connect(slot)

        if hasattr(self, 'workshop_main_top_bar'):
            home_button = self.workshop_main_top_bar.findChild(QToolButton, "FixedHomePageButton")
            if home_button:
                try:
                    home_button.clicked.disconnect()
                except Exception:
                    pass
                home_button.clicked.connect(lambda: self.switch_page_with_slide(self.WELCOME_PAGE_INDEX))

        safe_bind('load_stitch_image_button', 'clicked', self.load_stitch_image_action)
        safe_bind('save_stitched_button', 'clicked', self.save_stitched_image)
        safe_bind('clear_stitching_canvas_button', 'clicked',
                  lambda: getattr(self, 'stitching_canvas', None) and self.stitching_canvas.clear_all())
        safe_bind('workshop_asset_library_button', 'toggled', self.toggle_asset_library)

        if hasattr(self, 'workshop_item_tools_button'):
            safe_bind('workshop_item_tools_button', 'clicked',
                      functools.partial(self._on_top_panel_button_clicked, self.workshop_item_tools_button,
                                        "workshop_item_tools"))
        if hasattr(self, 'workshop_canvas_settings_button'):
            safe_bind('workshop_canvas_settings_button', 'clicked',
                      functools.partial(self._on_top_panel_button_clicked, self.workshop_canvas_settings_button,
                                        "stitch_canvas_settings"))
        if hasattr(self, 'workshop_layers_button'):
            safe_bind('workshop_layers_button', 'clicked',
                      functools.partial(self._on_top_panel_button_clicked, self.workshop_layers_button,
                                        "stitch_layers"))
        if hasattr(self, 'workshop_enhance_button'):
            safe_bind('workshop_enhance_button', 'clicked',
                      functools.partial(self._on_top_panel_button_clicked, self.workshop_enhance_button,
                                        "stitch_enhance"))

        safe_bind('segment_selected_button', 'clicked', lambda: self._enter_segmentation_mode(
            item_to_segment=getattr(self, 'stitching_canvas', None).get_primary_selected_item() if hasattr(self,
                                                                                                           'stitching_canvas') else None))
        safe_bind('save_selected_item_button', 'clicked', self.save_selected_stitched_item)

        safe_bind('seg_sam_mode_tool', 'toggled', self._on_segmentation_mode_tool_toggled)
        safe_bind('seg_undo_tool', 'clicked',
                  lambda: getattr(self, 'segmentation_overlay_label').undo_last_action() if hasattr(self,
                                                                                                    'segmentation_overlay_label') else None)
        safe_bind('seg_redo_tool', 'clicked',
                  lambda: getattr(self, 'segmentation_overlay_label').redo_last_action() if hasattr(self,
                                                                                                    'segmentation_overlay_label') else None)
        safe_bind('seg_preview_tool', 'pressed', self.start_mask_preview)
        safe_bind('seg_preview_tool', 'released', self.stop_mask_preview)
        safe_bind('seg_reset_view_tool', 'clicked',
                  lambda: getattr(self, 'segmentation_overlay_label').reset_view() if hasattr(self,
                                                                                              'segmentation_overlay_label') else None)

        if hasattr(self, 'seg_settings_button'):
            safe_bind('seg_settings_button', 'clicked',
                      functools.partial(self._on_top_panel_button_clicked, self.seg_settings_button,
                                        "segment_overlay_settings"))
        if hasattr(self, 'seg_refine_button'):
            safe_bind('seg_refine_button', 'clicked',
                      functools.partial(self._on_top_panel_button_clicked, self.seg_refine_button,
                                        "segment_overlay_refine"))
        if hasattr(self, 'seg_color_button'):
            safe_bind('seg_color_button', 'clicked',
                      functools.partial(self._on_top_panel_button_clicked, self.seg_color_button,
                                        "segment_overlay_color"))

        safe_bind('save_segmentation_button', 'clicked', self.save_segmentation_from_overlay)
        safe_bind('cancel_segmentation_button', 'clicked', self._exit_segmentation_mode)
        safe_bind('apply_segmentation_button', 'clicked', self._apply_segmentation_and_exit_mode)

        if hasattr(self, 'stitching_canvas') and self.stitching_canvas:
            try:
                self.stitching_canvas.selection_changed.disconnect()
            except Exception:
                pass
            self.stitching_canvas.selection_changed.connect(self.on_stitch_selection_changed)

            try:
                self.stitching_canvas.layers_changed.disconnect()
            except Exception:
                pass
            self.stitching_canvas.layers_changed.connect(self.update_stitch_layers_list)

            try:
                self.stitching_canvas.canvas_resized.disconnect()
            except Exception:
                pass
            self.stitching_canvas.canvas_resized.connect(self.on_stitch_canvas_resized)

        if hasattr(self, 'segmentation_overlay_label') and self.segmentation_overlay_label:
            try:
                self.segmentation_overlay_label.predict_request.disconnect()
            except Exception:
                pass
            self.segmentation_overlay_label.predict_request.connect(self.start_image_prediction)

            try:
                self.segmentation_overlay_label.refinement_started.disconnect()
            except Exception:
                pass
            self.segmentation_overlay_label.refinement_started.connect(self.show_refinement_progress_overlay)

            try:
                self.segmentation_overlay_label.refinement_finished.disconnect()
            except Exception:
                pass
            self.segmentation_overlay_label.refinement_finished.connect(self.hide_refinement_progress_overlay)

        safe_bind('smooth_slider', 'valueChanged', lambda val: self._on_refinement_changed())
        safe_bind('feather_slider', 'valueChanged', lambda val: self._on_refinement_changed())
        safe_bind('shift_slider', 'valueChanged', lambda val: self._on_refinement_changed())
        safe_bind('matteformer_checkbox', 'toggled', lambda checked: self._on_refinement_changed())

    @Slot()
    def select_image_for_segmentation_in_overlay(self):
        reply = QMessageBox.question(
            self, '确认加载新图',
            '加载一张新图片将会丢失当前所有的抠图进度。\n您确定要继续吗？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return

        fpath, _ = QFileDialog.getOpenFileName(self, "选择一张新图片进行抠图", "",
                                               "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.jfif *.gif)")

        if fpath:
            self._load_image_for_segmentation(
                source_image_data=fpath,
                target_label=self.segmentation_overlay_label
            )

    @Slot()
    def clear_current_video_target(self):
        if getattr(self, 'current_target_id', -1) == -1 or self.current_target_id not in getattr(self, 'target_points',
                                                                                                 {}):
            QMessageBox.warning(self, _TR("无法删除"), _TR("未选中任何目标对象。"))
            return

        # 确保回退显示也是从 1 开始
        fallback_name = f"{_TR('对象')} {self.current_target_id + 1}"
        target_name = self.target_points[self.current_target_id].get('name', fallback_name)

        reply = QMessageBox.question(
            self, _TR("确认删除"),
            _TR("您确定要彻底删除【") + target_name + _TR("】及其所有视频抠图轨道吗？\n(可通过点击撤回恢复)"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._save_video_state()

            obj_to_delete = self.current_target_id
            del self.target_points[obj_to_delete]

            for frame_idx in list(getattr(self, 'processed_masks', {}).keys()):
                if obj_to_delete in self.processed_masks[frame_idx]:
                    del self.processed_masks[frame_idx][obj_to_delete]
                    if not self.processed_masks[frame_idx]:
                        del self.processed_masks[frame_idx]

            if getattr(self, 'temp_render_dir', None) and os.path.exists(self.temp_render_dir):
                try:
                    shutil.rmtree(self.temp_render_dir, ignore_errors=True)
                except Exception:
                    pass
            self.temp_render_dir = None
            self.video_segmentation_finished = False

            if hasattr(self, '_video_frame_cache'):
                self._video_frame_cache.clear()

            if hasattr(self, 'video_display_label') and hasattr(self.video_display_label, 'temp_multi_masks'):
                for f_idx in list(self.video_display_label.temp_multi_masks.keys()):
                    if obj_to_delete in self.video_display_label.temp_multi_masks[f_idx]:
                        del self.video_display_label.temp_multi_masks[f_idx][obj_to_delete]

            if hasattr(self.video_display_label, 'temp_annotation_target_id') and getattr(self.video_display_label,
                                                                                          'temp_annotation_target_id',
                                                                                          -1) == obj_to_delete:
                self.video_display_label.temp_annotation_frame_mask = None
                self.video_display_label.temp_annotation_target_id = -1
                self.video_display_label.temp_annotation_mask_frame_idx = -1

            self.current_target_id = -1
            self._sync_sam2_after_history_jump()

            if not getattr(self, 'target_points', {}):
                self.next_target_id = 0
                self.video_segmentation_finished = False
                self.video_segmentation_saved = False
            else:
                self._distribute_global_masks_to_clips()
                self._rebuild_matted_preview_cache()

    def _create_separator(self, shape=QFrame.Shape.HLine):
        line = QFrame()
        line.setFrameShape(shape)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        if shape == QFrame.Shape.HLine:
            line.setFixedHeight(1)
        else:
            line.setFixedWidth(1)
        return line

    @Slot(str, int)
    def show_status_message(self, message, timeout=3000):
        """Redirects status messages to console instead of displaying on the status bar."""
        print(f"[STATUS] {message}")

    @Slot(str)
    def _on_worker_error(self, error_message):
        sender_worker = self.sender()
        worker_id = next((k for k, v in self.active_workers.items() if v[1] == sender_worker), None)
        worker_id_cn = WORKER_ID_TO_CN.get(worker_id, worker_id) if worker_id else "未知任务"

        if worker_id:
            self.show_status_message(f"任务“{worker_id_cn}”错误: {error_message}", 5000)
            if not ("cancel" in error_message.lower() or "取消" in error_message.lower()):
                QMessageBox.critical(self, f"任务“{worker_id_cn}”错误", error_message)

            if worker_id == "load_image_model":
                self.is_loading_model = False
                self.sam_image_load_failed = True
            elif worker_id == "load_video_model":
                self.is_loading_model = False
                self.sam_video_load_failed = True
            elif worker_id == "enhance":
                self.is_enhancing = False
            elif worker_id == "predict":
                self.is_predicting = False
                self._current_predict_cache_key = None
                self._current_prediction_cumulative = False
            elif worker_id == "extract":
                self.is_extracting_frames = False
                if hasattr(self, 'video_display_label'):
                    self.video_display_label.setText(f"提取失败: {error_message[:100]}")
                self.reset_video_state()
            elif worker_id == "propagate_video_v1991":
                self.video_segmentation_running = False
                self.processed_masks = {}
                self.update_video_preview_all_targets(self.current_frame_index)
                if hasattr(self, 'video_display_label'):
                    if self.video_display_label.current_pixmap is not None:
                        self.video_display_label.current_pixmap = None
                        self.video_display_label.clear()
                    self.video_display_label.setText(f"视频抠图失败: {error_message[:100]}")
                if self.current_propagate_video_progress_dialog:
                    try:
                        self.current_propagate_video_progress_dialog.close()
                    except RuntimeError:
                        pass
                    self.current_propagate_video_progress_dialog = None
            elif worker_id.startswith("save_"):
                self.is_saving = False
            self.update_button_states()
        else:
            self.log_message.emit(f"收到来自未知工作线程的错误: {error_message}")
            QMessageBox.critical(self, "未知任务错误", error_message)
            self.is_loading_model = False
            self.is_enhancing = False
            self.is_predicting = False
            self.is_extracting_frames = False
            self.video_segmentation_running = False
            self.is_saving = False
            self.update_button_states()

    @Slot(int, str)
    def _on_worker_progress(self, percentage, message):
        """Receives and visualizes sub-progress values emitted by backend tasks."""
        if "重排" in message or "解码" in message or "合并" in message or "提取" in message:
            display_msg = f"正在无损重构视频片段...\n{message} ({percentage}%)"
            self.show_global_loading_overlay(display_msg, percentage)
        elif "发丝" in message or "解析" in message or "追踪" in message:
            display_msg = f"发丝级双引擎优化渲染中...\n{message} ({percentage}%)"
            self.show_global_loading_overlay(display_msg, percentage)
        elif "特征" in message or "流特征" in message:
            display_msg = f"正在提取视频特征深度编码...\n{message} ({percentage}%)"
            self.show_global_loading_overlay(display_msg, percentage)
        elif "编码输出" in message or "混音" in message or "音频" in message:
            display_msg = f"正在混合多轨音视频流并导出...\n{message} ({percentage}%)"
            self.show_global_loading_overlay(display_msg, percentage)
        else:
            self.show_global_loading_overlay(f"系统处理中，请稍候...\n{message} ({percentage}%)", percentage)

    @Slot(str)
    def _remove_active_worker(self, worker_id):
        worker_id_cn = WORKER_ID_TO_CN.get(worker_id, worker_id)
        if worker_id in self.active_workers:
            was_video_worker = worker_id in ["extract", "propagate_video_v1991", "save_video", "load_video_model",
                                             "init_video_state"]

            if worker_id.startswith("load_"):
                self.is_loading_model = False
            elif worker_id == "enhance":
                self.is_enhancing = False
            elif worker_id == "predict":
                self.is_predicting = False
            elif worker_id == "extract":
                self.is_extracting_frames = False
            elif worker_id == "propagate_video_v1991":
                self.video_segmentation_running = False
            elif worker_id.startswith("save_"):
                self.is_saving = False

            progress_dialog_name = f"current_{worker_id}_progress"
            if hasattr(self, progress_dialog_name):
                progress_dialog = getattr(self, progress_dialog_name, None)
                if progress_dialog and isinstance(progress_dialog, QProgressDialog):
                    try:
                        progress_dialog.close()
                        setattr(self, progress_dialog_name, None)
                    except RuntimeError:
                        pass

            if worker_id in ["extract", "propagate_video_v1991", "save_video", "init_video_state",
                             "video_sync_history"]:
                if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                    self._global_loading_overlay.hide()

            del self.active_workers[worker_id]
            self.log_message.emit(f"工作线程“{worker_id_cn}”已从活动列表安全移除。")

            is_still_busy_video = self.is_extracting_frames or self.video_segmentation_running or \
                                  (self.is_saving and any(k == "save_video" for k in self.active_workers)) or \
                                  (self.is_loading_model and any(k == "load_video_model" for k in self.active_workers))

            if was_video_worker and not is_still_busy_video:
                final_status = "未加载素材"
                if self.video_path:
                    if self.video_segmentation_finished and self.processed_masks:
                        final_status = f"视频: {os.path.basename(self.video_path)}\n状态: 抠图完成"
                    elif self.target_points:
                        final_status = f"视频: {os.path.basename(self.video_path)}\n状态: 准备抠图"
                    else:
                        final_status = f"视频: {os.path.basename(self.video_path)}\n状态: 已加载"
                if hasattr(self, 'video_info_label_display'):
                    self.video_info_label_display.setText(final_status)

            self.show_status_message(f"任务“{worker_id_cn}”已完成。", 3000)
            self.update_button_states()
        else:
            self.log_message.emit(f"警告: 尝试移除不存在的工作线程“{worker_id_cn}”。")

    def _update_interaction_specific_controls(self):
        if not hasattr(self, 'interaction_specific_group'):
            return
        img_label = getattr(self, 'segment_label', None)
        is_sam_mode = img_label.interaction_mode == 'sam' if img_label else True
        is_paint_mode = img_label.interaction_mode == 'paint' if img_label else False

        children_cards = self.interaction_specific_group.findChildren(QWidget, "CardWidget",
                                                                      Qt.FindChildOption.FindDirectChildrenOnly)
        sam_card = children_cards[0] if len(children_cards) > 0 else None
        paint_card = children_cards[1] if len(children_cards) > 1 else None

        if sam_card:
            sam_card.setVisible(is_sam_mode)
        if paint_card:
            paint_card.setVisible(is_paint_mode)

        self.interaction_specific_group.setVisible(is_sam_mode or is_paint_mode)

    def _generate_overlay_pixmap(self, refined_mask_param: np.ndarray) -> QPixmap:
        """Returns a composite color-masked alpha QPixmap representation of the refined mask."""
        if refined_mask_param is None:
            return QPixmap()

        h_mask, w_mask = refined_mask_param.shape[:2]

        if refined_mask_param.dtype == bool:
            alpha_mask_u8 = refined_mask_param.astype(np.uint8) * 255
        elif np.issubdtype(refined_mask_param.dtype, np.floating):
            alpha_mask_u8 = np.clip(refined_mask_param * 255., 0, 255).astype(np.uint8)
        elif refined_mask_param.dtype == np.uint8:
            alpha_mask_u8 = refined_mask_param
        else:
            return QPixmap()

        if np.any(alpha_mask_u8 > 0):
            overlay_cv = np.zeros((h_mask, w_mask, 4), dtype=np.uint8)

            if hasattr(self, 'selected_mask_color'):
                fill_color = self.selected_mask_color
            elif hasattr(self, 'parent_window') and hasattr(self.parent_window, 'selected_mask_color'):
                fill_color = self.parent_window.selected_mask_color
            else:
                fill_color = QColor(60, 120, 220)

            fill_bgr = (fill_color.blue(), fill_color.green(), fill_color.red())
            mask_bool_fill = alpha_mask_u8 > 2
            overlay_cv[mask_bool_fill, 0:3] = fill_bgr

            default_alpha = getattr(self, 'DEFAULT_MASK_ALPHA_IMAGE', 127)
            overlay_alpha_channel = (alpha_mask_u8.astype(np.float32) / 255. * default_alpha).astype(np.uint8)
            overlay_cv[:, :, 3] = overlay_alpha_channel

            return convert_cv_to_pixmap(overlay_cv)
        else:
            return QPixmap()

    def handle_prediction_result(self, new_mask_from_sam: np.ndarray, was_cumulative: bool):
        """Updates internal segmentation mask objects with results from the SAM model prediction."""
        if new_mask_from_sam is None or not self._ensure_mask_exists():
            self.parent_window.log_message.emit("ImageLabel: 收到空的蒙版或内部蒙版不存在，无法处理结果。")
            return

        if new_mask_from_sam.dtype != bool:
            new_mask_from_sam = new_mask_from_sam > 0

        if new_mask_from_sam.shape != self.current_mask.shape:
            self.parent_window.log_message.emit(
                f"ImageLabel: 蒙版尺寸不匹配，无法处理。期望 {self.current_mask.shape}, 收到 {new_mask_from_sam.shape}")
            return

        prev_mask_for_history = self.current_mask.copy()
        potential_new_mask = None
        if was_cumulative:
            potential_new_mask = np.logical_or(self.current_mask, new_mask_from_sam)
        else:
            potential_new_mask = new_mask_from_sam

        mask_was_changed = not np.array_equal(self.current_mask, potential_new_mask)

        if mask_was_changed:
            self._push_mask_history(prev_mask_for_history)
            self.current_mask = potential_new_mask
            self._clear_all_caches()
            self.parent_window.log_message.emit("ImageLabel: SAM2预测导致蒙版更新，缓存已清除。")
        else:
            self.parent_window.log_message.emit("ImageLabel: 新预测的蒙版与当前相同，未做更改。")

        self.update_display()

        if self.parent_window:
            QTimer.singleShot(0, self.parent_window.update_button_states)

        if mask_was_changed:
            self._check_and_trigger_refinement_after_mask_change()

    @Slot(QColor)
    def _handle_color_selected(self, color: QColor):
        if not color.isValid():
            return

        active_label = getattr(self, 'segmentation_overlay_label', None)
        if not active_label:
            return

        self.selected_mask_color = color
        self._update_color_preview_button_style()
        active_label._clear_all_caches()
        active_label.update_display()

        if hasattr(self, 'color_dialog_instance') and self.color_dialog_instance:
            self.color_dialog_instance.accept()

    @Slot(int)
    def _on_color_dialog_finished(self, result: int):
        self.log_message.emit(f"颜色对话框已关闭，结果代码: {result}")
        self._is_color_dialog_open = False
        if hasattr(self, 'color_dialog_instance'):
            self.color_dialog_instance.deleteLater()
            self.color_dialog_instance = None

    @Slot()
    def _on_color_button_clicked(self):
        sender = self.sender()
        if not sender:
            return

        active_label = None
        if self.is_in_segmentation_overlay_mode and hasattr(self, 'segmentation_overlay_label'):
            active_label = self.segmentation_overlay_label

        if not active_label or active_label._is_previewing_mask:
            return

        name = sender.property("color_name")
        if name and name in MASK_COLORS and name != self.selected_color_button_name:
            self.selected_mask_color = MASK_COLORS[name]
            self.selected_color_button_name = name
            self._update_color_button_styles()

            self.log_message.emit(f"蒙版颜色已更改为 '{name}'。正在通知 active_label 更新。")
            active_label._clear_all_caches()
            active_label.update_display()

    @Slot(int)
    def update_brush_size(self, value):
        seg_label = getattr(self, 'segmentation_overlay_label', None)
        if seg_label:
            seg_label.set_brush_size(value)
        else:
            self.log_message.emit("错误: update_brush_size 无法找到 segmentation_overlay_label 实例。")

        if hasattr(self, 'brush_size_label'):
            self.brush_size_label.setText(f"{value} px")

    def _on_refinement_changed(self, context=None):
        required_widgets = ['smooth_slider', 'feather_slider', 'shift_slider', 'matteformer_checkbox']
        if not all(hasattr(self, name) for name in required_widgets):
            return

        if not self.is_in_segmentation_overlay_mode:
            return

        active_label = getattr(self, 'segmentation_overlay_label', None)
        if not active_label or active_label._is_previewing_mask:
            return

        if not hasattr(self, '_pending_refinement_values') or self._pending_refinement_values is None:
            self._pending_refinement_values = {}

        pending_params = self._pending_refinement_values.copy()
        pending_params['refine_smooth'] = self.smooth_slider.value()
        pending_params['refine_feather'] = self.feather_slider.value()
        actual_shift_float = self.shift_slider.value() / SHIFT_SLIDER_FACTOR
        pending_params['refine_shift'] = int(round(actual_shift_float))
        pending_params['refine_matteformer_enabled'] = self.matteformer_checkbox.isChecked()

        if isinstance(context, dict):
            pending_params.update(context)

        self._pending_refinement_values = pending_params

        if hasattr(self, 'smooth_value_label'):
            self.smooth_value_label.setText(f"{pending_params['refine_smooth']}")
        if hasattr(self, 'feather_value_label'):
            self.feather_value_label.setText(f"{pending_params['refine_feather']}")
        if hasattr(self, 'shift_value_label'):
            self.shift_value_label.setText(f"{actual_shift_float:.1f} px")

        if not hasattr(self, '_refinement_update_timer') or self._refinement_update_timer is None:
            self._refinement_update_timer = QTimer(self)
            self._refinement_update_timer.setSingleShot(True)
            self._refinement_update_timer.setInterval(150)
            self._refinement_update_timer.timeout.connect(self._apply_pending_refinements)

        self._refinement_update_timer.start()

    def _apply_pending_refinements(self):
        """Processes debounced parameters to compute high-fidelity boundaries on another thread."""
        if getattr(self, '_is_in_preview_transition', False):
            return
        if not getattr(self, 'is_in_segmentation_overlay_mode', False):
            return

        active_label = getattr(self, 'segmentation_overlay_label', None)
        if not active_label or not getattr(self, '_pending_refinement_values', None):
            return
        if getattr(active_label, '_is_previewing_mask', False):
            return

        if not hasattr(self, '_refinement_threads_pool'):
            self._refinement_threads_pool = set()

        dead_threads = [t for t in self._refinement_threads_pool if not t.isRunning()]
        for t in dead_threads:
            self._refinement_threads_pool.remove(t)

        if self._refinement_threads_pool:
            self._refinement_update_timer.start()
            return

        params_to_apply = self._pending_refinement_values.copy()
        dirty_rect_from_context = params_to_apply.pop('dirty_rect', None)
        self._pending_dirty_rect = None

        param_changed = active_label.set_refinement_params_batch(params_to_apply, dirty_rect=dirty_rect_from_context)

        has_valid_mask = active_label.current_mask is not None and np.any(active_label.current_mask)
        mask_changed_trigger = (dirty_rect_from_context is not None) or (active_label._cached_refined_mask is None)

        if not has_valid_mask or (not param_changed and not mask_changed_trigger):
            return

        device_str = self.get_current_device().type
        mat_model_to_pass = None

        if params_to_apply.get('refine_matteformer_enabled', False):
            if not getattr(self, 'matteformer_loaded', False) or getattr(self, 'matteformer_model', None) is None:
                self.show_status_message("正在初始化发丝级大模型，请稍候...", 2000)
                try:
                    resolved_path = MATANYONE_CHECKPOINT_PATH
                    app_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.getcwd()
                    test_path = os.path.join(app_path, MATANYONE_CHECKPOINT_PATH)
                    if os.path.exists(test_path):
                        resolved_path = test_path

                    if os.path.exists(resolved_path):
                        try:
                            if GlobalHydra is not None:
                                GlobalHydra.instance().clear()
                        except Exception:
                            pass
                        device_obj = torch.device(device_str)
                        mat_model = get_matanyone2_model(resolved_path, device_str)
                        self.matteformer_model = mat_model.to(device_obj).eval()
                        self.matteformer_loaded = True
                except Exception as lazy_err:
                    print(f"MatAnyone2 lazy loading failed: {lazy_err}")

            mat_model_to_pass = getattr(self, 'matteformer_model', None)

        self.show_refinement_progress_overlay("正在进行精细优化...")

        if getattr(active_label, '_cached_refined_mask', None) is not None:
            mask_copy = np.array(active_label._cached_refined_mask, copy=True, order='C')
        else:
            mask_copy = np.array(active_label.current_mask, copy=True, order='C')

        img_copy = np.array(active_label.original_cv_image, copy=True, order='C')
        is_full_rebuild = (active_label._cached_refined_mask is None) or (dirty_rect_from_context is None)

        dirty_rect_info = None
        if not is_full_rebuild:
            padding = 64
            h, w = img_copy.shape[:2]
            compute_rect = dirty_rect_from_context.adjusted(-padding, -padding, padding, padding).toRect().intersected(
                QRect(0, 0, w, h))
            if not compute_rect.isEmpty():
                dirty_rect_info = compute_rect.getRect()
            else:
                is_full_rebuild = True

        thread = RefinementComputeThread(
            mask_copy, img_copy, params_to_apply, is_full_rebuild, dirty_rect_info, mat_model_to_pass, device_str
        )

        self._refinement_threads_pool.add(thread)
        thread.compute_finished.connect(self._handle_refinement_completion)
        thread.finished.connect(lambda t=thread: self._refinement_threads_pool.discard(t) if hasattr(self,
                                                                                                     '_refinement_threads_pool') else None)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    @Slot(object, bool, object, bool)
    def _handle_refinement_completion(self, result_mask, full_rebuild, dirty_info, success):
        self.hide_refinement_progress_overlay()

        active_label = getattr(self, 'segmentation_overlay_label', None)
        if success and result_mask is not None and active_label:
            safe_result_mask = np.array(result_mask, copy=True, order='C')

            if full_rebuild:
                active_label._cached_refined_mask = safe_result_mask
            else:
                if active_label._cached_refined_mask is not None:
                    x, y, cw, ch = dirty_info
                    blend_weight = np.ones((ch, cw), dtype=np.float32)
                    feather_width = min(6, cw // 6, ch // 6)
                    if feather_width > 0:
                        for i in range(feather_width):
                            val = float(i) / feather_width
                            blend_weight[i, :] = np.minimum(blend_weight[i, :], val)
                            blend_weight[-1 - i, :] = np.minimum(blend_weight[-1 - i, :], val)
                            blend_weight[:, i] = np.minimum(blend_weight[:, i], val)
                            blend_weight[:, -1 - i] = np.minimum(blend_weight[:, -1 - i], val)

                    old_local_tile = active_label._cached_refined_mask[y:y + ch, x:x + cw]
                    grafted_tile = safe_result_mask * blend_weight + old_local_tile * (1.0 - blend_weight)
                    active_label._cached_refined_mask[y:y + ch, x:x + cw] = grafted_tile
                else:
                    active_label._cached_refined_mask = safe_result_mask

            active_label._last_valid_refined_mask = active_label._cached_refined_mask.copy()
            active_label._cached_overlay_pixmap = None
            active_label._cached_cutout_pixmap = None
            active_label.update_display()

    def reset_mask_refinement_controls(self):
        """Reverts active parameters inside refinement sliders back to their defined standard configurations."""
        self.log_message.emit("Resetting mask refinement controls and syncing state...")

        if hasattr(self, '_refinement_update_timer') and self._refinement_update_timer.isActive():
            self._refinement_update_timer.stop()
            self._pending_refinement_values.clear()

        default_params = {
            'refine_smooth': DEFAULT_REFINE_SMOOTH,
            'refine_feather': DEFAULT_REFINE_FEATHER,
            'refine_shift': int(round(DEFAULT_REFINE_SHIFT_SLIDER / SHIFT_SLIDER_FACTOR)),
            'refine_matteformer_enabled': False
        }

        if hasattr(self, 'smooth_slider'):
            self.smooth_slider.setValue(default_params['refine_smooth'])
        if hasattr(self, 'feather_slider'):
            self.feather_slider.setValue(default_params['refine_feather'])
        if hasattr(self, 'shift_slider'):
            self.shift_slider.setValue(int(default_params['refine_shift'] * SHIFT_SLIDER_FACTOR))
        if hasattr(self, 'matteformer_checkbox'):
            self.matteformer_checkbox.setChecked(default_params['refine_matteformer_enabled'])

        self._on_refinement_changed()
        if hasattr(self, '_refinement_update_timer'):
            self._refinement_update_timer.stop()

        active_label = getattr(self, 'segmentation_overlay_label', None)
        if active_label:
            self.log_message.emit("Forcing sync of default states to ImageLabel...")
            active_label.set_refinement_params_batch(default_params.copy())

        self.log_message.emit("Mask refinement controls synchronized.")

    def _update_refinement_control_enable_state(self):
        """Deprecated stub function (removed)."""
        pass

    @Slot(bool)
    def on_paint_render_mode_changed(self, checked: bool):
        seg_label = getattr(self, 'segmentation_overlay_label', None)
        if seg_label:
            new_mode = 'live' if checked else 'precise'
            seg_label.set_paint_render_mode(new_mode)
            self.log_message.emit(f"Paint render mode set to: {new_mode}")
        else:
            self.log_message.emit("Error: on_paint_render_mode_changed unable to find segmentation_overlay_label.")
        self.update_button_states()

    @Slot()
    def confirm_and_clear_segmentation(self):
        img_label = getattr(self, 'segment_label', None)
        if not img_label or img_label.original_cv_image is None:
            return

        has_prompts = bool(img_label.points) or img_label.input_box is not None or img_label.painting
        mask_modified = len(img_label.mask_history) > 1
        settings_modified = False

        try:
            if hasattr(self, 'smooth_slider'):
                settings_modified = (
                        self.smooth_slider.value() != DEFAULT_REFINE_SMOOTH or
                        self.feather_slider.value() != DEFAULT_REFINE_FEATHER or
                        int(round(self.shift_slider.value() / SHIFT_SLIDER_FACTOR)) != DEFAULT_REFINE_SHIFT
                )
        except AttributeError:
            settings_modified = False
        except Exception:
            settings_modified = True

        if not has_prompts and not mask_modified and not settings_modified:
            QMessageBox.information(self, _TR("无需重置"), _TR("没有可重置的交互提示、蒙版修改或非默认优化设置。"))
            return

        reply = QMessageBox.question(
            self, _TR('确认重置抠图状态'),
            _TR("您确定要重置当前图像的抠图状态吗？\n\n"
                "这将：\n"
                "  • 清除所有交互提示 (点/框/绘制)。\n"
                "  • 将蒙版恢复到初始图像状态。\n"
                "  • 将所有“蒙版优化”设置恢复为默认值。\n"
                "  • 清除撤销/重做历史记录。\n\n"
                "此操作无法撤销啦！！！"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if img_label.reset_mask_and_history():
                self.show_status_message(_TR("图像分割状态和优化设置已重置。"), 3000)
            else:
                self.show_status_message(_TR("重置图像分割状态时出现问题。"), 4000)

            self.reset_mask_refinement_controls()
            QTimer.singleShot(0, self.update_button_states)
        else:
            self.show_status_message(_TR("重置操作已取消。"), 2000)

    def keyPressEvent(self, event: QKeyEvent):
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        super().keyReleaseEvent(event)

    @Slot()
    def start_mask_preview(self):
        if self.stacked_widget.currentIndex() == self.CREATIVE_WORKSHOP_INDEX and self.is_in_segmentation_overlay_mode:
            active_label = getattr(self, 'segmentation_overlay_label', None)
            if active_label and not active_label._is_previewing_mask:
                self.log_message.emit("Start mask preview.")
                active_label.set_mask_preview_mode(True)
                if hasattr(self, 'seg_preview_tool'):
                    self.seg_preview_tool.setDown(True)
                self.update_button_states()

    @Slot()
    def stop_mask_preview(self):
        if self.stacked_widget.currentIndex() == self.CREATIVE_WORKSHOP_INDEX and self.is_in_segmentation_overlay_mode:
            active_label = getattr(self, 'segmentation_overlay_label', None)
            if active_label and active_label._is_previewing_mask:
                self.log_message.emit("Stop mask preview.")
                active_label.set_mask_preview_mode(False)
                if hasattr(self, 'seg_preview_tool'):
                    self.seg_preview_tool.setDown(False)
                self.update_button_states()

    @Slot(str)
    def load_sam_predictor(self, predictor_type: str):
        if self.is_loading_model and f"load_{predictor_type}_model" in self.active_workers:
            self.show_status_message("模型已在加载中...", 3000)
            return

        model_path, cfg_path, loaded_flag_attr, fail_flag_attr, nav_button = "", "", "", "", None
        deps_ok = False
        p_type_cn = "图像" if predictor_type == "image" else "视频"

        if predictor_type == "image":
            deps_ok = SAM2_IMAGE_PREDICTOR_AVAILABLE and TORCH_AVAILABLE
            model_path, cfg_path = SAM2_IMAGE_CHECKPOINT_PATH, SAM2_IMAGE_MODEL_CFG_PATH
            loaded_flag_attr = 'image_predictor_loaded'
            fail_flag_attr = 'sam_image_load_failed'
            nav_button = getattr(self, 'nav_segment_button', None)
        elif predictor_type == "video":
            deps_ok = SAM2_VIDEO_PREDICTOR_AVAILABLE and TORCH_AVAILABLE
            model_path, cfg_path = SAM2_VIDEO_CHECKPOINT_PATH, SAM2_VIDEO_MODEL_CFG_PATH
            loaded_flag_attr = 'video_predictor_loaded'
            fail_flag_attr = 'sam_video_load_failed'
            nav_button = getattr(self, 'nav_video_seg_button', None)
        else:
            self.log_message.emit(f"Error: Unknown predictor type '{predictor_type}'.")
            return

        if not deps_ok:
            self.log_message.emit(f"SAM2 {p_type_cn} dependencies missing.")
            setattr(self, fail_flag_attr, True)
            if nav_button:
                original_text = nav_button.text().split('\n')[0]
                nav_button.setText(f"{original_text}\n(禁用)")
                nav_button.setEnabled(False)
            self.update_button_states()
            return

        resolved_model_path, resolved_cfg_path = model_path, cfg_path
        try:
            app_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.getcwd()
            test_model_path = os.path.join(app_path, model_path)
            test_cfg_path = os.path.join(app_path, cfg_path)

            if os.path.exists(test_model_path):
                resolved_model_path = test_model_path
            if os.path.exists(test_cfg_path):
                resolved_cfg_path = test_cfg_path

            if not os.path.exists(resolved_model_path):
                resolved_model_path = os.path.abspath(model_path)
            if not os.path.exists(resolved_cfg_path):
                resolved_cfg_path = os.path.abspath(cfg_path)
        except Exception:
            resolved_model_path = os.path.abspath(model_path)
            resolved_cfg_path = os.path.abspath(cfg_path)

        if not os.path.exists(resolved_model_path) or not os.path.exists(resolved_cfg_path):
            error_msg = f"SAM2 {p_type_cn} 模型或配置文件未找到:\n模型: {resolved_model_path}\n配置: {resolved_cfg_path}"
            self.log_message.emit(f"错误: {error_msg}")
            setattr(self, fail_flag_attr, True)
            if nav_button:
                original_text = nav_button.text().split('\n')[0]
                nav_button.setText(f"{original_text}\n(失败)")
            self.update_button_states()
            return

        self.is_loading_model = True
        setattr(self, fail_flag_attr, False)

        if nav_button:
            original_text = nav_button.text().split('\n')[0]
            nav_button.setText(f"{original_text}\n(加载中...)")

        self.show_status_message(f"后台加载 SAM2 {p_type_cn} 模型中...", 0)
        self.update_button_states()

        worker = ModelLoaderWorker()
        worker_id = f"load_{predictor_type}_model"
        self._start_worker(worker_id, worker, "run_load", predictor_type, resolved_model_path, resolved_cfg_path,
                           self.get_current_device())

    @Slot(object, bool, str)
    def _handle_load_image_model_completion(self, predictor_instance, success, error_message):
        nav_button = getattr(self, 'nav_segment_button', None)
        original_nav_text = nav_button.text().split('\n')[0] if nav_button else "抠图"

        if success and predictor_instance:
            self.image_predictor = predictor_instance
            self.image_predictor_loaded = True
            self.sam_image_load_failed = False
            if nav_button:
                nav_button.setText(original_nav_text)
            self.show_status_message("SAM2 图像分割模型加载成功。", 3000)

            if self.stacked_widget.currentIndex() == getattr(self, 'CREATIVE_WORKSHOP_INDEX', 2) and getattr(self,
                                                                                                             'is_in_segmentation_overlay_mode',
                                                                                                             False):
                active_label = getattr(self, 'segmentation_overlay_label', None)
                if getattr(self, 'segment_image_path',
                           None) and active_label and active_label.original_cv_image is not None and not self.image_set_in_predictor:
                    self._set_image_in_predictor_after_load(active_label)
        else:
            self.image_predictor = None
            self.image_predictor_loaded = False
            self.sam_image_load_failed = True
            log_msg = f"加载 SAM2 图像分割模型失败: {error_message}"
            self.log_message.emit(log_msg)
            if nav_button:
                nav_button.setText(f"{original_nav_text}\n(失败)")
            if self.isVisible() and not getattr(self, 'is_loading_model', False):
                QMessageBox.critical(self, "模型加载错误", log_msg)

        self.is_loading_model = False
        self.update_button_states()

        if getattr(self, '_pending_video_model_load', False):
            self._pending_video_model_load = False
            if getattr(self, 'SAM2_VIDEO_PREDICTOR_AVAILABLE', True):
                self.load_sam_predictor("video")

    @Slot(object, bool, str)
    def _handle_load_video_model_completion(self, predictor_instance, success, error_message):
        nav_button = getattr(self, 'nav_video_seg_button', None)
        original_nav_text = nav_button.text().split('\n')[0] if nav_button else "视频"

        if success and predictor_instance:
            self.video_predictor = predictor_instance
            try:
                if hasattr(self.video_predictor, 'model'):
                    self.video_predictor.model.to(dtype=torch.float32)
                    self.video_predictor.model.to(self.get_current_device())
            except Exception as e:
                self.log_message.emit(f"Warning: Issue converting video model: {e}")

            self.video_predictor_loaded = True
            self.sam_video_load_failed = False
            if nav_button:
                nav_button.setText(original_nav_text)
            self.show_status_message("SAM2 视频抠图模型加载成功。", 3000)

            target_dir = getattr(self, 'clip_sandbox_dir', getattr(self, 'temp_frame_dir', None))
            if getattr(self, 'video_path', None) and target_dir:
                if getattr(self, 'video_inference_state', None) is None:
                    self._initialize_video_predictor_state()
        else:
            self.video_predictor = None
            self.video_predictor_loaded = False
            self.sam_video_load_failed = True
            log_msg = f"加载 SAM2 视频抠图模型失败: {error_message}"
            self.log_message.emit(log_msg)
            if nav_button:
                nav_button.setText(f"{original_nav_text}\n(失败)")
            if self.isVisible() and not getattr(self, 'is_loading_model', False):
                QMessageBox.critical(self, "模型加载错误", log_msg)

        self.is_loading_model = False
        self.update_button_states()

    @Slot()
    def select_image_for_enhance(self):
        if not REALESRGAN_AVAILABLE:
            QMessageBox.warning(self, "不可用", "增强功能需要 realesrgan/basicsr。")
            return
        if self.is_enhancing or "enhance" in self.active_workers or self.is_loading_model or self.is_saving:
            QMessageBox.information(self, "忙碌", "其他操作正在运行中。")
            return

        fpath, _ = QFileDialog.getOpenFileName(
            self, "选择图像 (增强)", "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.jfif)"
        )
        if fpath:
            self._load_image_for_enhance(fpath)

    def _load_image_for_enhance(self, file_path):
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "加载错误", f"文件未找到: {file_path}")
            return

        try:
            pixmap_for_display = QPixmap(file_path)
            if pixmap_for_display.isNull():
                cv_img_temp = imread_unicode(file_path, cv2.IMREAD_UNCHANGED)
                if cv_img_temp is None:
                    raise ValueError(f"使用QPixmap和OpenCV均加载图像失败: {file_path}")
                pixmap_for_display = convert_cv_to_pixmap(cv_img_temp)
                if pixmap_for_display.isNull():
                    raise ValueError(f"转换OpenCV图像为QPixmap失败: {file_path}")

            self.original_image_path_enhance = file_path
            self.enhanced_image_cv = None

            if hasattr(self, 'image_compare_widget'):
                self.image_compare_widget.set_images(original=pixmap_for_display, enhanced=None)

            self.show_status_message(f"已加载用于增强的图像: {os.path.basename(file_path)}", 3000)

            if hasattr(self, 'tile_mode_combo'):
                self._on_tile_mode_changed(self.tile_mode_combo.currentIndex())

        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "加载错误", f"加载图像时出错: {str(e)}")
            if hasattr(self, 'image_compare_widget'):
                self.image_compare_widget.clear_content()
            self.original_image_path_enhance = None
        finally:
            self.update_button_states()

    @Slot()
    def enhance_image_action(self):
        if not getattr(self, 'original_image_path_enhance', None):
            QMessageBox.warning(self, _TR("提示"), _TR("请选择要增强的图像。"))
            return
        if not REALESRGAN_AVAILABLE:
            QMessageBox.critical(self, _TR("错误"), _TR("增强模块不可用。"))
            return

        if hasattr(self, 'enhance_scale_combo'):
            selected_scale_str = self.enhance_scale_combo.currentText()
            try:
                self.last_enhancement_scale = int(selected_scale_str.replace('x', ''))
                if self.last_enhancement_scale < 1:
                    self.last_enhancement_scale = 1
                    self.enhance_scale_combo.setCurrentText(f"{self.last_enhancement_scale}x")
            except (ValueError, AttributeError):
                self.last_enhancement_scale = 4
        else:
            self.last_enhancement_scale = 4

        tile_value = 0
        tile_pad_value = 10

        if hasattr(self, 'tile_mode_combo'):
            selected_tile_mode_text = self.tile_mode_combo.currentText().strip()
            if selected_tile_mode_text in ["自动 (推荐)", "Auto (Rec)", "Auto"]:
                compare_widget = getattr(self, 'image_compare_widget_enhance',
                                         getattr(self, 'image_compare_widget', None))
                if self.original_image_path_enhance and compare_widget and getattr(compare_widget, 'original_pixmap',
                                                                                   None):
                    try:
                        pixmap_for_size = compare_widget.original_pixmap
                        longest_side = max(pixmap_for_size.width(), pixmap_for_size.height())
                        tile_value = 128 if longest_side > 2000 else (256 if longest_side > 1080 else 512)
                    except Exception:
                        tile_value = 0
            elif selected_tile_mode_text in ["自定义", "Custom"]:
                tile_value = getattr(self, 'custom_tile_size_spinbox', None).value() if hasattr(self,
                                                                                                'custom_tile_size_spinbox') else 0
            elif selected_tile_mode_text in ["大图模式", "Large Image"]:
                tile_value = 128
            elif selected_tile_mode_text in ["中图模式", "Medium Image"]:
                tile_value = 256
            elif selected_tile_mode_text in ["小图模式", "Small Image"]:
                tile_value = 512

        model_filename = ENHANCE_MODELS.get(DEFAULT_ENHANCE_MODEL_NAME)
        if hasattr(self, 'enhance_model_combo'):
            selected_model_display_name = self.enhance_model_combo.currentText()
            if "动漫" in selected_model_display_name or "Anime" in selected_model_display_name:
                model_filename = ENHANCE_MODELS.get("动漫")
            elif "通用" in selected_model_display_name or "General" in selected_model_display_name:
                model_filename = ENHANCE_MODELS.get("通用")

        resolved_enhance_model_path = model_filename
        try:
            app_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.getcwd()
            test_path = os.path.join(app_path, model_filename)
            if os.path.exists(test_path):
                resolved_enhance_model_path = test_path
        except Exception:
            resolved_enhance_model_path = os.path.abspath(model_filename)

        if getattr(self, 'is_enhancing', False) or "enhance" in getattr(self, 'active_workers', {}):
            QMessageBox.information(self, _TR("忙碌"), _TR("其他 AI 任务正在运行中，请稍后再试。"))
            return

        self.is_enhancing = True
        self.enhanced_image_cv = None

        from config.settings import get_app_lang
        status_msg = f"Enhancing ({self.last_enhancement_scale}x)..." if get_app_lang() == "en" else f"正在进行 AI 超清放大 ({self.last_enhancement_scale}x)..."

        # 呼出遮罩（动画会自动流转，无需手动干预循环）
        self.show_global_loading_overlay(status_msg, 0)
        self.update_button_states()

        # 彻底移除危险的 _enhance_heartbeat_timer 定时器
        worker = EnhanceWorker()
        self._start_worker(
            "enhance", worker, "run_enhance",
            self.original_image_path_enhance,
            resolved_enhance_model_path,
            ENHANCE_FIXED_DENOISE,
            self.last_enhancement_scale,
            self.get_current_device(),
            tile_value,
            tile_pad_value
        )

    @Slot(object, bool, str)
    def _handle_enhance_completion(self, result_cv_image, success, error_message):
        """处理全局增强模块回调：延迟转换超大图像防止UI白屏未响应"""
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        if hasattr(self, 'current_enhance_progress') and getattr(self, 'current_enhance_progress'):
            try:
                self.current_enhance_progress.close()
            except Exception:
                pass

        if success and result_cv_image is not None:
            self.show_global_loading_overlay(_TR("AI增强完毕，正在生成超高分辨率渲染图层..."), 0)

            def _do_convert_and_display():
                try:
                    self.enhanced_image_cv = result_cv_image
                    pixmap = convert_cv_to_pixmap(self.enhanced_image_cv)

                    if pixmap and not pixmap.isNull():
                        if hasattr(self, 'image_compare_widget'):
                            original_pixmap = self.image_compare_widget.original_pixmap
                            if original_pixmap:
                                self.image_compare_widget.set_images(original=original_pixmap, enhanced=pixmap)
                        self.show_status_message(_TR("增强完成。"), 3000)
                    else:
                        QMessageBox.warning(self, _TR("显示错误"), _TR("转换增强结果以便显示时失败。"))
                finally:
                    if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                        self._global_loading_overlay.hide()
                    self.is_enhancing = False
                    self.update_button_states()

            QTimer.singleShot(100, _do_convert_and_display)

        elif not success and error_message and not ("cancel" in error_message.lower() or "用户取消" in error_message):
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            QMessageBox.warning(self, _TR("增强失败"), f"{_TR('增强失败:')}\n{error_message}")
            self.is_enhancing = False
            self.update_button_states()
        else:
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            self.is_enhancing = False
            self.update_button_states()

        if hasattr(self, 'image_compare_widget') and not self.image_compare_widget.original_pixmap:
            self.image_compare_widget.clear_content()

    @Slot(object, bool, str)
    def _handle_enhance_for_canvas_item_completion(self, result_cv_image, success, error_message):
        """画布素材增强回调：处理 4K 巨图拼接防止卡顿假死"""
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        if hasattr(self, 'current_enhance_canvas_item_progress') and getattr(self,
                                                                             'current_enhance_canvas_item_progress'):
            try:
                self.current_enhance_canvas_item_progress.close()
            except Exception:
                pass

        if success and result_cv_image is not None and self.item_being_enhanced:
            item = self.item_being_enhanced
            self.show_global_loading_overlay(_TR("正在将超清材质投射到物理画布..."), 0)

            def _do_update_canvas():
                try:
                    enhanced_pixmap = convert_cv_to_pixmap(result_cv_image)
                    if enhanced_pixmap.isNull():
                        QMessageBox.warning(self, _TR("增强失败"), _TR("无法将增强结果转换为有效图像。"))
                    else:
                        if hasattr(self,
                                   'image_compare_widget_enhance') and self.image_compare_widget_enhance.original_pixmap:
                            self.image_compare_widget_enhance.set_images(
                                original=self.image_compare_widget_enhance.original_pixmap,
                                enhanced=enhanced_pixmap
                            )

                        item.pixmap = enhanced_pixmap
                        new_raw_size = QSizeF(item.pixmap.size())
                        canvas_w = self.stitching_canvas.canvas_size.width()
                        canvas_h = self.stitching_canvas.canvas_size.height()
                        scale_factor = 1.0

                        if new_raw_size.width() > canvas_w or new_raw_size.height() > canvas_h:
                            scale_w = canvas_w / new_raw_size.width() if new_raw_size.width() > 0 else 1
                            scale_h = canvas_h / new_raw_size.height() if new_raw_size.height() > 0 else 1
                            scale_factor = min(scale_w, scale_h) * 0.95

                        old_center = item.get_transformed_bounding_rect().center()
                        item.size = new_raw_size * scale_factor
                        item.pos = old_center - QPointF(item.size.width() / 2, item.size.height() / 2)
                        item.rotation = 0
                        item.is_enhanced = True

                        self.stitching_canvas.update()
                        if hasattr(self, 'stitching_canvas'):
                            self.stitching_canvas.selection_changed.emit()
                        self.show_status_message(f"素材 '{item.name}' 高清增强完成。", 3000)
                finally:
                    if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                        self._global_loading_overlay.hide()
                    self.is_enhancing = False
                    self.item_being_enhanced = None
                    self.update_button_states()

            QTimer.singleShot(100, _do_update_canvas)

        else:
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            if not success:
                QMessageBox.warning(self, _TR("增强失败"), f"{_TR('增强素材失败:')} {error_message}")
                if hasattr(self, 'image_compare_widget_enhance'):
                    self.image_compare_widget_enhance.clear_content()

            self.is_enhancing = False
            self.item_being_enhanced = None
            self.update_button_states()

    @Slot()
    def save_enhanced_result(self):
        if getattr(self, 'enhanced_image_cv', None) is None:
            QMessageBox.warning(self, "信息", "没有可保存的增强结果。")
            return

        if self.is_saving or "save_enhanced" in self.active_workers:
            QMessageBox.information(self, "忙碌", "其他保存操作正在进行中。")
            return

        base = "enhanced_output"
        if getattr(self, 'original_image_path_enhance', None):
            base = os.path.splitext(os.path.basename(self.original_image_path_enhance))[0]

        has_alpha = len(self.enhanced_image_cv.shape) == 3 and self.enhanced_image_cv.shape[2] == 4
        ext = '.png' if has_alpha else '.jpg'

        scale_suffix = f"_adjusted_1x" if self.last_enhancement_scale == 1 else f"_enhanced_{self.last_enhancement_scale}x"
        default_name = f"{base}{scale_suffix}{ext}"

        filters = "PNG (*.png);;JPEG (*.jpg *.jpeg);;所有文件 (*)" if has_alpha else "JPEG (*.jpg *.jpeg);;PNG (*.png);;所有文件 (*)"
        save_path, sel_filter = QFileDialog.getSaveFileName(self, "保存增强图像", default_name, filters)

        if save_path:
            _, cur_ext = os.path.splitext(save_path)
            if not cur_ext:
                if 'png' in sel_filter.lower():
                    save_path += '.png'
                elif 'jp' in sel_filter.lower():
                    save_path += '.jpg'
                else:
                    save_path += ext

            self.is_saving = True
            self.update_button_states()
            self.show_status_message(f"保存增强图像到 {os.path.basename(save_path)}...", 0)

            worker = SaveWorker()
            worker_id = "save_enhanced"
            self._start_worker(worker_id, worker, "run_save", "enhanced_image", self.enhanced_image_cv, save_path)

    @Slot(bool, str, str)
    def _handle_save_enhanced_completion(self, success, saved_path, error_message):
        self.is_saving = False
        self.update_button_states()
        if success:
            QMessageBox.information(self, "保存成功", f"增强图像已保存:\n{saved_path}")
        else:
            QMessageBox.critical(self, "保存失败", f"保存增强图像时出错:\n{error_message}")

    @Slot()
    def select_image_for_segment(self):
        if not (SAM2_IMAGE_PREDICTOR_AVAILABLE and TORCH_AVAILABLE):
            QMessageBox.warning(self, "不可用", "图像分割需要 sam2 和 PyTorch。")
            return

        if self.is_loading_model or self.is_predicting or "predict" in self.active_workers or self.is_saving:
            QMessageBox.information(self, "忙碌", "其他操作正在运行中。")
            return

        if not self.image_predictor_loaded:
            QMessageBox.warning(self, "预测器未就绪", "SAM2 图像预测器未加载。")
            return

        if self.sam_image_load_failed:
            QMessageBox.critical(self, "错误", "SAM2 图像预测器先前加载失败。")
            return

        img_label = getattr(self, 'segment_label', None)
        if img_label and img_label._is_previewing_mask:
            QMessageBox.information(self, "需要操作", "请松开“预览抠图”按钮。")
            return

        if img_label and img_label.original_cv_image is not None and len(img_label.mask_history) > 1:
            reply = QMessageBox.question(
                self, '加载新图像 - 未保存更改',
                '当前图像分割有未保存的更改。加载新图像将丢弃这些更改。\n您确定要继续吗？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return

        fpath, _ = QFileDialog.getOpenFileName(
            self, "选择图像 (分割)", "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.jfif *.gif)"
        )
        if fpath:
            self._load_image_for_segmentation(fpath)

    def _resize_mask_advanced(self, mask_to_resize: np.ndarray, guide_image_target_res: np.ndarray,
                              target_size: tuple[int, int]) -> np.ndarray:
        self.log_message.emit("Executing advanced mask resize with Guided Filter...")
        try:
            if not CV2_CONTRIB_AVAILABLE:
                raise RuntimeError("OpenCV Contrib not available.")

            mask_float_src = mask_to_resize.astype(np.float32)
            initial_resized_mask_float = cv2.resize(mask_float_src, target_size, interpolation=cv2.INTER_LANCZOS4)

            guide_image_gray = None
            if len(guide_image_target_res.shape) == 3:
                if guide_image_target_res.shape[2] == 4:
                    guide_image_gray = cv2.cvtColor(guide_image_target_res, cv2.COLOR_BGRA2GRAY)
                else:
                    guide_image_gray = cv2.cvtColor(guide_image_target_res, cv2.COLOR_BGR2GRAY)
            else:
                guide_image_gray = guide_image_target_res

            guide_image_float = guide_image_gray.astype(np.float32) / 255.0

            radius = 5
            eps = 0.01 ** 2
            refined_mask_float = cv2.ximgproc.guidedFilter(
                guide=guide_image_float,
                src=initial_resized_mask_float,
                radius=radius,
                eps=eps
            )

            refined_mask_float = np.clip(refined_mask_float, 0, 1)
            final_mask_bool = (refined_mask_float > 0.5)

            return final_mask_bool

        except Exception as e:
            self.log_message.emit(f"Warning: Advanced resize failed ({e}). Falling back to INTER_NEAREST.")
            mask_u8 = mask_to_resize.astype(np.uint8)
            resized_mask_u8 = cv2.resize(mask_u8, target_size, interpolation=cv2.INTER_NEAREST)
            return resized_mask_u8.astype(bool)

    def _load_image_for_segmentation(self, source_image_data, existing_mask_to_scale: Optional[np.ndarray] = None,
                                     is_resolution_change: bool = False, target_label: Optional['ImageLabel'] = None):
        """Prepares target matting dimensions and feeds pixel data into sandbox layouts."""
        if not target_label:
            target_label = getattr(self, 'segmentation_overlay_label', None)
            if not target_label:
                return

        if not is_resolution_change:
            target_label.clear_all()
            self.reset_mask_refinement_controls()

        self.image_set_in_predictor = False
        self.sam_prediction_cache.clear()

        original_cv_image_full_res = None
        if isinstance(source_image_data, str) and os.path.exists(source_image_data):
            original_cv_image_full_res = imread_unicode(source_image_data, cv2.IMREAD_UNCHANGED)
            self.segment_image_path = source_image_data
        elif isinstance(source_image_data, np.ndarray):
            original_cv_image_full_res = source_image_data.copy()
            self.current_seg_source_image_cv = original_cv_image_full_res

        if original_cv_image_full_res is None:
            return

        current_max_dim = -1
        device_type = self.get_current_device().type

        if device_type == "cpu":
            current_max_dim = 512
            self.segmentation_working_resolution_mode = "512px"
            self._update_work_resolution_controls_state()
        else:
            if getattr(self, 'segmentation_working_resolution_mode', '') == "512px":
                current_max_dim = 512
            elif getattr(self, 'segmentation_working_resolution_mode', '') == "768px":
                current_max_dim = 768
            elif getattr(self, 'segmentation_working_resolution_mode', '') == "1280px":
                current_max_dim = 1280
            elif getattr(self, 'segmentation_working_resolution_mode', '') == "1920px":
                current_max_dim = 1920
            elif getattr(self, 'segmentation_working_resolution_mode', '') == "custom":
                current_max_dim = getattr(self, 'segmentation_custom_max_dim', 1280)

        if current_max_dim > 0:
            working_cv_image, scale_factor = resize_image_to_max_dim(original_cv_image_full_res, current_max_dim)
        else:
            working_cv_image, scale_factor = original_cv_image_full_res.copy(), 1.0

        work_h, work_w = working_cv_image.shape[:2]
        initial_mask_data = None

        if existing_mask_to_scale is not None and np.any(existing_mask_to_scale):
            mask_h, mask_w = existing_mask_to_scale.shape[:2]
            if mask_h != work_h or mask_w != work_w:
                if np.issubdtype(existing_mask_to_scale.dtype, np.floating):
                    resized_mask = cv2.resize(existing_mask_to_scale, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
                    initial_mask_data = np.clip(resized_mask, 0.0, 1.0)
                else:
                    resized_mask = cv2.resize(existing_mask_to_scale.astype(np.uint8), (work_w, work_h),
                                              interpolation=cv2.INTER_NEAREST)
                    initial_mask_data = (resized_mask > 127)
            else:
                initial_mask_data = existing_mask_to_scale.copy()

        bool_mask = (initial_mask_data > 0.5) if initial_mask_data is not None and np.issubdtype(
            initial_mask_data.dtype, np.floating) else initial_mask_data
        target_label.set_image(
            working_cv_image,
            initial_mask=bool_mask,
            original_full_res_image=original_cv_image_full_res,
            scale_factor=scale_factor
        )

        if initial_mask_data is not None and np.issubdtype(initial_mask_data.dtype, np.floating):
            target_label._has_auto_matted = True
            target_label._clear_all_caches()
            target_label.current_mask = (initial_mask_data > 0.5)
            target_label._cached_refined_mask = initial_mask_data.copy()
            target_label._last_valid_refined_mask = initial_mask_data.copy()
            target_label.mask_history.clear()
            target_label.mask_history.append((target_label.current_mask.copy(), initial_mask_data.copy()))
            target_label.update_display()

        self._set_image_in_predictor_after_load(target_label)

    def remove_from_recent_projects(self, file_path_to_delete: str):
        if not file_path_to_delete:
            return

        recent_files = self.settings.value("recent_files", [], type=list)
        norm_path_to_delete = os.path.normpath(file_path_to_delete)

        original_len = len(recent_files)
        recent_files = [p for p in recent_files if os.path.normpath(p) != norm_path_to_delete]

        if len(recent_files) < original_len:
            self.settings.setValue("recent_files", recent_files)
            self.settings.sync()
            self.log_message.emit(f"已从最近项目中移除: {file_path_to_delete}")
            self.populate_recent_projects()

    def _update_work_resolution_controls_state(self):
        """Synchronizes checked status parameters inside working resolution widgets."""
        is_custom = getattr(self, 'segmentation_working_resolution_mode', '') == "custom"
        if hasattr(self, 'custom_max_dim_spinbox'):
            self.custom_max_dim_spinbox.setEnabled(is_custom)

        if hasattr(self, 'res_original_radio'):
            self.res_original_radio.setChecked(self.segmentation_working_resolution_mode == "original")
            if hasattr(self, 'res_512_radio'):
                self.res_512_radio.setChecked(self.segmentation_working_resolution_mode == "512px")
            if hasattr(self, 'res_768_radio'):
                self.res_768_radio.setChecked(self.segmentation_working_resolution_mode == "768px")
            if hasattr(self, 'res_1280_radio'):
                self.res_1280_radio.setChecked(self.segmentation_working_resolution_mode == "1280px")
            if hasattr(self, 'res_1920_radio'):
                self.res_1920_radio.setChecked(self.segmentation_working_resolution_mode == "1920px")
            if hasattr(self, 'res_custom_radio'):
                self.res_custom_radio.setChecked(self.segmentation_working_resolution_mode == "custom")

    def _set_image_in_predictor_after_load(self, target_label: ImageLabel):
        """Runs background feature extraction routines for the newly initialized image element."""
        if not self.image_predictor_loaded or not target_label or target_label.original_cv_image is None:
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            return

        if self.image_set_in_predictor:
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            return

        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.setGeometry(self.rect())
            self._global_loading_overlay.show()
            self._global_loading_overlay.raise_()

        self.show_status_message("正在提取并编码图像深度特征...", 0)

        try:
            cv_img = target_label.original_cv_image
            channels = cv_img.shape[2] if len(cv_img.shape) == 3 else 1
            img_rgb = None

            if channels == 1:
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
            elif channels == 4:
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
            elif channels == 3:
                img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"不支持的通道数: {channels}")

            if not img_rgb.flags['C_CONTIGUOUS']:
                img_rgb = np.ascontiguousarray(img_rgb)
            if img_rgb.dtype != np.uint8:
                img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

            thread = QThread(self)
            worker = SetImageWorker(self.image_predictor, img_rgb)
            worker.moveToThread(thread)

            def on_set_image_complete(success, error_msg):
                thread.quit()
                worker.deleteLater()

                if success:
                    self.image_set_in_predictor = True
                    self.sam_prediction_cache.clear()
                    self.show_status_message("图像特征深度编码完成，抠图功能已解锁！", 3000)
                else:
                    self.image_set_in_predictor = False
                    self.sam_prediction_cache.clear()
                    QMessageBox.critical(self, "特征提取失败", f"大模型提取特征失败: {error_msg}")
                    self.show_status_message("在特征提取中出错。", 5000)

                if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                    self._global_loading_overlay.hide()

                if hasattr(self, 'segmentation_overlay_label'):
                    self.segmentation_overlay_label.setFocus()

                self.update_button_states()

            worker.finished.connect(on_set_image_complete)
            thread.started.connect(worker.run)

            self._set_image_thread_holder = thread
            self._set_image_worker_holder = worker

            thread.start()

        except Exception as e:
            self.image_set_in_predictor = False
            self.sam_prediction_cache.clear()
            QMessageBox.critical(self, "特征编码错误", f"准备特征提取数据失败: {e}")
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            self.update_button_states()

    @Slot(object, tuple, object, bool, object)
    def start_image_prediction(self, predictor, point_data, box_data, cumulative, image_rgb=None):
        if self.is_predicting or "predict" in self.active_workers:
            self.show_status_message("预测已在进行中。", 3000)
            return

        if not self.image_set_in_predictor:
            QMessageBox.warning(self, "预测器错误", "图像未在预测器中设置。")
            return

        cache_key = None
        input_pts, input_lbls = point_data
        try:
            if box_data == "AUTO":
                cache_key = None
            elif box_data is not None:
                cache_key = ('box', tuple(np.array(box_data).flatten()), cumulative)
            elif input_pts is not None and input_lbls is not None:
                pts_list = sorted(zip(map(tuple, input_pts.tolist()), input_lbls.tolist()))
                cache_key = ('points', tuple(pts_list), cumulative)
        except Exception:
            cache_key = None

        if cache_key is not None and cache_key in self.sam_prediction_cache:
            cached_mask = self.sam_prediction_cache[cache_key]
            active_label = getattr(self, 'segmentation_overlay_label', getattr(self, 'segment_label', None))
            if active_label:
                active_label.handle_prediction_result(cached_mask, cumulative)
            self.show_status_message("分割完成 (来自缓存)。", 2000)
            self.update_button_states()
            return

        self.is_predicting = True
        self._current_predict_cache_key = cache_key
        self._current_prediction_cumulative = cumulative

        self.update_button_states()
        self.show_status_message("正在调用模型生成分割...", 0)

        worker = PredictWorker()
        worker_id = "predict"
        self._start_worker(worker_id, worker, "run_predict", predictor, point_data, box_data, cumulative, image_rgb)

    @Slot(object, bool, str)
    def _handle_predict_completion(self, result_mask, success, error_message):
        print("\n[DEBUG SAM2] P1. Entering _handle_predict_completion")
        sys.stdout.flush()

        self.hide_refinement_progress_overlay()

        cache_key = self._current_predict_cache_key
        active_label = getattr(self, 'segmentation_overlay_label', getattr(self, 'segment_label', None))

        if not active_label:
            print("[DEBUG SAM2] ERROR: active_label does not exist!")
            self.is_predicting = False
            self.update_button_states()
            return

        is_auto_predict = (cache_key is not None and len(cache_key) >= 2 and cache_key[0] == 'auto_everything')

        if success and result_mask is not None:
            print("[DEBUG SAM2] P2. Prediction succeeded, preparing dispatch.")
            sys.stdout.flush()

            def safe_copy_result(mask_data):
                if isinstance(mask_data, tuple):
                    return tuple(
                        np.array(arr, copy=True, order='C') if isinstance(arr, np.ndarray) else arr
                        for arr in mask_data
                    )
                elif isinstance(mask_data, np.ndarray):
                    return np.array(mask_data, copy=True, order='C')
                return mask_data

            safe_mask = safe_copy_result(result_mask)

            if cache_key is not None:
                self.sam_prediction_cache[cache_key] = safe_copy_result(safe_mask)

            active_label.handle_prediction_result(safe_mask, self._current_prediction_cumulative)

            if is_auto_predict:
                self.show_status_message("一键精细抠图处理完成。", 3000)
            else:
                self.show_status_message("分割完成。", 3000)

            print("[DEBUG SAM2] P3. handle_prediction_result processing finalized.")
            sys.stdout.flush()

        elif not success and error_message:
            self.show_status_message(f"发生分割错误: {error_message}", 5000)
        else:
            self.show_status_message("分割失败或已取消。", 3000)

        self._current_predict_cache_key = None
        self._current_prediction_cumulative = False
        self.is_predicting = False
        print("[DEBUG SAM2] P4. Invoking update_button_states.")
        sys.stdout.flush()
        self.update_button_states()
        print("[DEBUG SAM2] P5. _handle_predict_completion completed.")
        sys.stdout.flush()

    @Slot()
    def save_segmented_result(self):
        interactive_label = getattr(self, 'segment_label', None)
        if not interactive_label or interactive_label.current_mask is None or getattr(self,
                                                                                      'segment_original_cv_image_full_res',
                                                                                      None) is None:
            QMessageBox.warning(self, "保存错误", "没有有效的图像分割结果可保存。")
            return

        if self.is_saving or "save_segment" in self.active_workers:
            QMessageBox.information(self, "忙碌", "其他保存操作正在进行中。")
            return

        working_res_mask = interactive_label.current_mask
        original_cv_image_full_res = self.segment_original_cv_image_full_res
        scale_factor = getattr(self, 'segment_original_scale_factor', 1.0)
        refine_params = interactive_label.get_current_refinement_params_as_dict()

        is_empty = not np.any(working_res_mask)
        if is_empty:
            reply = QMessageBox.question(
                self, "保存确认", "最终蒙版为空。是否保存完全透明的图像？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        base_name = "segmented_output"
        ext = ".png"
        if getattr(self, 'segment_image_path', None):
            base_name = os.path.splitext(os.path.basename(self.segment_image_path))[0]

        default_name = f"{base_name}_segmented{ext}"
        save_path, _ = QFileDialog.getSaveFileName(self, "保存分割图像 (PNG)", default_name, "PNG 图像 (*.png)")

        if save_path:
            if not save_path.lower().endswith('.png'):
                save_path += '.png'

            self.is_saving = True
            self.update_button_states()
            self.show_status_message(f"保存分割图像到 {os.path.basename(save_path)}...", 0)

            worker = SaveWorker()
            worker_id = "save_segment"
            data_to_save = (original_cv_image_full_res, working_res_mask, scale_factor, refine_params)
            self._start_worker(worker_id, worker, "run_save", "segmented_image", data_to_save, save_path)

    @Slot(bool, str, str)
    def _handle_save_segment_completion(self, success, saved_path, error_message):
        # 【核心修复】：强制隐藏全局遮罩
        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.hide()

        self.is_saving = False
        self.update_button_states()

        if success:
            QMessageBox.information(self, "保存成功", f"抠图图像已保存:\n{saved_path}")
            interactive_label = getattr(self, 'segment_label', None)
            if interactive_label and interactive_label.current_mask is not None:
                current_state = interactive_label.current_mask.copy()
                interactive_label.mask_history.clear()
                interactive_label.redo_stack.clear()
                interactive_label.mask_history.append(current_state)
                interactive_label._cached_refined_mask = None
                QTimer.singleShot(0, self.update_button_states)
        else:
            QMessageBox.critical(self, "保存失败", f"保存抠图图像时出错:\n{error_message}")

    @Slot()
    def select_video_for_segment(self):
        if not (getattr(self, 'SAM2_VIDEO_PREDICTOR_AVAILABLE', True) and getattr(self, 'TORCH_AVAILABLE',
                                                                                  True) and getattr(self,
                                                                                                    'PILLOW_AVAILABLE',
                                                                                                    True)):
            QMessageBox.warning(self, "功能不可用", "视频/GIF抠图需要 SAM2 (视频组件)、PyTorch 和 Pillow。")
            return

        busy_video_op = getattr(self, 'is_extracting_frames', False) or getattr(self, 'video_segmentation_running',
                                                                                False) or (
                                getattr(self, 'is_saving', False) and "save_video" in getattr(self, 'active_workers',
                                                                                              {}))

        if getattr(self, 'is_loading_model', False) or busy_video_op:
            QMessageBox.information(self, "处理中", "其他视频相关操作正在进行中，请稍候。")
            return

        if not getattr(self, 'video_predictor_loaded', False):
            QMessageBox.warning(self, "预测器未就绪", "视频抠图预测器未加载。请稍候或检查日志。")
            return

        if getattr(self, 'sam_video_load_failed', False):
            QMessageBox.critical(self, "错误", "视频抠图预测器先前加载失败。")
            return

        if getattr(self, 'virtual_timeline', []):
            reply = QMessageBox.question(
                self, '加载新视频',
                '独立加载新视频将清空当前故事板所有片段和抠图结果。\n您确定要继续吗？\n(如需拼接，请点击左侧"项目库"的"添加"按钮)',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            self.virtual_timeline.clear()

        supported = ' '.join(['*' + ext for ext in SUPPORTED_VIDEO_FORMATS])
        fpath, _ = QFileDialog.getOpenFileName(
            self, "选择视频或GIF文件", "",
            f"视频/GIF 文件 ({supported});;所有文件 (*)"
        )

        if fpath:
            if hasattr(self, 'virtual_timeline'):
                self.virtual_timeline.clear()
            self._load_video_for_segmentation(fpath)

    def _load_video_for_segmentation(self, file_path):
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "加载错误", f"视频文件未找到: {file_path}")
            return

        self.add_to_recent_projects(file_path)

        if os.path.splitext(file_path)[1].lower() == '.gif' and not getattr(self, 'PILLOW_AVAILABLE', True):
            QMessageBox.warning(self, "GIF处理错误", "处理GIF文件需要 Pillow 库。")
            return

        if hasattr(self, '_add_video_to_library_ui'):
            self._add_video_to_library_ui(file_path)
            self._update_asset_library_ui()

        if hasattr(self, 'virtual_timeline') and len(self.virtual_timeline) == 0:
            self.reset_video_state()
            self._add_video_to_storyboard(file_path)
        else:
            self.show_status_message(f"已将视频添加到项目库: {os.path.basename(file_path)}", 3000)

    def _pump_gui_heartbeat(self):
        """Pumps GUI thread events manually to keep main layout responsive."""
        try:
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        except Exception:
            pass

    def _initialize_video_predictor_state(self) -> bool:
        if not self.video_predictor_loaded or not self.video_predictor:
            self.log_message.emit("无法初始化状态: 预测器未加载。")
            return False

        target_dir = getattr(self, 'clip_sandbox_dir', self.temp_frame_dir)
        if not target_dir:
            return False
        if getattr(self, 'video_inference_state', None) is not None:
            return True

        self.show_global_loading_overlay("正在编码视频特征")
        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.repaint()
        QApplication.processEvents()

        self.update_button_states()
        worker = VideoInitStateWorker()
        worker_id = "init_video_state"

        if hasattr(self, '_heartbeat_timer') and self._heartbeat_timer:
            try:
                self._heartbeat_timer.stop()
                self._heartbeat_timer.deleteLater()
            except Exception:
                pass

        self._heartbeat_timer = QTimer(self)
        self._heartbeat_timer.setInterval(50)
        self._heartbeat_timer.timeout.connect(self._pump_gui_heartbeat)
        self._heartbeat_timer.start()

        self._start_worker(worker_id, worker, "run_init", self.video_predictor, target_dir)

        ugly_dialog = getattr(self, f"current_{worker_id}_progress", None)
        if ugly_dialog:
            try:
                ugly_dialog.hide()
                ugly_dialog.setParent(None)
            except Exception:
                pass

        return True

    @Slot(object, bool, str)
    def _handle_init_video_state_completion(self, inference_state, success, error_msg):
        """
        特征编码工作线程完成后的回调槽函数。
        已重构：完全移除传统阻塞式 QProgressDialog 弹窗，采用非阻塞现代系统遮罩配合异步唤醒，避免未响应。
        """
        if hasattr(self, '_heartbeat_timer') and self._heartbeat_timer:
            try:
                self._heartbeat_timer.stop()
                self._heartbeat_timer.deleteLater()
            except Exception:
                pass
            self._heartbeat_timer = None

        if getattr(self, 'current_init_video_state_progress', None):
            try:
                self.current_init_video_state_progress.hide()
                self.current_init_video_state_progress.setParent(None)
                self.current_init_video_state_progress.close()
                self.current_init_video_state_progress.deleteLater()
            except Exception:
                pass
            self.current_init_video_state_progress = None

        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()

        if success and inference_state is not None:
            self.video_inference_state = inference_state
            self.log_message.emit("视频预测器状态初始化成功。")

            if hasattr(self, 'video_display_label'):
                self.video_display_label.set_allow_interaction(True)

            is_matting_page = getattr(self, 'vid_editor_stack', None) and \
                              self.vid_editor_stack.currentWidget() == getattr(self, 'vid_dedicated_matting_page', None)

            pending_resume = getattr(self, '_pending_interaction_resume', None)
            needs_heavy_compute = (pending_resume is not None) or is_matting_page

            if needs_heavy_compute:
                # 使用自定义非阻塞式全局遮罩层，锁定用户输入的同时保证系统事件流能够自由分发
                self.show_global_loading_overlay(_TR("正在唤醒 AI 引擎并重构历史图层，请勿操作..."), 0)

                def do_heavy_compute_async():
                    if pending_resume is not None:
                        self._pending_interaction_resume = None
                        pending_resume()
                    elif is_matting_page:
                        self._sync_sam2_after_history_jump()

                # 延迟 100ms 异步唤醒，给主线程留出充足的时间片刷新遮罩层界面
                QTimer.singleShot(100, do_heavy_compute_async)
            else:
                if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                    self._global_loading_overlay.hide()
                self.update_button_states()
                self.show_status_message(_TR("视频特征编码完毕。"), 3000)
        else:
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            self.log_message.emit(f"初始化视频预测器状态出错: {error_msg}")
            QTimer.singleShot(50, lambda: QMessageBox.critical(self, _TR("预测器错误"),
                                                               f"{_TR('初始化视频预测器状态失败:')} {error_msg}"))
            self.reset_video_state()
            self.update_button_states()

    def _cleanup_temp_frame_dir(self):
        if self.temp_frame_dir and os.path.exists(self.temp_frame_dir):
            self.log_message.emit(f"Cleaning up temp frame directory: {self.temp_frame_dir}")
            try:
                shutil.rmtree(self.temp_frame_dir)
                self.temp_frame_dir = None
            except Exception as e:
                self.log_message.emit(f"Warning: Unable to remove temp frame directory '{self.temp_frame_dir}': {e}")
        else:
            self.temp_frame_dir = None

    def reset_video_state(self):
        """
        重置全局视频编辑轨道状态。
        物理释放所有临时音频、烘焙帧缓存，并重置相关 UI 视口控件至初始状态。
        """
        import os
        import shutil

        # 1. 物理暂停视频播放并切断音频流
        if getattr(self, 'is_playing', False):
            self.pause_video()

        self._clear_audio_sources_safe()

        # 2. 销毁异步寻帧线程，释放 PyAV 文件占用
        if hasattr(self, '_stop_async_scrub_reader'):
            self._stop_async_scrub_reader()

        # 3. 安全清理所有独立片段专属的烘焙融合缓存目录
        if hasattr(self, 'virtual_timeline') and self.virtual_timeline:
            for clip in self.virtual_timeline:
                baked_dir = clip.get('baked_preview_dir', None)
                if baked_dir and os.path.exists(baked_dir):
                    try:
                        shutil.rmtree(baked_dir, ignore_errors=True)
                    except Exception as e:
                        print(f"清理片段烘焙目录失败: {e}")
            self.virtual_timeline.clear()

        # 4. 清理时间轴合并帧主目录
        self._cleanup_temp_frame_dir()

        # 5. 安全清理局部沙盒以及全局预览烘焙目录
        if getattr(self, 'clip_sandbox_dir', None) and os.path.exists(self.clip_sandbox_dir):
            try:
                shutil.rmtree(self.clip_sandbox_dir, ignore_errors=True)
            except Exception:
                pass
        self.clip_sandbox_dir = None

        if getattr(self, 'temp_render_dir', None) and os.path.exists(self.temp_render_dir):
            try:
                shutil.rmtree(self.temp_render_dir, ignore_errors=True)
            except Exception:
                pass
        self.temp_render_dir = None

        # 6. 重置缓存和时序特征标志
        if hasattr(self, '_video_frame_cache'):
            self._video_frame_cache.clear()

        if hasattr(self, '_timeline_signature'):
            self._timeline_signature = None

        self._last_built_sandbox_clip_idx = -1

        # 7. 中止正在运行中的后台视频任务线程
        for worker_id in ["extract", "propagate_video_v1991", "save_video", "init_video_state", "video_sync_history"]:
            if worker_id in getattr(self, 'active_workers', {}):
                self._cancel_worker(worker_id)

        if "load_video_model" in getattr(self, 'active_workers', {}):
            self._cancel_worker("load_video_model")

        # 8. 属性变量复位
        self.video_path = None
        self.total_frames = 0
        self.video_fps = VIDEO_DEFAULT_FPS
        self.video_width = 0
        self.video_height = 0
        self.current_frame_index = -1
        self.is_gif_input = False
        self.gif_frame_duration_ms = int(1000 / VIDEO_DEFAULT_FPS)
        self.gif_frame_durations_ms = []
        self.video_thumbnail_paths = []
        self.video_inference_state = None
        self.processed_masks = {}
        self.target_points = {}
        self.current_target_id = -1
        self.next_target_id = 0
        self.video_segmentation_running = False
        self.video_segmentation_finished = False
        self.video_segmentation_saved = False

        # 9. UI 视口及故事板列表重置
        if hasattr(self, 'storyboard_list'):
            self.storyboard_list.blockSignals(True)
            self.storyboard_list.clear()
            self.storyboard_list.blockSignals(False)

        if hasattr(self, '_update_storyboard_ui'):
            self._update_storyboard_ui()

        if hasattr(self, 'video_display_label'):
            self.video_display_label.clear_display()
            self.video_display_label.set_allow_interaction(False)

        if hasattr(self, 'video_result_preview_label'):
            self._clear_video_result_preview()

        if hasattr(self, 'video_thumbnail_scrubber'):
            self.video_thumbnail_scrubber.set_params(0, [], 0, 0, VIDEO_DEFAULT_FPS, False, [])
            self.video_thumbnail_scrubber.set_info_text("", "[0/0]")

        if hasattr(self, 'video_frame_spinbox'):
            self.video_frame_spinbox.setMaximum(1)
            self.video_frame_spinbox.setValue(1)

        if hasattr(self, 'video_info_label_display'):
            self.video_info_label_display.setText("请先加载视频或GIF文件...")

        if hasattr(self, 'play_pause_button'):
            self.play_pause_button.setChecked(False)

        self._update_current_target_label()
        self.update_button_states()

    def _get_frame_path(self, frame_index: int) -> Optional[str]:
        if self.temp_frame_dir is None or frame_index < 0:
            return None

        padded_path = os.path.join(self.temp_frame_dir, f"{frame_index:05d}.jpg")
        if os.path.exists(padded_path):
            return padded_path

        return os.path.join(self.temp_frame_dir, f"{frame_index}{VIDEO_FRAME_EXT}")

    def _get_current_bg_frame(self, target_h: int, target_w: int, frame_index: int = -1) -> np.ndarray:
        """
        根据帧索引，精准获取其对应视频分段的专属背景画布，防止段间背景溢出。
        """
        if frame_index == -1:
            frame_index = getattr(self, 'current_frame_index', 0)

        # 1. 确定该全局帧位置所匹配的片段索引
        clip_idx = self._get_clip_index_for_frame(frame_index)
        if clip_idx == -1 or not hasattr(self, 'virtual_timeline') or clip_idx >= len(self.virtual_timeline):
            # 默认兜底绿幕背景
            return np.full((target_h, target_w, 3), (0, 255, 0), dtype=np.uint8)

        # 2. 仅获取对应片段的背景设置，不访问其他片段
        clip = self.virtual_timeline[clip_idx]
        bg_is_transparent = clip.get('bg_is_transparent', False)
        bg_image_path = clip.get('bg_image_path', None)
        bg_color = clip.get('bg_color', QColor(0, 255, 0))

        # 3. 渲染透明网格背景
        if bg_is_transparent:
            checker_size = 20
            y_indices = (np.arange(target_h) // checker_size) % 2
            x_indices = (np.arange(target_w) // checker_size) % 2
            mask = (y_indices[:, None] == x_indices[None, :])
            bg = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            bg[mask] = [40, 40, 40]
            bg[~mask] = [60, 60, 60]
            return bg

        # 4. 渲染自定义背景图片并按比例填充
        if bg_image_path and os.path.exists(bg_image_path):
            try:
                custom_bg_img = imread_unicode(bg_image_path, cv2.IMREAD_COLOR)
                if custom_bg_img is not None:
                    if len(custom_bg_img.shape) == 2:
                        custom_bg_img = cv2.cvtColor(custom_bg_img, cv2.COLOR_GRAY2BGR)
                    elif custom_bg_img.shape[2] == 4:
                        custom_bg_img = cv2.cvtColor(custom_bg_img, cv2.COLOR_BGRA2BGR)

                    bg_h, bg_w = custom_bg_img.shape[:2]
                    scale = max(target_w / float(bg_w), target_h / float(bg_h))
                    new_w = max(target_w, int(math.ceil(bg_w * scale)))
                    new_h = max(target_h, int(math.ceil(bg_h * scale)))

                    resized_bg = cv2.resize(custom_bg_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    x_offset = (new_w - target_w) // 2
                    y_offset = (new_h - target_h) // 2
                    crop_bg = resized_bg[y_offset:y_offset + target_h, x_offset:x_offset + target_w].copy()
                    return crop_bg
            except Exception as e:
                print(f"解析片段背景图片失败: {e}")

        # 5. 渲染纯色背景
        bg_bgr = (bg_color.blue(), bg_color.green(), bg_color.red())
        return np.full((target_h, target_w, 3), bg_bgr, dtype=np.uint8)

    def _do_prefetch_task(self, frame_index: int):
        """后台预读小工：默默把未来的帧从硬盘搬到内存里"""
        if not getattr(self, 'temp_frame_dir', None):
            return

        cache_key = f"global_{self.temp_frame_dir}_{frame_index}"

        # 检查是否已经被主线程或其他小工读过了
        with self._prefetch_lock:
            if cache_key in self._video_frame_cache:
                self._prefetching_indices.discard(frame_index)
                return

        # 极其耗时的 I/O 操作：读盘与解码 (在后台线程执行，完全不卡界面)
        frame_path = os.path.join(self.temp_frame_dir, f"{frame_index:05d}.jpg")
        if os.path.exists(frame_path):
            img = imread_unicode(frame_path)
            if img is not None:
                # 写入全局缓存池
                with self._prefetch_lock:
                    self._video_frame_cache[cache_key] = img
                    # 防止撑爆内存，预读池设为 300 帧 (约10秒视频，占用1GB左右显存/内存)
                    if len(self._video_frame_cache) > 300:
                        self._video_frame_cache.popitem(last=False)

        # 汇报当前帧预读任务完成
        with self._prefetch_lock:
            self._prefetching_indices.discard(frame_index)

    def _read_frame_from_disk(self, frame_index: int) -> Optional[np.ndarray]:
        """
        读取视频帧：当抠图完成后，定向切换为高帧率无卡顿读取已烘焙完成的离屏融合图像。
        """
        if frame_index < 0:
            return None

        global_frame_index = frame_index

        # 当抠图完成且烘焙目录准备就绪，自动重定向到烘焙目录，省去所有实时混合开销
        active_frame_dir = self.temp_frame_dir
        if getattr(self, 'video_segmentation_finished', False) and getattr(self, 'temp_render_dir', None) and os.path.exists(self.temp_render_dir):
            active_frame_dir = self.temp_render_dir

        if not hasattr(self, '_prefetch_pool'):
            from concurrent.futures import ThreadPoolExecutor
            import threading
            import collections
            self._prefetch_pool = ThreadPoolExecutor(max_workers=4)
            self._prefetch_lock = threading.Lock()
            self._prefetching_indices = set()
            self._video_frame_cache = collections.OrderedDict()

        cache_key = f"global_{active_frame_dir}_{global_frame_index}"
        hit_image = None
        cache_hit = False

        if active_frame_dir and os.path.exists(active_frame_dir):
            with self._prefetch_lock:
                if cache_key in self._video_frame_cache:
                    self._video_frame_cache.move_to_end(cache_key)
                    hit_image = self._video_frame_cache[cache_key]
                    cache_hit = True

            if not cache_hit:
                frame_path = os.path.join(active_frame_dir, f"{global_frame_index:05d}.jpg")
                if os.path.exists(frame_path):
                    hit_image = imread_unicode(frame_path)
                    if hit_image is not None:
                        with self._prefetch_lock:
                            self._video_frame_cache[cache_key] = hit_image
                            if len(self._video_frame_cache) > 300:
                                self._video_frame_cache.popitem(last=False)

            # 触发未来帧的异步读取
            if getattr(self, 'is_playing', False):
                for i in range(1, 6):
                    next_idx = global_frame_index + i
                    if next_idx >= getattr(self, 'total_frames', 99999):
                        break
                    next_cache_key = f"global_{active_frame_dir}_{next_idx}"
                    needs_prefetch = False
                    with self._prefetch_lock:
                        if next_cache_key not in self._video_frame_cache and next_idx not in self._prefetching_indices:
                            self._prefetching_indices.add(next_idx)
                            needs_prefetch = True
                    if needs_prefetch:
                        self._prefetch_pool.submit(self._do_prefetch_task_custom, next_idx, active_frame_dir)

            if hit_image is not None:
                return hit_image

        return None

    def _do_prefetch_task_custom(self, frame_index: int, active_frame_dir: str):
        """流式后台高频读盘预载小工（兼容烘焙重定向）"""
        if not active_frame_dir:
            return

        cache_key = f"global_{active_frame_dir}_{frame_index}"

        with self._prefetch_lock:
            if cache_key in self._video_frame_cache:
                self._prefetching_indices.discard(frame_index)
                return

        frame_path = os.path.join(active_frame_dir, f"{frame_index:05d}.jpg")
        if os.path.exists(frame_path):
            img = imread_unicode(frame_path)
            if img is not None:
                with self._prefetch_lock:
                    self._video_frame_cache[cache_key] = img
                    if len(self._video_frame_cache) > 300:
                        self._video_frame_cache.popitem(last=False)

        with self._prefetch_lock:
            self._prefetching_indices.discard(frame_index)

    def _read_unmatted_frame_from_disk(self, frame_index: int) -> Optional[np.ndarray]:
        """
        专门提供给对比滑块在播放时读取原未抠图视频帧。
        """
        if not getattr(self, 'temp_frame_dir', None):
            return None
        frame_path = os.path.join(self.temp_frame_dir, f"{frame_index:05d}.jpg")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(self.temp_frame_dir, f"{frame_index}.jpg")
        if os.path.exists(frame_path):
            return imread_unicode(frame_path)
        return None

    def _display_frame_wrapper(self, frame_index: int, cv_image: Optional[np.ndarray] = None):
        """
        展示帧核心渲染包装器。
        【修复优化】：在对比模式（Compare Mode）下，强制使左侧视图保持纯净原图（无遮罩色块覆膜），
        右侧视图实时或缓存渲染带背景抠图成果，提升可视交互体验。
        """
        if not hasattr(self, 'video_display_label'):
            return

        if getattr(self, 'is_playing', False) and hasattr(self, 'virtual_timeline') and self.virtual_timeline:
            active_clip_idx = self._get_clip_index_for_frame(frame_index)
            if active_clip_idx != -1 and active_clip_idx != getattr(self, '_current_playing_clip_idx', -1):
                accum_frames = sum(v['frames'] for v in self.virtual_timeline[:active_clip_idx])
                local_frame_idx = frame_index - accum_frames
                self._sync_audio_engine_to_current_frame(
                    force_clip_idx=active_clip_idx,
                    force_local_frame=local_frame_idx
                )

        is_main_page = getattr(self, 'vid_editor_stack', None) and self.vid_editor_stack.currentWidget() == getattr(self, 'vid_main_editor_page', None)
        is_matting_page = getattr(self, 'vid_editor_stack', None) and self.vid_editor_stack.currentWidget() == getattr(self, 'vid_dedicated_matting_page', None)

        if is_matting_page:
            bgr_frame = self._read_unmatted_frame_from_disk(frame_index)
        else:
            if cv_image is None:
                cv_image = self._read_frame_from_disk(frame_index)
            bgr_frame = cv_image

        if bgr_frame is None:
            bgr_frame = self._read_frame_from_disk(frame_index)

        if bgr_frame is None:
            self.video_display_label.clear_display()
            return

        bgr_frame = bgr_frame.copy()
        if len(bgr_frame.shape) == 2:
            bgr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_GRAY2BGR)
        elif bgr_frame.shape[2] == 4:
            bgr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGRA2BGR)
        h, w = bgr_frame.shape[:2]

        is_pre_rendered = getattr(self, 'video_segmentation_finished', False) and getattr(self, 'temp_render_dir', None) and os.path.exists(self.temp_render_dir)

        if is_pre_rendered:
            if is_matting_page:
                final_preview_bgr = self._read_frame_from_disk(frame_index)
                if final_preview_bgr is None:
                    final_preview_bgr = bgr_frame.copy()
            else:
                final_preview_bgr = bgr_frame.copy()
            display_frame = bgr_frame.copy()
        else:
            final_preview_bgr = self._get_current_bg_frame(h, w)
            combined_alpha_for_frame = np.zeros((h, w), dtype=np.float32)
            has_mask = False
            ui_display_masks = {}

            def align_mask_to_canvas(mask_in):
                if mask_in.shape[:2] == (h, w):
                    return mask_in
                mh, mw = mask_in.shape[:2]
                m_scale = min(w / mw, h / mh)
                new_mw, new_mh = int(mw * m_scale), int(mh * m_scale)
                resized_m = cv2.resize(mask_in.astype(np.float32), (new_mw, new_mh), interpolation=cv2.INTER_LINEAR)
                canvas_m = np.zeros((h, w), dtype=np.float32)
                mx_off = (w - new_mw) // 2
                my_off = (h - new_mh) // 2
                canvas_m[my_off:my_off + new_mh, mx_off:mx_off + new_mw] = resized_m
                return canvas_m

            frame_masks = self.processed_masks.get(frame_index, {})
            for obj_id, mask_np in frame_masks.items():
                if mask_np is not None:
                    mask_float = mask_np.astype(np.float32) if mask_np.dtype == bool else np.clip(mask_np.astype(np.float32), 0.0, 1.0)
                    mask_aligned = align_mask_to_canvas(mask_float)
                    combined_alpha_for_frame = np.maximum(combined_alpha_for_frame, mask_aligned)
                    ui_display_masks[obj_id] = mask_aligned
                    has_mask = True

            if hasattr(self, 'video_display_label'):
                temp_multi = getattr(self.video_display_label, 'temp_multi_masks', {}).get(frame_index, {})
                for obj_id, temp_np in temp_multi.items():
                    if temp_np is not None:
                        temp_float = temp_np.astype(np.float32) if temp_np.dtype == bool else np.clip(temp_np.astype(np.float32), 0.0, 1.0)
                        temp_aligned = align_mask_to_canvas(temp_float)
                        combined_alpha_for_frame = np.maximum(combined_alpha_for_frame, temp_aligned)
                        ui_display_masks[obj_id] = temp_aligned
                        has_mask = True

            if has_mask:
                alpha_3d = combined_alpha_for_frame[:, :, np.newaxis]
                blended = bgr_frame.astype(np.float32) * alpha_3d + final_preview_bgr.astype(np.float32) * (1.0 - alpha_3d)
                final_preview_bgr = blended.astype(np.uint8)
            else:
                final_preview_bgr = bgr_frame.copy()

            display_frame = bgr_frame.copy()

        if is_matting_page:
            ui_display_masks = {}
            frame_masks = self.processed_masks.get(frame_index, {})

            def align_mask_to_canvas_local(mask_in):
                if mask_in.shape[:2] == (h, w):
                    return mask_in
                mh, mw = mask_in.shape[:2]
                m_scale = min(w / mw, h / mh)
                new_mw, new_mh = int(mw * m_scale), int(mh * m_scale)
                resized_m = cv2.resize(mask_in.astype(np.float32), (new_mw, new_mh), interpolation=cv2.INTER_LINEAR)
                canvas_m = np.zeros((h, w), dtype=np.float32)
                mx_off = (w - new_mw) // 2
                my_off = (h - new_mh) // 2
                canvas_m[my_off:my_off + new_mh, mx_off:mx_off + new_mw] = resized_m
                return canvas_m

            for obj_id, mask_np in frame_masks.items():
                if mask_np is not None:
                    mask_float = mask_np.astype(np.float32) if mask_np.dtype == bool else np.clip(mask_np.astype(np.float32), 0.0, 1.0)
                    ui_display_masks[obj_id] = align_mask_to_canvas_local(mask_float)

            if hasattr(self, 'video_display_label'):
                temp_multi = getattr(self.video_display_label, 'temp_multi_masks', {}).get(frame_index, {})
                for obj_id, temp_np in temp_multi.items():
                    if temp_np is not None:
                        temp_float = temp_np.astype(np.float32) if temp_np.dtype == bool else np.clip(temp_np.astype(np.float32), 0.0, 1.0)
                        ui_display_masks[obj_id] = align_mask_to_canvas_local(temp_float)

            if ui_display_masks:
                active_id = getattr(self, 'current_target_id', -1)
                for obj_id, mask_float in ui_display_masks.items():
                    if mask_float is None:
                        continue
                    mask_alpha = 0.35 if active_id == -1 else (0.65 if active_id == obj_id else 0.15)
                    color_idx = obj_id % len(VIDEO_TARGET_COLORS)
                    qcolor = VIDEO_TARGET_COLORS[color_idx]
                    color_bgr = np.array([qcolor.blue(), qcolor.green(), qcolor.red()], dtype=np.float32)

                    alpha_map = (mask_float * mask_alpha)[:, :, np.newaxis]
                    display_frame = (display_frame.astype(np.float32) * (1.0 - alpha_map) + color_bgr * alpha_map).astype(np.uint8)

                    if active_id == obj_id:
                        mask_bool = mask_float > 0.1
                        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cv2.drawContours(display_frame, contours, -1, (255, 255, 255), 2)

        is_compare = False
        if is_matting_page and hasattr(self, 'video_display_label'):
            is_compare = getattr(self.video_display_label, '_is_in_compare_mode', False)

        if is_main_page and hasattr(self, 'video_display_label'):
            self.video_display_label.set_compare_mode(False)

        points_for_display = {}
        boxes_for_display = {}
        if is_matting_page:
            for target_id, target_data in getattr(self, 'target_points', {}).items():
                if target_data.get('annotation_frame') == frame_index:
                    points_for_display[target_id] = target_data.get('points', [])
                    if target_data.get('box'):
                        boxes_for_display[target_id] = target_data['box']

        if is_compare:
            # 【重要修复】：左侧对比原图不再显示 display_frame 的标注覆膜，始终统一采用最纯净的原始 unmatted bgr_frame！
            self.video_display_label.set_compare_pixmaps(
                convert_cv_to_pixmap(bgr_frame),
                convert_cv_to_pixmap(final_preview_bgr)
            )
            self.video_display_label.set_frame(
                display_frame, frame_index,
                interaction_pts_for_display=points_for_display,
                seg_masks_for_display={},
                temp_annotation_mask_tuple=None,
                interaction_boxes_for_display=boxes_for_display
            )
        else:
            if is_main_page and getattr(self, 'video_segmentation_finished', False):
                self.video_display_label.set_frame(final_preview_bgr, frame_index)
            else:
                self.video_display_label.set_frame(
                    display_frame, frame_index,
                    interaction_pts_for_display=points_for_display,
                    seg_masks_for_display={},
                    temp_annotation_mask_tuple=None,
                    interaction_boxes_for_display=boxes_for_display
                )

        if hasattr(self, 'video_thumbnail_scrubber'):
            is_isolated = getattr(self, '_matting_clip_idx', -1) != -1
            if is_isolated:
                local_idx = frame_index - getattr(self, '_matting_global_start', 0)
                self.video_thumbnail_scrubber.set_current_frame(max(0, local_idx))
            else:
                self.video_thumbnail_scrubber.set_current_frame(frame_index)

        self._update_video_time_label()

    def _get_clip_index_for_frame(self, frame_index: int) -> int:
        """
        根据全局绝对帧索引计算当前属于哪一个视频片段。
        """
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline:
            return -1
        current_count = 0
        for i, vid in enumerate(self.virtual_timeline):
            if frame_index < current_count + vid['frames']:
                return i
            current_count += vid['frames']
        return len(self.virtual_timeline) - 1

    def _safe_copy_virtual_timeline(self, timeline):
        """Safely clones active video timeline structures without serializing QPixmap handles."""
        copied_timeline = []
        for clip in timeline:
            clip_copy = {}
            for k, v in clip.items():
                if k == 'pixmap':
                    if isinstance(v, QPixmap) and not v.isNull():
                        clip_copy[k] = QPixmap(v)
                    else:
                        clip_copy[k] = QPixmap()
                elif k == 'local_masks':
                    masks_copy = {}
                    for f_idx, obj_masks in v.items():
                        masks_copy[f_idx] = {}
                        for o_id, mask in obj_masks.items():
                            if isinstance(mask, np.ndarray):
                                masks_copy[f_idx][o_id] = mask.copy()
                            else:
                                masks_copy[f_idx][o_id] = None
                    clip_copy[k] = masks_copy
                elif k == 'local_targets':
                    clip_copy[k] = copy.deepcopy(v)
                else:
                    clip_copy[k] = copy.deepcopy(v)
            copied_timeline.append(clip_copy)
        return copied_timeline

    def _is_state_identical(self, s1, s2):
        """Compares target parameters of timeline states to filter out redundant updates."""
        if s1['current_target_id'] != s2['current_target_id']:
            return False
        if s1['next_target_id'] != s2['next_target_id']:
            return False
        if s1['current_frame_index'] != s2['current_frame_index']:
            return False

        if len(s1['virtual_timeline']) != len(s2['virtual_timeline']):
            return False
        for c1, c2 in zip(s1['virtual_timeline'], s2['virtual_timeline']):
            if c1['path'] != c2['path']:
                return False
            if c1['in_point'] != c2['in_point']:
                return False
            if c1['out_point'] != c2['out_point']:
                return False
            if c1.get('mute_original') != c2.get('mute_original'):
                return False
            if c1.get('original_audio_volume') != c2.get('original_audio_volume'):
                return False
            if c1.get('custom_audio_path') != c2.get('custom_audio_path'):
                return False

        if len(s1['target_points']) != len(s2['target_points']):
            return False
        for t_id, t_data in s1['target_points'].items():
            if t_id not in s2['target_points']:
                return False
            t_data2 = s2['target_points'][t_id]
            if t_data['points'] != t_data2['points']:
                return False
            if t_data['box'] != t_data2['box']:
                return False
            if t_data.get('annotation_frame') != t_data2.get('annotation_frame'):
                return False

        return True

    def _save_video_state(self):
        """Saves current state properties onto the video undo stack."""
        if not hasattr(self, 'video_undo_stack'):
            self.video_undo_stack = []
            self.video_redo_stack = []

        timeline_snapshot = self._safe_copy_virtual_timeline(self.virtual_timeline)

        processed_masks_copy = {}
        for f_idx, masks in self.processed_masks.items():
            processed_masks_copy[f_idx] = {}
            for o_id, m in masks.items():
                if m is not None:
                    processed_masks_copy[f_idx][o_id] = m.copy()
                else:
                    processed_masks_copy[f_idx][o_id] = None

        state = {
            'virtual_timeline': timeline_snapshot,
            'target_points': copy.deepcopy(self.target_points),
            'current_target_id': self.current_target_id,
            'next_target_id': self.next_target_id,
            'processed_masks': processed_masks_copy,
            'current_frame_index': self.current_frame_index
        }

        if not self.video_undo_stack or not self._is_state_identical(state, self.video_undo_stack[-1]):
            self.video_undo_stack.append(state)
            if len(self.video_undo_stack) > 30:
                self.video_undo_stack.pop(0)

        self.video_redo_stack.clear()
        self.update_button_states()

    def _restore_video_state(self, state):
        """Restores timeline elements from historical state variables."""
        self.virtual_timeline = self._safe_copy_virtual_timeline(state['virtual_timeline'])
        self.target_points = copy.deepcopy(state['target_points'])
        self.current_target_id = state['current_target_id']
        self.next_target_id = state['next_target_id']

        self.processed_masks = {}
        for f_idx, masks in state['processed_masks'].items():
            self.processed_masks[f_idx] = {}
            for o_id, m in masks.items():
                if m is not None:
                    self.processed_masks[f_idx][o_id] = m.copy()
                else:
                    self.processed_masks[f_idx][o_id] = None

        self.current_frame_index = state.get('current_frame_index', 0)
        self._current_playing_clip_idx = -1

        if hasattr(self, 'audio_player'):
            self.audio_player.stop()
            self.bgm_player.stop()

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

        if hasattr(self, '_update_storyboard_highlight'):
            self._update_storyboard_highlight()
        self._update_storyboard_ui()

        self._recalc_global_timeline()
        self._refresh_video_objects_list()
        self._update_current_target_label()
        self.update_button_states()
        self._trigger_global_timeline_rebuild()

    @Slot()
    def undo_video_action(self):
        """Pops and rolls back last recorded state from history stacks."""
        if not hasattr(self, 'video_undo_stack') or not self.video_undo_stack:
            return

        if getattr(self, 'is_playing', False):
            self.pause_video()

        current_state = {
            'virtual_timeline': self._safe_copy_virtual_timeline(self.virtual_timeline),
            'target_points': copy.deepcopy(self.target_points),
            'current_target_id': self.current_target_id,
            'next_target_id': self.next_target_id,
            'processed_masks': {
                f_idx: {o_id: m.copy() if m is not None else None for o_id, m in masks.items()}
                for f_idx, masks in self.processed_masks.items()
            },
            'current_frame_index': self.current_frame_index
        }

        if not self.video_redo_stack or not self._is_state_identical(current_state, self.video_redo_stack[-1]):
            self.video_redo_stack.append(current_state)

        state = self.video_undo_stack.pop()
        self._restore_video_state(state)

    @Slot()
    def redo_video_action(self):
        """Pops and re-applies configurations from redo structures."""
        if not hasattr(self, 'video_redo_stack') or not self.video_redo_stack:
            return

        if getattr(self, 'is_playing', False):
            self.pause_video()

        current_state = {
            'virtual_timeline': self._safe_copy_virtual_timeline(self.virtual_timeline),
            'target_points': copy.deepcopy(self.target_points),
            'current_target_id': self.current_target_id,
            'next_target_id': self.next_target_id,
            'processed_masks': {
                f_idx: {o_id: m.copy() if m is not None else None for o_id, m in masks.items()}
                for f_idx, masks in self.processed_masks.items()
            },
            'current_frame_index': self.current_frame_index
        }

        if not self.video_undo_stack or not self._is_state_identical(current_state, self.video_undo_stack[-1]):
            self.video_undo_stack.append(current_state)
            if len(self.video_undo_stack) > 30:
                self.video_undo_stack.pop(0)

        state = self.video_redo_stack.pop()
        self._restore_video_state(state)

    def _sync_sam2_after_history_jump(self):
        """
        异步后台线程计算，同步当前活动追踪器与时间线修改，防止主线程未响应。
        已重构：直接将子线程的图层重算百分比对接到全局非阻塞遮罩层，让进度条动起来。
        """
        self.show_global_loading_overlay("正在重建历史图层关联结构...", 0)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

        if not getattr(self, 'video_predictor', None) or not getattr(self, 'video_inference_state', None):
            self._refresh_video_objects_list()
            self._display_frame_wrapper(self.current_frame_index)
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            self.update_button_states()
            return

        self._refresh_video_objects_list()

        worker = VideoSyncHistoryWorker()
        worker_id = "video_sync_history"

        try:
            worker.setParent(None)
        except Exception:
            pass

        thread = QThread(self)
        worker.moveToThread(thread)

        # 【核心改进】：直接将异步计算的实时进度关联至全局现代遮罩，保持心跳更新
        worker.progress.connect(lambda pct, msg: self.show_global_loading_overlay(f"重构历史图层: {msg}", pct))

        def on_sync_complete(temp_multi_masks, success, error_msg):
            if success:
                if hasattr(self, 'video_display_label') and self.video_display_label:
                    self.video_display_label.temp_multi_masks = temp_multi_masks

                    active_id = self.current_target_id
                    current_frame = self.current_frame_index

                    if active_id != -1 and current_frame in temp_multi_masks and active_id in temp_multi_masks[
                        current_frame]:
                        self.video_display_label.temp_annotation_frame_mask = temp_multi_masks[current_frame][
                            active_id].copy()
                        self.video_display_label.temp_annotation_target_id = active_id
                        self.video_display_label.temp_annotation_mask_frame_idx = current_frame
                    else:
                        self.video_display_label.temp_annotation_frame_mask = None
                        self.video_display_label.temp_annotation_target_id = -1
                        self.video_display_label.temp_annotation_mask_frame_idx = -1

                    self.video_display_label.set_active_object(self.current_target_id)
            else:
                self.log_message.emit(f"异步图层结构计算出错: {error_msg}")

            thread.quit()
            worker.deleteLater()
            self._remove_active_worker(worker_id)

            self._display_frame_wrapper(self.current_frame_index)
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

            # 重构工作全部结束后，平滑隐退全局遮罩
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            self.update_button_states()
            self.show_status_message(_TR("视频环境准备完毕，操作已解锁！"), 3000)

        worker.sync_complete.connect(on_sync_complete)
        thread.finished.connect(thread.deleteLater)

        thread.started.connect(lambda: worker.run_sync(
            self.video_predictor,
            self.video_inference_state,
            self.target_points,
            getattr(self, '_matting_global_start', 0)
        ))

        self.active_workers[worker_id] = (thread, worker)
        thread.start()

    def _clear_video_result_preview(self):
        if hasattr(self, 'video_result_preview_label'):
            self.video_result_preview_label.clear()
            self.video_result_preview_label.setText("视频抠图结果预览")

    @Slot(int)
    def seek_video_via_scrubber(self, frame_index: int):
        """
        视频进度条拖拽槽。
        升级：音频精准对齐剥离到帧变更判断之外，确保松手瞬间必然执行声音对齐。
        """
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline:
            return

        is_isolated = getattr(self, '_matting_clip_idx', -1) != -1

        # 通过判断鼠标左键是否按下，决定当前是否正处于 Scrubbing（拖动中）
        left_button_pressed = bool(QApplication.mouseButtons() & Qt.MouseButton.LeftButton)
        self._is_scrubbing_timeline = left_button_pressed

        if is_isolated:
            clip_frames = self.virtual_timeline[self._matting_clip_idx]['frames']
            target_local_idx = max(0, min(frame_index, clip_frames - 1))
            target_global_idx = self._matting_global_start + target_local_idx

            if target_global_idx != self.current_frame_index:
                self.current_frame_index = target_global_idx
                if getattr(self, 'is_playing', False):
                    self.pause_video()

                # 仅绘制视频图像
                self._display_frame_wrapper(self.current_frame_index)
        else:
            if 0 <= frame_index < getattr(self, 'total_frames', 0):
                if frame_index != self.current_frame_index:
                    self.current_frame_index = frame_index
                    if getattr(self, 'is_playing', False):
                        self.pause_video()

                    # 仅绘制视频图像
                    self._display_frame_wrapper(self.current_frame_index)

        # 【核心修复】：将音频对齐逻辑移出 if 帧数变化的块！
        # 只要没有在拖拽（也就是松开了鼠标左键），就必须强制执行一次声音底层对齐！
        if not self._is_scrubbing_timeline:
            self._sync_audio_engine_to_current_frame()

    @Slot()
    def seek_video_via_spinbox(self):
        if not hasattr(self, 'video_frame_spinbox') or not self.video_path:
            return

        self.video_frame_spinbox.blockSignals(True)
        frame_index_one_based = self.video_frame_spinbox.value()
        frame_index_zero_based = frame_index_one_based - 1

        if 0 <= frame_index_zero_based < self.total_frames:
            if frame_index_zero_based != self.current_frame_index:
                self.current_frame_index = frame_index_zero_based
                if self.is_playing:
                    self.pause_video()
                self._display_frame_wrapper(self.current_frame_index)
        else:
            self.video_frame_spinbox.setValue(self.current_frame_index + 1)

        self.video_frame_spinbox.blockSignals(False)
        self.video_frame_spinbox.clearFocus()

    @Slot(bool)
    def toggle_play_pause(self, checked: bool):
        if not self.video_path or self.total_frames <= 0:
            if hasattr(self, 'play_pause_button'):
                self.play_pause_button.blockSignals(True)
                self.play_pause_button.setChecked(False)
                self.play_pause_button.blockSignals(False)
            return

        if checked:
            self.play_video()
        else:
            self.pause_video()
        self.update_button_states()

    @Slot()
    def play_video(self):
        """Starts background rendering sequence and triggers audio tracks sync."""
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline or self.total_frames <= 0 or getattr(self,
                                                                                                                   'is_playing',
                                                                                                                   False):
            return

        # 【核心修复1：显卡/硬件加速注入】
        # 强制系统渲染走独立显卡与底层硬件通道，消除集显带来的渲染卡顿
        import os
        os.environ["QT_OPENGL_BUILTIN"] = "1"
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

        if self.current_frame_index >= self.total_frames - 1:
            self.current_frame_index = 0
            self._display_frame_wrapper(self.current_frame_index)

        self.is_playing = True

        # 记录内部时钟起点，供无声视频使用
        import time
        self._play_start_time = time.time()
        self._play_start_frame = self.current_frame_index

        if hasattr(self, 'play_pause_button'):
            self.play_pause_button.setIcon(self._create_svg_icon("pause-fill.svg", color="#E0E0E0"))

        self._sync_audio_engine_to_current_frame()

        if hasattr(self, 'mpv_audio') and getattr(self.mpv_audio, 'path', None):
            self.mpv_audio.pause = False

        clip_idx = getattr(self, '_current_playing_clip_idx', -1)
        if clip_idx != -1 and clip_idx < len(self.virtual_timeline):
            bgm_path = self.virtual_timeline[clip_idx].get('custom_audio_path')
            if bgm_path and os.path.exists(bgm_path) and hasattr(self, 'mpv_bgm') and getattr(self.mpv_bgm, 'path',
                                                                                              None):
                self.mpv_bgm.pause = False

        self.show_status_message("视频播放中...", 0)
        if hasattr(self, 'video_display_label'):
            self.video_display_label.set_allow_interaction(False)

        self._update_bgm_remove_button_state()
        self.update_button_states()

        # 【核心修复2：启动混合时钟监控狗】
        if not hasattr(self, '_playback_watchdog'):
            self._playback_watchdog = QTimer(self)
            self._playback_watchdog.setInterval(16)  # 约 60fps 刷新率，保证极致流畅
            self._playback_watchdog.timeout.connect(self._on_playback_watchdog_tick)
        self._playback_watchdog.start()

    @Slot()
    def _on_playback_watchdog_tick(self):
        """
        【混合高保真时钟监控狗】：始终作为第一主时钟推动画面前进，
        同时结合 mpv 音频反馈进行自适应校准，彻底解决跨片、无声视频、音频 EOF 导致的卡顿问题。
        """
        if not getattr(self, 'is_playing', False):
            if hasattr(self, '_playback_watchdog'):
                self._playback_watchdog.stop()
            return

        is_isolated = getattr(self, '_matting_clip_idx', -1) != -1
        max_frame = getattr(self, '_matting_global_end',
                            self.total_frames - 1) if is_isolated else self.total_frames - 1

        # 1. 依靠系统绝对物理时间计算期望帧，保证时钟流逝的连续性
        import time
        elapsed_sec = time.time() - getattr(self, '_play_start_time', time.time())
        expected_frame = self._calculate_expected_frame(
            getattr(self, '_play_start_frame', self.current_frame_index), elapsed_sec)

        # 2. 如果原声音频正在流畅播放，则使用 mpv 的高精度 C 指针时间对齐当前帧，防止音画不同步
        audio_is_driving = False
        if hasattr(self, 'mpv_audio') and getattr(self.mpv_audio, 'path', None) is not None:
            if not getattr(self.mpv_audio, 'pause', True) and not getattr(self.mpv_audio, 'eof_reached', True):
                audio_is_driving = True

        if audio_is_driving:
            audio_time = getattr(self.mpv_audio, 'time_pos', None)
            if audio_time is not None:
                clip_idx = getattr(self, '_current_playing_clip_idx', -1)
                if clip_idx != -1 and clip_idx < len(self.virtual_timeline):
                    clip = self.virtual_timeline[clip_idx]
                    fps = clip.get('fps', 30.0)
                    in_point = clip.get('in_point', 0)

                    # 换算当前原声音频对应的局部帧
                    local_target_frame = int((audio_time + 0.03) * fps) - in_point

                    # 映射到全局绝对帧位置
                    if is_isolated:
                        target_global_frame = self._matting_global_start + local_target_frame
                    else:
                        target_global_frame = sum(
                            v['frames'] for v in self.virtual_timeline[:clip_idx]) + local_target_frame

                    if 0 <= target_global_frame <= max_frame:
                        expected_frame = target_global_frame

                        # 反向高精度修正绝对时间锚点，以便在跨片/无声时无缝接管
                        current_vid_sec = local_target_frame / max(1.0, fps)
                        self._play_start_time = time.time() - current_vid_sec
                        self._play_start_frame = expected_frame - local_target_frame

        # 3. 边界限定保护
        if expected_frame > max_frame:
            expected_frame = max_frame

        # 4. 如果计算出的目标帧有改变，立即刷新视口渲染并检查 BGM 的播放边界
        if expected_frame != self.current_frame_index:
            self.current_frame_index = expected_frame
            self._display_frame_wrapper(self.current_frame_index)
            self._sync_audio_playback_tick()

        # 5. 到达视频片段尽头，执行安全停止
        if self.current_frame_index >= max_frame:
            self.stop_video()

    @Slot()
    def _check_playback_eof(self):
        """【防卡死监控狗】：接管末尾 0.0x 秒的画面补齐与跳转逻辑"""
        if not getattr(self, 'is_playing', False):
            if hasattr(self, '_eof_watchdog'):
                self._eof_watchdog.stop()
            return

        is_isolated = getattr(self, '_matting_clip_idx', -1) != -1
        max_frame = getattr(self, '_matting_global_end',
                            self.total_frames - 1) if is_isolated else self.total_frames - 1

        # 1. 正常到达最后一帧，触发停止并跳回开头
        if self.current_frame_index >= max_frame:
            self.stop_video()
            return

        # 2. 如果 MPV 音频比视频短，音频提前结束了 (eof_reached 为 True)
        # 这时时间戳停止更新，画面会卡住。我们手动接管，把剩下的几帧飞速播完，然后回到开头
        if hasattr(self, 'mpv_audio') and getattr(self.mpv_audio, 'eof_reached', False):
            self.current_frame_index += 1
            if self.current_frame_index > max_frame:
                self.current_frame_index = max_frame

            self._display_frame_wrapper(self.current_frame_index)

            if self.current_frame_index >= max_frame:
                self.stop_video()

    def _start_playback_timer(self):
        interval = VIDEO_PLAYBACK_INTERVAL_MS
        if self.is_gif_input and self.gif_frame_durations_ms and 0 <= self.current_frame_index < len(
                self.gif_frame_durations_ms):
            interval = self.gif_frame_durations_ms[self.current_frame_index]
        elif self.video_fps > 0:
            interval = int(1000 / self.video_fps)

        self.playback_timer.start(max(10, interval))

    def pause_video(self):
        if not getattr(self, 'is_playing', False):
            return

        self.is_playing = False

        # 停止混合时钟监控狗
        if hasattr(self, '_playback_watchdog'):
            self._playback_watchdog.stop()

        if hasattr(self, 'mpv_audio'):
            self.mpv_audio.pause = True
        if hasattr(self, 'mpv_bgm'):
            self.mpv_bgm.pause = True

        if hasattr(self, 'play_pause_button'):
            self.play_pause_button.setIcon(self._create_svg_icon("play-fill.svg", color="#E0E0E0"))
            if self.play_pause_button.isChecked():
                self.play_pause_button.blockSignals(True)
                self.play_pause_button.setChecked(False)
                self.play_pause_button.blockSignals(False)

        self.show_status_message("视频已暂停。", 2000)
        self.update_button_states()

    @Slot()
    def stop_video(self):
        """【MPV版】停止视频：解决播放完毕未跳转第一帧及引擎重置问题"""
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline:
            return

        self.pause_video()

        is_isolated = getattr(self, '_matting_clip_idx', -1) != -1
        self.current_frame_index = getattr(self, '_matting_global_start', 0) if is_isolated else 0

        self._display_frame_wrapper(self.current_frame_index)
        self._sync_audio_engine_to_current_frame(force_local_frame=0)

        self.show_status_message("视频已停止并回到首帧。", 2000)
        self.update_button_states()

    @Slot(float)
    def _sync_frame_to_audio_clock(self, current_audio_time_sec):
        """
        音频物理时钟回调（已作安全纯被动处理）。
        主推动力现已统一交由 watch dog 独立进行，此处的被动刷新已被降级以防指令抖动。
        """
        pass

    def _calculate_expected_frame(self, start_frame: int, elapsed_sec: float) -> int:
        """
        推算辅助函数：根据真实流逝时间精准定位目标帧。
        """
        is_isolated = getattr(self, '_matting_clip_idx', -1) != -1
        if is_isolated:
            clip = self.virtual_timeline[self._matting_clip_idx]
            fps = clip.get('fps', 30.0)
            return start_frame + int(elapsed_sec * fps)

        current_count = 0
        accumulated_time = 0.0

        start_clip_idx = 0
        local_start_frame = 0
        for i, vid in enumerate(self.virtual_timeline):
            if start_frame < current_count + vid['frames']:
                start_clip_idx = i
                local_start_frame = start_frame - current_count
                break
            current_count += vid['frames']

        for i in range(start_clip_idx, len(self.virtual_timeline)):
            vid = self.virtual_timeline[i]
            fps = vid.get('fps', 30.0)

            if i == start_clip_idx:
                frames_left_in_clip = vid['frames'] - local_start_frame
                time_left_in_clip = frames_left_in_clip / max(1.0, fps)

                if elapsed_sec < accumulated_time + time_left_in_clip:
                    time_in_this_clip = elapsed_sec - accumulated_time
                    return current_count + local_start_frame + int(time_in_this_clip * fps)
                accumulated_time += time_left_in_clip
            else:
                clip_time = vid['frames'] / max(1.0, fps)
                if elapsed_sec < accumulated_time + clip_time:
                    time_in_this_clip = elapsed_sec - accumulated_time
                    return current_count + int(time_in_this_clip * fps)
                accumulated_time += clip_time

            current_count += vid['frames']

        return self.total_frames - 1

    def _sync_audio_playback_tick(self):
        """【MPV版】背景播放音频实时同步Tick（用于精准卡点，判定 BGM 是否应该停止）"""
        clip_idx = getattr(self, '_current_playing_clip_idx', -1)
        if clip_idx == -1 or not hasattr(self, 'virtual_timeline') or clip_idx >= len(self.virtual_timeline):
            return

        clip = self.virtual_timeline[clip_idx]
        bgm_path = clip.get('custom_audio_path')

        # 【关键修改】：如果没有设置背景音乐，强制物理清理底层缓冲区，防止漏音
        if not bgm_path or not os.path.exists(bgm_path):
            if hasattr(self, 'mpv_bgm'):
                try:
                    self.mpv_bgm.command("stop")
                except Exception:
                    pass
            return

        if not hasattr(self, 'mpv_bgm') or not getattr(self.mpv_bgm, 'path', None):
            return

        local_frame = self.current_frame_index - sum(v['frames'] for v in self.virtual_timeline[:clip_idx])
        fps = clip.get('fps', 30.0)
        video_time_sec = local_frame / fps

        video_insert_sec = clip.get('custom_audio_start_sec', 0.0)
        bgm_clip_start_sec = clip.get('custom_audio_clip_start', 0.0)
        bgm_clip_end_sec = clip.get('custom_audio_clip_end', 0.0)

        bgm_play_duration = bgm_clip_end_sec - bgm_clip_start_sec
        if bgm_play_duration <= 0:
            bgm_play_duration = 9999.0

        # 如果播放超出音乐裁剪范围，令其物理暂停
        if video_time_sec < video_insert_sec or (video_time_sec - video_insert_sec) >= bgm_play_duration:
            if not getattr(self.mpv_bgm, 'pause', True):
                self.mpv_bgm.pause = True
        else:
            if getattr(self, 'is_playing', False):
                if getattr(self.mpv_bgm, 'pause', True):
                    bgm_play_time_sec = (video_time_sec - video_insert_sec) + bgm_clip_start_sec
                    try:
                        self.mpv_bgm.seek(bgm_play_time_sec, reference='absolute', precision='exact')
                    except Exception:
                        pass
                    self.mpv_bgm.pause = False
                else:
                    bgm_pos_sec = getattr(self.mpv_bgm, 'time_pos', 0.0)
                    if bgm_pos_sec is not None and bgm_pos_sec >= bgm_clip_end_sec:
                        self.mpv_bgm.pause = True

    @Slot(int)
    def step_video_frame(self, delta: int):
        """
        逐帧步进控制（前进/后退 1 帧）。
        修复：1. 在独立抠图沙盒 (is_isolated=True) 下，将上下界正确锁定在 clip 的绝对全局帧区间，
                 防止点击下一帧按钮时因 local 0-N Clamping 导致画面瞬间回弹到第一个视频。
        """
        if getattr(self, 'is_playing', False):
            self.pause_video()

        total = getattr(self, 'total_frames', 0)
        if total <= 0:
            return

        is_isolated = getattr(self, '_matting_clip_idx', -1) != -1
        if is_isolated:
            # =========================================================================
            # 【核心修复】：将限幅的 min_frame 与 max_frame 指向沙盒在合并轨上的绝对物理索引区间。
            # 彻底杜绝使用 0 到 local_frames - 1 克制而产生的首视频越界渗漏。
            # =========================================================================
            min_frame = getattr(self, '_matting_global_start', 0)
            max_frame = getattr(self, '_matting_global_end', 0)
        else:
            min_frame = 0
            max_frame = total - 1

        current = getattr(self, 'current_frame_index', 0)
        new_frame = max(min_frame, min(current + delta, max_frame))

        if new_frame != current:
            self.current_frame_index = new_frame
            self._display_frame_wrapper(self.current_frame_index)
            self._sync_audio_engine_to_current_frame()

    @Slot(bool)
    def set_clip_mute_state(self, is_muted: bool):
        """智能抠图沙盒界面的静音勾选框，控制片段全部声音"""
        idx = getattr(self, '_matting_clip_idx', -1)
        if idx == -1 or not hasattr(self, 'virtual_timeline') or idx >= len(self.virtual_timeline):
            return

        clip = self.virtual_timeline[idx]
        clip['mute_all'] = is_muted

        # 即时同步播放器
        if getattr(self, '_current_playing_clip_idx', -1) == idx:
            self._sync_audio_engine_to_current_frame(idx)

        # 同步故事板图标
        if hasattr(self, 'storyboard_list'):
            item_widget = self.storyboard_list.itemWidget(self.storyboard_list.item(idx))
            if hasattr(item_widget, 'sync_ui_with_data'):
                item_widget.sync_ui_with_data(is_muted)

    def _sync_audio_engine_to_current_frame(self, force_clip_idx=-1, force_local_frame=-1):
        """
        音频位置追踪与深度切换器。
        【核心修复】：精准区分 mute_all, mute_original, mute_bgm 控制逻辑
        """
        if not hasattr(self, 'mpv_audio') or not hasattr(self, 'virtual_timeline'):
            return

        if getattr(self, '_is_scrubbing_timeline', False):
            return

        clip_idx = force_clip_idx
        local_frame = force_local_frame
        is_isolated = getattr(self, '_matting_clip_idx', -1) != -1

        if clip_idx == -1:
            if is_isolated:
                clip_idx = self._matting_clip_idx
                local_frame = self.current_frame_index - getattr(self, '_matting_global_start', 0)
            else:
                current_count = 0
                for i, vid in enumerate(self.virtual_timeline):
                    if self.current_frame_index < current_count + vid['frames']:
                        clip_idx = i
                        local_frame = self.current_frame_index - current_count
                        break
                    current_count += vid['frames']

        if clip_idx == -1 or clip_idx >= len(self.virtual_timeline):
            return

        clip = self.virtual_timeline[clip_idx]
        fps = clip.get('fps', 30.0)

        # 精准判定静音状态
        total_mute_orig = clip.get('mute_all', False) or clip.get('mute_original', False) or getattr(self,
                                                                                                     '_is_global_muted',
                                                                                                     False)
        total_mute_bgm = clip.get('mute_all', False) or clip.get('mute_bgm', False) or getattr(self, '_is_global_muted',
                                                                                               False)

        orig_vol = clip.get('original_audio_volume', 1.0)
        bgm_vol = clip.get('custom_audio_volume', 1.0)

        clip_changed = getattr(self, '_current_playing_clip_idx', -1) != clip_idx

        try:
            # ================== A. 原声音轨物理切换 ==================
            target_path = clip['path']
            current_audio_path = getattr(self.mpv_audio, 'path', None)

            in_point = clip.get('in_point', 0)
            target_physical_frame = in_point + local_frame
            target_orig_time_sec = target_physical_frame / max(1.0, fps)

            is_eof = getattr(self.mpv_audio, 'eof_reached', False)

            if current_audio_path is None or clip_changed or os.path.normpath(current_audio_path) != os.path.normpath(
                    target_path) or is_eof:
                self.mpv_audio.pause = True
                self.mpv_audio['start'] = str(target_orig_time_sec)
                self.mpv_audio.play(target_path)
                self._current_playing_clip_idx = clip_idx

                if getattr(self, 'is_playing', False):
                    self.mpv_audio.pause = False
            else:
                self._current_playing_clip_idx = clip_idx
                current_pos = getattr(self.mpv_audio, 'time_pos', None)
                if current_pos is None:
                    current_pos = 0.0

                if abs(current_pos - target_orig_time_sec) > 0.08:
                    try:
                        self.mpv_audio.seek(target_orig_time_sec, reference='absolute', precision='exact')
                    except Exception:
                        pass

                if getattr(self, 'is_playing', False):
                    self.mpv_audio.pause = False

            # 应用静音与音量
            self.mpv_audio.volume = 0 if total_mute_orig else int(orig_vol * 100)

            # ================== B. BGM 物理切换与彻底隔离 ==================
            bgm_path = clip.get('custom_audio_path', None)
            if bgm_path and os.path.exists(bgm_path):
                video_time_sec = local_frame / max(1.0, fps)
                video_insert_sec = clip.get('custom_audio_start_sec', 0.0)
                bgm_clip_start_sec = clip.get('custom_audio_clip_start', 0.0)
                bgm_clip_end_sec = clip.get('custom_audio_clip_end', 0.0)

                bgm_play_duration = bgm_clip_end_sec - bgm_clip_start_sec
                if bgm_play_duration <= 0:
                    bgm_play_duration = 9999.0

                bgm_play_time_sec = (video_time_sec - video_insert_sec) + bgm_clip_start_sec

                current_bgm_path = getattr(self.mpv_bgm, 'path', None)
                bgm_is_eof = getattr(self.mpv_bgm, 'eof_reached', False)

                if current_bgm_path is None or clip_changed or os.path.normpath(current_bgm_path) != os.path.normpath(
                        bgm_path) or bgm_is_eof:
                    self.mpv_bgm.pause = True
                    self.mpv_bgm['start'] = str(bgm_play_time_sec)
                    self.mpv_bgm.play(bgm_path)
                else:
                    current_bgm_pos = getattr(self.mpv_bgm, 'time_pos', None)
                    if current_bgm_pos is None:
                        current_bgm_pos = 0.0

                    if abs(current_bgm_pos - bgm_play_time_sec) > 0.08:
                        try:
                            self.mpv_bgm.seek(bgm_play_time_sec, reference='absolute', precision='exact')
                        except Exception:
                            pass

                # 应用 BGM 静音与音量
                self.mpv_bgm.volume = 0 if total_mute_bgm else int(bgm_vol * 100)

                if video_time_sec < video_insert_sec or (video_time_sec - video_insert_sec) >= bgm_play_duration:
                    self.mpv_bgm.pause = True
                else:
                    if getattr(self, 'is_playing', False):
                        self.mpv_bgm.pause = False
            else:
                if hasattr(self, 'mpv_bgm'):
                    try:
                        self.mpv_bgm.command("stop")
                    except Exception:
                        pass

        except Exception as e:
            print(f"[Audio Sync Error] MPV 同步异常: {e}")

    def _update_bgm_remove_button_state(self):
        """【MPV版】根据播放状态更新删除按钮可用性"""
        if not hasattr(self, 'btn_remove_bgm') or self.btn_remove_bgm is None:
            return

        has_bgm = False
        idx = getattr(self, '_current_crop_clip_idx', -1)
        if idx != -1 and hasattr(self, 'virtual_timeline') and idx < len(self.virtual_timeline):
            clip = self.virtual_timeline[idx]
            if clip.get('custom_audio_path'):
                has_bgm = True

        is_playing_now = False
        if has_bgm:
            try:
                bgm_is_playing = False
                if hasattr(self, 'mpv_bgm'):
                    bgm_is_playing = not getattr(self.mpv_bgm, 'pause', True)

                is_playing_now = (
                        getattr(self, 'is_playing', False) or
                        getattr(self, '_is_playing_bgm_preview', False) or
                        bgm_is_playing
                )
            except Exception:
                is_playing_now = False

        self.btn_remove_bgm.setEnabled(not is_playing_now)

        if is_playing_now:
            self.btn_remove_bgm.setToolTip(_TR("音乐播放中，请先暂停播放再移除背景音乐"))
            self.btn_remove_bgm.setStyleSheet("""
                QPushButton { background-color: #1C1C1E; color: #555558; border-radius: 6px; padding: 10px 12px; font-weight: bold; font-size: 13px; border: 1px solid #2C2C2E; }
            """)
        else:
            self.btn_remove_bgm.setToolTip("")
            self.btn_remove_bgm.setStyleSheet("""
                QPushButton { background-color: #2C2C2E; color: #FFFFFF; border-radius: 6px; padding: 10px 12px; font-weight: bold; font-size: 13px; border: 1px solid #3A3A3C; } 
                QPushButton:hover { background-color: #3A3A3C; border: 1px solid #555555; }
                QPushButton:pressed { background-color: #1C1C1E; }
            """)

    def _clear_audio_sources_safe(self):
        """【MPV版】安全切断所有的发声源"""
        self._current_playing_clip_idx = -1
        if hasattr(self, 'mpv_audio'):
            try:
                self.mpv_audio.command("stop")
            except Exception:
                pass
        if hasattr(self, 'mpv_bgm'):
            try:
                self.mpv_bgm.command("stop")
            except Exception:
                pass

    def _update_video_time_label(self):
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline or getattr(self, 'total_frames', 0) <= 0:
            if hasattr(self, 'video_thumbnail_scrubber'):
                self.video_thumbnail_scrubber.set_info_text("", "[0/0]")
            if hasattr(self, 'time_label_curr'):
                self.time_label_curr.setText("0:00.00")
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

    def _select_video_object_by_id(self, obj_id: int):
        if not hasattr(self, 'video_objects_list'):
            return

        self.video_objects_list.blockSignals(True)
        self.video_objects_list.clearSelection()
        self.current_target_id = obj_id

        if obj_id != -1:
            for i in range(self.video_objects_list.count()):
                item = self.video_objects_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == obj_id:
                    item.setSelected(True)
                    break
        self.video_objects_list.blockSignals(False)

        if hasattr(self, 'video_display_label'):
            self.video_display_label.set_active_object(self.current_target_id)

        self._display_frame_wrapper(self.current_frame_index)
        self.update_button_states()

    def handle_video_interaction_point(self, pos_img: QPointF, button_type: int):
        """
        Handles mouse clicks representing positive/negative feedback points on the video viewport.
        Saves undo states, registers coordinates under the active target, and schedules async model runs.
        """
        busy_video_op = getattr(self, 'is_extracting_frames', False) or getattr(self, 'video_segmentation_running',
                                                                                False) or (
                                getattr(self, 'is_saving', False) and "save_video" in getattr(self, 'active_workers',
                                                                                              {})
                        )
        if not getattr(self, 'video_path', None) or getattr(self, 'is_playing', False) or busy_video_op:
            return

        if getattr(self, 'video_segmentation_finished', False):
            self.video_segmentation_finished = False
            if hasattr(self, 'video_display_label'):
                self.video_display_label.set_compare_mode(False)

        if getattr(self, 'current_target_id', -1) == -1:
            return

        target_data = self.target_points.get(self.current_target_id)
        if not target_data:
            return

        global_frame_idx = self.current_frame_index

        if 'annotation_frame' not in target_data or target_data['annotation_frame'] is None:
            target_data['annotation_frame'] = global_frame_idx
            self._refresh_video_objects_list()

        if target_data['annotation_frame'] != global_frame_idx:
            reply = QMessageBox.question(
                self, "帧不匹配", "是否立刻跳转到首帧？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                is_isolated = getattr(self, '_matting_clip_idx', -1) != -1
                if is_isolated:
                    local_target_idx = target_data['annotation_frame'] - self._matting_global_start
                    self.seek_video_via_scrubber(local_target_idx)
                else:
                    self.seek_video_via_scrubber(target_data['annotation_frame'])
            return
        else:
            self._save_video_state()
            target_data['points'].append((pos_img.x(), pos_img.y(), button_type))
            self._refresh_video_objects_list()

        try:
            self.video_display_label.video_points_changed.emit()
        except AttributeError:
            pass

        self._display_frame_wrapper(self.current_frame_index)

        if getattr(self, 'video_predictor', None):
            if getattr(self, 'video_inference_state', None) is None:
                def resume_interaction_after_init():
                    self._start_async_sam2_video_interaction(self.current_target_id, global_frame_idx, target_data)

                self._pending_interaction_resume = resume_interaction_after_init
                self._initialize_video_predictor_state()
                return

            self._start_async_sam2_video_interaction(self.current_target_id, global_frame_idx, target_data)
        else:
            self.update_button_states()

    def _execute_sam2_interaction(self, target_data):
        """
        Executes raw SAM2 point/box inclusion queries inside the video predictor interface.
        """
        print(f"DEBUG_INTERACT: Executed mouse click. Active Target ID = {self.current_target_id}")

        input_points_np = np.array([(p[0], p[1]) for p in target_data['points']], dtype=np.float32)
        if input_points_np.ndim == 1 and input_points_np.size > 0:
            input_points_np = input_points_np[np.newaxis, :]
        input_labels_np = np.array([p[2] for p in target_data['points']], dtype=np.int32)

        box_np = np.array(target_data['box'], dtype=np.float32) if target_data.get('box') else None

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            local_ann_frame = self.current_frame_index - getattr(self, '_matting_global_start', 0)

            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=self.video_inference_state,
                frame_idx=local_ann_frame,
                obj_id=self.current_target_id,
                points=input_points_np,
                labels=input_labels_np,
                box=box_np
            )

            if out_mask_logits is not None and len(out_mask_logits) > 0 and out_obj_ids is not None:
                for i, obj_id_tensor in enumerate(out_obj_ids):
                    obj_id_val = int(obj_id_tensor.item()) if hasattr(obj_id_tensor, 'item') else int(obj_id_tensor)

                    mask = (out_mask_logits[i].float() > 0.0).cpu().numpy().squeeze()
                    if mask.dtype != bool:
                        mask = mask.astype(bool)
                    mask = np.ascontiguousarray(mask)

                    if not hasattr(self.video_display_label, 'temp_multi_masks'):
                        self.video_display_label.temp_multi_masks = {}
                    if self.current_frame_index not in self.video_display_label.temp_multi_masks:
                        self.video_display_label.temp_multi_masks[self.current_frame_index] = {}

                    self.video_display_label.temp_multi_masks[self.current_frame_index][obj_id_val] = mask

        except Exception as e:
            print(f"DEBUG_INTERACT: SAM2 inference failed.\n{e}")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    @Slot()
    def add_new_video_target(self):
        """
        Creates and activates a new tracking layer ID.
        【核心优化】：点击添加新对象时，强制退出对比模式准备交互。由于我们没有摧毁预测器内存，旧目标完美保留且不会丢失！
        """
        busy_video_op = getattr(self, 'is_extracting_frames', False) or getattr(self, 'video_segmentation_running',
                                                                                False) or (
                                getattr(self, 'is_saving', False) and "save_video" in getattr(self, 'active_workers',
                                                                                              {})
                        )
        if not getattr(self, 'video_path', None) or getattr(self, 'is_playing', False) or busy_video_op:
            QMessageBox.warning(self, "无法添加", "请加载视频或等待当前操作完成。")
            return
        if getattr(self, 'total_frames', 0) <= 0:
            return

        # =========================================================================
        # 【核心修改 5】：退出纯享对比模式，进入蒙版标记模式
        # =========================================================================
        if hasattr(self, 'video_compare_mode_button') and self.video_compare_mode_button.isChecked():
            self.video_compare_mode_button.setChecked(False)
        if hasattr(self, 'video_display_label'):
            self.video_display_label.set_compare_mode(False)

        max_allowed = globals().get('MAX_VIDEO_OBJS', 10)
        if len(self.target_points) >= max_allowed:
            QMessageBox.warning(self, "无法添加", f"已达到最大目标对象数量 ({max_allowed})。")
            return

        self._save_video_state()

        existing_ids = list(self.target_points.keys())
        new_id = 0
        while new_id in existing_ids:
            new_id += 1

        color_idx = new_id % len(VIDEO_TARGET_COLORS)
        self.target_points[new_id] = {
            'points': [], 'box': None, 'color': VIDEO_TARGET_COLORS[color_idx],
            'annotation_frame': None, 'name': f"对象 {new_id + 1}", 'visible': True
        }

        self.current_target_id = new_id
        self.next_target_id = max(existing_ids + [new_id]) + 1
        self.log_message.emit(f"Added new target {new_id}.")

        self._refresh_video_objects_list()

        if hasattr(self, 'video_display_label'):
            self.video_display_label.temp_annotation_frame_mask = None
            self.video_display_label.temp_annotation_target_id = -1

        self._display_frame_wrapper(self.current_frame_index)
        self.update_button_states()

    @Slot()
    def _on_video_object_selection_changed(self):
        """
        追踪对象列表中选择的目标改变时的响应。
        【核心优化】：只要用户点击了列表里的旧对象，立刻解除纯享对比模式，展现出交互蒙版！
        """
        selected_items = self.video_objects_list.selectedItems()
        if selected_items:
            obj_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
            self.current_target_id = obj_id

            # 自动对齐寻道
            target_data = self.target_points.get(obj_id)
            if target_data and target_data.get('annotation_frame') is not None:
                ann_frame = target_data['annotation_frame']
                if ann_frame != self.current_frame_index:
                    # 暂停播放以安全对齐时间轴
                    if getattr(self, 'is_playing', False):
                        self.pause_video()

                    self.current_frame_index = ann_frame
                    if hasattr(self, 'video_thumbnail_scrubber'):
                        self.video_thumbnail_scrubber.set_current_frame(ann_frame)

                    self._sync_audio_engine_to_current_frame()

            # =========================================================================
            # 【核心修改 4】：点击旧对象时，立刻关闭滑块退出对比模式，展现底层遮罩以便重新编辑
            # =========================================================================
            if hasattr(self, 'video_compare_mode_button') and self.video_compare_mode_button.isChecked():
                self.video_compare_mode_button.setChecked(False)
            if hasattr(self, 'video_display_label'):
                self.video_display_label.set_compare_mode(False)

            self.log_message.emit(f"Selected target: {obj_id}, entering edit mode.")
        else:
            self.current_target_id = -1
            self.log_message.emit("Deselected targets, exiting edit mode.")

        if hasattr(self, 'video_display_label'):
            self.video_display_label.set_active_object(self.current_target_id)

        # 重新渲染视口
        self._display_frame_wrapper(self.current_frame_index)
        self.update_button_states()

    def _refresh_video_objects_list(self):
        """
        Re-renders items populating the multi-target list component in the video matting view.
        """
        if not hasattr(self, 'video_objects_list'):
            return

        self.video_objects_list.blockSignals(True)
        self.video_objects_list.clear()

        for obj_id, target_data in self.target_points.items():
            list_item = QListWidgetItem(self.video_objects_list)
            list_item.setData(Qt.ItemDataRole.UserRole, obj_id)

            item_widget = QWidget()
            item_widget.setStyleSheet("background: transparent;")

            main_layout = QHBoxLayout(item_widget)
            main_layout.setContentsMargins(4, 0, 4, 0)
            main_layout.setSpacing(10)

            color_indicator = QLabel()
            color_indicator.setFixedSize(10, 10)
            qcolor = target_data.get('color', QColor(255, 0, 0))
            color_indicator.setStyleSheet(
                f"background-color: {qcolor.name()}; border-radius: 5px; border: 1px solid #555;"
            )
            main_layout.addWidget(color_indicator)

            raw_name = target_data.get('name', f"对象 {obj_id + 1}")
            translated_name = raw_name

            if "对象" in raw_name:
                translated_name = raw_name.replace("对象", _TR("对象"))
            else:
                translated_name = _TR(raw_name)

            name_label = QLabel(translated_name)
            name_label.setStyleSheet("font-size: 13px; background: transparent; border: none;")

            if obj_id == getattr(self, 'current_target_id', -1):
                name_label.setStyleSheet(
                    "font-weight: bold; color: #FFFFFF; font-size: 13px; background: transparent; border: none;"
                )
                list_item.setSelected(True)
            else:
                name_label.setStyleSheet("color: #8E8E93; font-size: 13px; background: transparent; border: none;")

            main_layout.addWidget(name_label, 1)
            list_item.setSizeHint(QSize(280, 32))
            self.video_objects_list.setItemWidget(list_item, item_widget)

        self.video_objects_list.blockSignals(False)

    def _update_current_target_label(self):
        """
        更新当前工作区选择的目标对象的文本标签，防止显示 0 进制索引（如“对象 0”）。
        """
        if not hasattr(self, 'current_target_label'):
            return

        if self.current_target_id == -1 or self.current_target_id not in self.target_points:
            self.current_target_label.setText(_TR("无 (请先添加)"))
            self.current_target_label.setProperty("orig_text", "无 (请先添加)")
            self.current_target_label.setStyleSheet("")
        else:
            obj_id = self.current_target_id
            target_data = self.target_points[obj_id]
            qcolor = target_data['color']
            annotation_frame = target_data.get('annotation_frame')
            frame_defined_on_str = str(annotation_frame + 1) if isinstance(annotation_frame, int) else '?'
            num_points = len(target_data.get('points', []))

            # 采用 1-based 进制防止出现“目标 0”
            display_id = obj_id + 1
            text_cn = f"目标 {display_id} (标注于帧 {frame_defined_on_str}, 点数: {num_points})"
            translated_text = f"{_TR('目标')} {display_id} ({_TR('标注于帧')} {frame_defined_on_str}, {_TR('点数:')} {num_points})"

            self.current_target_label.setText(translated_text)
            self.current_target_label.setProperty("orig_text", text_cn)

            text_color = 'white' if qcolor.lightnessF() < 0.5 else 'black'
            self.current_target_label.setStyleSheet(
                f"background-color: {qcolor.name()}; color: {text_color}; border-radius: 8px; padding: 4px 6px;"
            )

    @Slot()
    def clear_current_target_points(self):
        if getattr(self, 'current_target_id', -1) == -1 or self.current_target_id not in self.target_points:
            QMessageBox.warning(self, "无法清除", "未选择目标对象。")
            return

        target_data = self.target_points[self.current_target_id]
        display_id = self.current_target_id + 1
        if not target_data.get('points') and target_data.get('box') is None:
            QMessageBox.information(self, "无需清除", f"目标 {display_id} 尚无交互点或框。")
            return

        reply = QMessageBox.question(
            self, "确认清除",
            f"您确定要清除该目标的所有交互点和框吗？\n(可通过撤回恢复)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._is_matting_dirty = True
            self._save_video_state()

            if hasattr(self.video_display_label,
                       'temp_annotation_target_id') and self.video_display_label.temp_annotation_target_id == self.current_target_id:
                self.video_display_label.temp_annotation_frame_mask = None
                self.video_display_label.temp_annotation_target_id = -1
                self.video_display_label.temp_annotation_mask_frame_idx = -1

            if self.current_frame_index in self.processed_masks and self.current_target_id in self.processed_masks[
                self.current_frame_index]:
                del self.processed_masks[self.current_frame_index][self.current_target_id]

            target_data['points'] = []
            target_data['box'] = None

            if getattr(self, 'temp_render_dir', None) and os.path.exists(self.temp_render_dir):
                try:
                    shutil.rmtree(self.temp_render_dir, ignore_errors=True)
                except Exception:
                    pass
            self.temp_render_dir = None
            self.video_segmentation_finished = False

            if hasattr(self, '_video_frame_cache'):
                self._video_frame_cache.clear()

            self._distribute_global_masks_to_clips()
            self._rebuild_matted_preview_cache()

            self.video_display_label.video_points_changed.emit()
            self._display_frame_wrapper(self.current_frame_index)
            self._refresh_video_objects_list()
            self.update_button_states()

    @Slot()
    def clear_all_video_targets(self):
        if not self.target_points:
            QMessageBox.information(self, _TR("无需清除"), _TR("当前未定义目标对象。"))
            return

        reply = QMessageBox.question(
            self, _TR("确认清除"),
            _TR("您确定要清除所有目标对象及其交互点吗？\n这也将清除所有生成的预览和完整抠图结果！"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._is_matting_dirty = True
            self.target_points = {}
            self.current_target_id = -1
            self.next_target_id = 0
            self.processed_masks = {}
            self.video_segmentation_finished = False
            self.video_segmentation_saved = False

            # [修复 Issue 2]: 清除全部目标时，物理移除该片段对应的预览混合缓存
            if getattr(self, 'temp_render_dir', None) and os.path.exists(self.temp_render_dir):
                try:
                    shutil.rmtree(self.temp_render_dir, ignore_errors=True)
                except Exception:
                    pass
            self.temp_render_dir = None

            if hasattr(self, '_video_frame_cache'):
                self._video_frame_cache.clear()

            self.log_message.emit("Cleared all video targets and segmentation results.")

            if hasattr(self, 'video_display_label') and hasattr(self.video_display_label, 'temp_annotation_target_id'):
                self.video_display_label.temp_annotation_frame_mask = None
                self.video_display_label.temp_annotation_target_id = -1
                self.video_display_label.temp_annotation_mask_frame_idx = -1

            if self.video_predictor and self.video_inference_state:
                try:
                    self.video_predictor.reset_state(self.video_inference_state)
                    self.log_message.emit("Video predictor state reset.")
                except Exception as e:
                    self.log_message.emit(f"Warning: Error resetting predictor state: {e}")

            self._update_current_target_label()
            self._display_frame_wrapper(self.current_frame_index)
            self.update_button_states()

    @Slot(QRectF)
    def handle_video_interaction_box(self, box_rect: QRectF):
        """
        Handles dragging of selection boxes inside the workspace layout and forwards coordinates.
        """
        busy_video_op = getattr(self, 'is_extracting_frames', False) or getattr(self, 'video_segmentation_running',
                                                                                False) or (
                                getattr(self, 'is_saving', False) and "save_video" in getattr(self, 'active_workers',
                                                                                              {})
                        )
        if not getattr(self, 'video_path', None) or getattr(self, 'is_playing', False) or busy_video_op:
            return

        if getattr(self, 'video_segmentation_finished', False):
            self.video_segmentation_finished = False
            if hasattr(self, 'video_display_label'):
                self.video_display_label.set_compare_mode(False)

        if getattr(self, 'current_target_id', -1) == -1:
            return

        target_data = self.target_points.get(self.current_target_id)
        if not target_data:
            return

        global_frame_idx = self.current_frame_index

        if 'annotation_frame' not in target_data or target_data['annotation_frame'] is None:
            target_data['annotation_frame'] = global_frame_idx
            self._refresh_video_objects_list()

        if target_data['annotation_frame'] != global_frame_idx:
            reply = QMessageBox.question(
                self, "帧不匹配", "是否立刻跳转到首帧？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                is_isolated = getattr(self, '_matting_clip_idx', -1) != -1
                if is_isolated:
                    local_target_idx = target_data['annotation_frame'] - self._matting_global_start
                    self.seek_video_via_scrubber(local_target_idx)
                else:
                    self.seek_video_via_scrubber(target_data['annotation_frame'])
            return
        else:
            self._save_video_state()
            target_data['box'] = [box_rect.left(), box_rect.top(), box_rect.right(), box_rect.bottom()]
            self._refresh_video_objects_list()

        try:
            self.video_display_label.video_points_changed.emit()
        except AttributeError:
            pass

        self._display_frame_wrapper(self.current_frame_index)

        if getattr(self, 'video_predictor', None):
            self._start_async_sam2_video_interaction(self.current_target_id, global_frame_idx, target_data)
            return

        self.update_button_states()

    def _start_async_sam2_video_interaction(self, target_id: int, frame_idx: int, target_data: dict):
        """
        [重构版]：视频打点/框选交互。
        直达模式下秒速调用静态 ImagePredictor 分割，避开耗时的视频特征初始编码。
        """
        has_points = bool(target_data and target_data.get('points'))
        has_box = bool(target_data and target_data.get('box') is not None)

        if not (has_points or has_box):
            return

        input_points_np, input_labels_np, box_np = None, None, None

        if has_points:
            input_points_np = np.array([(p[0], p[1]) for p in target_data['points']], dtype=np.float32)
            if input_points_np.ndim == 1 and input_points_np.size > 0:
                input_points_np = input_points_np[np.newaxis, :]
            input_labels_np = np.array([p[2] for p in target_data['points']], dtype=np.int32)
        if has_box:
            box_np = np.array(target_data['box'], dtype=np.float32)

        is_isolated = getattr(self, '_matting_clip_idx', -1) != -1
        local_ann_frame = frame_idx
        if is_isolated and frame_idx >= getattr(self, '_matting_global_start', 0):
            local_ann_frame = frame_idx - getattr(self, '_matting_global_start', 0)

        # =========================================================================
        # 【无需定位直达模式】：点击后秒速调用不占视频显存的 ImagePredictor 获取初始遮罩
        # =========================================================================
        use_direct_matting = hasattr(self,
                                     'vid_direct_matting_checkbox') and self.vid_direct_matting_checkbox.isChecked()

        if use_direct_matting:
            self.show_global_loading_overlay("正在分割目标区域，请稍候...", -1)

            img_cv = self._read_frame_from_disk(frame_idx)
            if img_cv is not None:
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                # 启动单帧快速 Predict 线程，跳过 Video 视频状态初始化
                worker = PredictWorker()
                self._start_worker(
                    "predict", worker, "run_predict",
                    self.image_predictor, (input_points_np, input_labels_np), box_np, False, img_rgb
                )

                self._pending_interaction_frame_idx = frame_idx
                self._pending_interaction_target_id = target_id
            return

        # =========================================================================
        # 【粗定位模式】：按需动态初始化并运行视频时序交互
        # =========================================================================
        self._pending_interaction_frame_idx = (
                    self._matting_global_start + local_ann_frame) if is_isolated else frame_idx
        self._pending_interaction_target_id = target_id

        if getattr(self, 'video_predictor', None):
            if getattr(self, 'video_inference_state', None) is None:
                def resume_interaction_after_init():
                    self._start_async_sam2_video_interaction(self.current_target_id, frame_idx, target_data)

                self._pending_interaction_resume = resume_interaction_after_init
                self._initialize_video_predictor_state()
                return

            self.show_global_loading_overlay("正在进行时序关联分割，请稍候...", -1)
            worker = VideoInteractionWorker()
            self._start_worker(
                "video_interact", worker, "run_interaction",
                self.video_predictor, self.video_inference_state,
                local_ann_frame, target_id,
                input_points_np, input_labels_np, box_np, True
            )

    @Slot(object, bool, str)
    def _handle_video_interaction_completion(self, result_masks, success, error_msg):
        """
        Receives interaction outcomes and triggers view repaints.
        """
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.hide()

        frame_idx = getattr(self, '_pending_interaction_frame_idx', self.current_frame_index)
        target_id = getattr(self, '_pending_interaction_target_id', getattr(self, 'current_target_id', -1))

        if success and result_masks is not None:
            self._is_matting_dirty = True
            if not hasattr(self.video_display_label, 'temp_multi_masks'):
                self.video_display_label.temp_multi_masks = {}
            if frame_idx not in self.video_display_label.temp_multi_masks:
                self.video_display_label.temp_multi_masks[frame_idx] = {}

            temp_mask_for_display = None
            for obj_id_val, mask in result_masks.items():
                self.video_display_label.temp_multi_masks[frame_idx][obj_id_val] = mask
                if obj_id_val == target_id:
                    temp_mask_for_display = mask

            if hasattr(self, 'video_display_label'):
                self.video_display_label.temp_annotation_frame_mask = temp_mask_for_display
                self.video_display_label.temp_annotation_target_id = target_id
                self.video_display_label.temp_annotation_mask_frame_idx = frame_idx if temp_mask_for_display is not None else -1

        elif not success:
            print(f"SAM2 video interaction inference failed: {error_msg}")

        self._pending_interaction_frame_idx = None
        self._pending_interaction_target_id = None

        self._display_frame_wrapper(self.current_frame_index)
        self.update_button_states()

    def _update_predictor_for_annotation_frame_preview(self, target_id_to_update: int, annotation_frame_idx: int):
        """
        Generates lightweight previews during interactive target changes without editing history contexts.
        """
        if not getattr(self, 'video_predictor', None) or not getattr(self, 'video_inference_state', None):
            return

        target_data = self.target_points.get(target_id_to_update)
        has_points = bool(target_data and target_data.get('points'))
        has_box = bool(target_data and target_data.get('box') is not None)

        if not target_data or not (has_points or has_box) or target_data.get(
                'annotation_frame') != annotation_frame_idx:
            self._display_frame_wrapper(self.current_frame_index)
            return

        input_points_np = None
        input_labels_np = None
        box_np = None

        if has_points:
            input_points_np = np.array([(p[0], p[1]) for p in target_data['points']], dtype=np.float32)
            if input_points_np.ndim == 1 and input_points_np.size > 0:
                input_points_np = input_points_np[np.newaxis, :]
            input_labels_np = np.array([p[2] for p in target_data['points']], dtype=np.int32)
        if has_box:
            box_np = np.array(target_data['box'], dtype=np.float32)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        temp_mask_for_display = None
        temp_id_for_display = -1

        try:
            local_ann_frame = annotation_frame_idx - getattr(self, '_matting_global_start', 0)

            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=self.video_inference_state,
                frame_idx=local_ann_frame,
                obj_id=target_id_to_update,
                points=input_points_np,
                labels=input_labels_np,
                box=box_np,
                clear_old_points=True
            )

            if out_mask_logits is not None and len(out_mask_logits) > 0 and out_obj_ids is not None:
                for i, obj_id_tensor in enumerate(out_obj_ids):
                    obj_id_val = int(obj_id_tensor.item()) if hasattr(obj_id_tensor, 'item') else int(obj_id_tensor)

                    mask = (out_mask_logits[i].float() > 0.0).cpu().numpy().squeeze()
                    if mask.dtype != bool:
                        mask = mask.astype(bool)
                    mask = np.ascontiguousarray(mask)

                    if obj_id_val == target_id_to_update:
                        temp_mask_for_display = mask
                        temp_id_for_display = target_id_to_update

                    if not hasattr(self.video_display_label, 'temp_multi_masks'):
                        self.video_display_label.temp_multi_masks = {}

                    if annotation_frame_idx not in self.video_display_label.temp_multi_masks:
                        self.video_display_label.temp_multi_masks[annotation_frame_idx] = {}
                    self.video_display_label.temp_multi_masks[annotation_frame_idx][obj_id_val] = mask

        except Exception:
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

            if hasattr(self, 'video_display_label'):
                self.video_display_label.temp_annotation_frame_mask = temp_mask_for_display
                self.video_display_label.temp_id_for_display = temp_id_for_display
                self.video_display_label.temp_annotation_mask_frame_idx = annotation_frame_idx if temp_mask_for_display is not None else -1

            self._display_frame_wrapper(self.current_frame_index)

    @Slot()
    def start_video_segmentation_propagation(self):
        """
        [重构版]：启动视频时序抠图传播
        主界面一键解耦，将所有的初始化、粗定位和精抠逻辑全部交给顺序流水线后台执行。
        """
        if getattr(self, 'video_segmentation_running', False):
            return
        if not getattr(self, 'temp_frame_dir', None) or getattr(self, 'total_frames', 0) == 0:
            return

        clip_idx = getattr(self, '_matting_clip_idx', -1)
        if clip_idx == -1: return

        active_clip = self.virtual_timeline[clip_idx]
        fps = active_clip.get('fps', 30.0)
        duration_sec = active_clip['frames'] / max(1.0, fps)

        # 严格限制单次抠图物理时长不能超过 10 秒
        if duration_sec > 10.0:
            QMessageBox.warning(
                self, _TR("时长超限保护"),
                _TR(f"当前片段时长为 {duration_sec:.1f} 秒。\n\n为了保证系统显存绝对安全和抠图精度，单次抠图最多不能超过 10 秒。\n请先使用【剪裁】功能将其修剪至 10 秒以内！")
            )
            return

        global_start_frame = sum(v['frames'] for v in self.virtual_timeline[:clip_idx])
        global_end_frame = global_start_frame + active_clip['frames'] - 1

        use_matanyone = hasattr(self, 'vid_matteformer_checkbox') and self.vid_matteformer_checkbox.isChecked()

        valid_targets_for_propagation = {}
        for tid, data in self.target_points.items():
            ann_frame = data.get('annotation_frame')
            if ann_frame is None or not (global_start_frame <= ann_frame <= global_end_frame):
                continue

            tdata = copy.deepcopy(data)
            initial_mask = None

            if ann_frame in getattr(self, 'processed_masks', {}) and tid in self.processed_masks[ann_frame]:
                if self.processed_masks[ann_frame][tid] is not None:
                    initial_mask = self.processed_masks[ann_frame][tid].copy()

            if hasattr(self, 'video_display_label') and self.video_display_label:
                temp_multi = getattr(self.video_display_label, 'temp_multi_masks', {})
                if ann_frame in temp_multi and tid in temp_multi[ann_frame]:
                    if temp_multi[ann_frame][tid] is not None:
                        initial_mask = temp_multi[ann_frame][tid].copy()

                temp_idx = getattr(self.video_display_label, 'temp_annotation_mask_frame_idx', -1)
                temp_mask = getattr(self.video_display_label, 'temp_annotation_frame_mask', None)
                temp_obj_id = getattr(self.video_display_label, 'temp_annotation_target_id', -1)
                if temp_idx == ann_frame and temp_obj_id == tid and temp_mask is not None:
                    initial_mask = temp_mask.copy()

            if initial_mask is not None:
                tdata['initial_mask'] = initial_mask

            if 'initial_mask' in tdata or bool(tdata.get('points')) or tdata.get('box') is not None:
                valid_targets_for_propagation[tid] = tdata

        if not valid_targets_for_propagation:
            QMessageBox.warning(self, "无提示信息", "请在【当前选中的视频片段】内，添加提示点、框或画笔蒙版。")
            return

        reply = QMessageBox.question(
            self, "确认开始", "将对当前选中的片段进行智能渲染处理。\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Yes
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return

        self.video_segmentation_running = True
        self.video_segmentation_finished = False

        # 清理待重写帧的旧掩码
        for f_idx in range(global_start_frame, global_end_frame + 1):
            is_ann_frame = any(d.get('annotation_frame') == f_idx for d in valid_targets_for_propagation.values())
            if not is_ann_frame and f_idx in getattr(self, 'processed_masks', {}):
                del self.processed_masks[f_idx]

        self.video_segmentation_saved = False
        if getattr(self, 'is_playing', False):
            self.pause_video()

        if hasattr(self, 'video_display_label') and hasattr(self.video_display_label, 'temp_annotation_target_id'):
            self.video_display_label.temp_annotation_frame_mask = None
            self.video_display_label.temp_annotation_target_id = -1

        self._clear_video_result_preview()
        self.update_button_states()

        # =========================================================================
        # 【双重保险】：在启动传播前，彻底注销并重置异步预读缓存池，确保时序渲染更新视口时读取的是最新帧
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

        if hasattr(self, '_propagation_heartbeat_timer') and self._propagation_heartbeat_timer:
            try:
                self._propagation_heartbeat_timer.stop()
                self._propagation_heartbeat_timer.deleteLater()
            except Exception:
                pass

        self._propagation_heartbeat_timer = QTimer(self)
        self._propagation_heartbeat_timer.setInterval(30)
        self._propagation_heartbeat_timer.timeout.connect(self._pump_gui_heartbeat)
        self._propagation_heartbeat_timer.start()

        self.show_global_loading_overlay("正在初始化智能视频抠图传播引擎...", 0)

        erode_val = self.vid_erode_slider.value() if hasattr(self, 'vid_erode_slider') else 10
        dilate_val = self.vid_dilate_slider.value() if hasattr(self, 'vid_dilate_slider') else 10

        worker = VideoMatAnyoneWorker()

        # 分离 temp_frame_dir 和 clip_sandbox_dir 目录，执行安全的流式时序求解
        thread, active_worker = self._start_worker(
            "propagate_video_v1991", worker, "run_sam2_guided_matanyone_propagation",
            self.video_predictor, None, self.matteformer_model,
            self.temp_frame_dir, getattr(self, 'clip_sandbox_dir', self.temp_frame_dir), valid_targets_for_propagation,
            global_start_frame, global_end_frame,
            erode_val, dilate_val, use_matanyone
        )

        if active_worker and hasattr(active_worker, 'frame_updated'):
            active_worker.frame_updated.connect(self._display_frame_wrapper)

    @Slot(bool, str)
    def _handle_pre_render_completion(self, success, result_msg):
        """
        [该函数已废弃且被内联替代，仅保留桩防止其他位置报错]
        """
        pass

    @Slot(object, bool, str)
    def _handle_video_propagation_v1991_completion(self, processed_masks_dict, success, error_message):
        """
        重构版时序抠图完成回调：
        【核心修复】：将自动抠图生成的融合帧正确归档到 `clip['baked_preview_dir']`，
        使得点击“确定”退出时，极速链结器能正确找到带背景的烘焙帧！
        """
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        # 移除原传播工作线程
        self._remove_active_worker("propagate_video_v1991")
        if hasattr(self, '_propagation_heartbeat_timer') and self._propagation_heartbeat_timer:
            self._propagation_heartbeat_timer.stop()

        if success and processed_masks_dict:
            if not hasattr(self, 'processed_masks'):
                self.processed_masks = {}
            self.processed_masks.update(processed_masks_dict)

            self._distribute_global_masks_to_clips()
            self._gather_global_masks_from_clips()

            clip_idx = getattr(self, '_matting_clip_idx', -1)
            if clip_idx == -1: return

            clip = self.virtual_timeline[clip_idx]
            local_baked_dir = os.path.join(TEMP_BASE_DIR, f"baked_clip_{uuid.uuid4().hex[:8]}")

            self.show_global_loading_overlay("抠图特征关联成功，正在烘焙并构建高速播放音轨...", 0)

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

            # 使用闭包直接处理完成逻辑，取代原来的 _handle_pre_render_completion
            def on_auto_matting_bake_complete(ok, result_msg):
                self._single_bake_thread.quit()
                self._single_bake_thread.wait()
                self._single_bake_thread.deleteLater()
                self._single_bake_worker.deleteLater()

                if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                    self._global_loading_overlay.hide()

                if ok:
                    # 【关键修复】：将产出的文件正确归档到片段缓存中！
                    old_baked_dir = clip.get('baked_preview_dir', None)
                    clip['baked_preview_dir'] = result_msg
                    if old_baked_dir and os.path.exists(old_baked_dir):
                        try:
                            import shutil
                            shutil.rmtree(old_baked_dir, ignore_errors=True)
                        except Exception:
                            pass

                    # 极速重组时间线
                    self._assemble_global_render_dir()

                    # 刷新故事板封面为新抠出的带背景图片
                    first_baked_frame_path = os.path.join(result_msg, "00000.jpg")
                    if os.path.exists(first_baked_frame_path):
                        baked_cv = cv2.imread(first_baked_frame_path)
                        if baked_cv is not None:
                            pixmap = convert_cv_to_pixmap(baked_cv)
                            if pixmap and not pixmap.isNull():
                                clip['pixmap'] = pixmap
                                if hasattr(self, 'storyboard_list'):
                                    item = self.storyboard_list.item(clip_idx)
                                    if item:
                                        widget = self.storyboard_list.itemWidget(item)
                                        if hasattr(widget, 'bg_label'):
                                            scaled_pix = pixmap.scaled(240, 135, Qt.AspectRatioMode.KeepAspectRatio,
                                                                       Qt.TransformationMode.SmoothTransformation)
                                            widget.bg_label.setPixmap(scaled_pix)

                    # 重置脏标记，因为刚自动抠完且已烘焙完
                    self._is_matting_dirty = False
                    self.video_segmentation_finished = True
                    self.video_segmentation_saved = False

                    if hasattr(self, 'video_display_label') and self.video_display_label:
                        self.video_display_label.temp_annotation_frame_mask = None
                        self.video_display_label.temp_annotation_target_id = -1
                        self.video_display_label.temp_annotation_mask_frame_idx = -1
                        if hasattr(self.video_display_label, 'temp_multi_masks'):
                            self.video_display_label.temp_multi_masks.clear()

                        # 自动开启抠图沙盒对比模式滑块
                        is_compare_active = True
                        if hasattr(self, 'video_compare_mode_button') and self.video_compare_mode_button:
                            self.video_compare_mode_button.blockSignals(True)
                            self.video_compare_mode_button.setChecked(True)
                            self.video_compare_mode_button.blockSignals(False)
                        self.video_display_label.set_compare_mode(is_compare_active)

                    self.show_status_message("智能抠图轨道烘焙完成。", 3000)
                    if hasattr(self, '_video_frame_cache'):
                        self._video_frame_cache.clear()
                    self._display_frame_wrapper(self.current_frame_index)
                else:
                    self.video_segmentation_finished = False
                    QMessageBox.warning(self, "渲染失败", f"生成播放轨道失败: {result_msg}")

                self.update_button_states()

            self._single_bake_worker.finished.connect(on_auto_matting_bake_complete)
            self._single_bake_thread.started.connect(self._single_bake_worker.run)
            self._single_bake_thread.start()

        else:
            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()
            self.video_segmentation_finished = False
            if error_message and not ("cancel" in error_message.lower() or "取消" in error_message):
                QMessageBox.warning(self, "操作失败", f"视频抠图特征同步失败: {error_message}")
            self.update_button_states()

    def update_video_preview_all_targets(self, frame_index: int, cv_image: Optional[np.ndarray] = None):
        """Updates preview layers based on currently accumulated mask tracks."""
        preview_label = getattr(self, 'video_result_preview_label', None)
        if not preview_label:
            return

        if cv_image is None:
            cv_image = self._read_frame_from_disk(frame_index)

        if cv_image is None:
            self._clear_video_result_preview()
            return

        try:
            h, w = cv_image.shape[:2]
            bgr_frame = cv_image

            if len(cv_image.shape) == 2:
                bgr_frame = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
                bgr_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)

            final_preview_bgr = self._get_current_bg_frame(h, w)
            if final_preview_bgr is None:
                self._clear_video_result_preview()
                return

            # 改用高精度连续浮点蒙版
            combined_mask_for_frame = np.zeros((h, w), dtype=np.float32)
            has_mask = False

            def align_mask_to_canvas(mask_in):
                if mask_in.shape[:2] == (h, w):
                    return mask_in
                mh, mw = mask_in.shape[:2]
                m_scale = min(w / mw, h / mh)
                new_mw, new_mh = int(mw * m_scale), int(mh * m_scale)
                resized_m = cv2.resize(mask_in.astype(np.float32), (new_mw, new_mh),
                                       interpolation=cv2.INTER_LINEAR)
                canvas_m = np.zeros((h, w), dtype=np.float32)
                mx_off = (w - new_mw) // 2
                my_off = (h - new_mh) // 2
                canvas_m[my_off:my_off + new_mh, mx_off:mx_off + new_mw] = resized_m
                return canvas_m

            frame_masks_for_preview = self.processed_masks.get(frame_index, {})
            for obj_id, mask_np in frame_masks_for_preview.items():
                if mask_np is not None:
                    if mask_np.dtype == bool:
                        mask_float = mask_np.astype(np.float32)
                    else:
                        mask_float = np.clip(mask_np.astype(np.float32), 0.0, 1.0)
                    combined_mask_for_frame = np.maximum(combined_mask_for_frame, align_mask_to_canvas(mask_float))
                    has_mask = True

            if hasattr(self, 'video_display_label'):
                temp_multi = getattr(self.video_display_label, 'temp_multi_masks', {}).get(frame_index, {})
                for temp_np in temp_multi.values():
                    if temp_np is not None:
                        if temp_np.dtype == bool:
                            temp_float = temp_np.astype(np.float32)
                        else:
                            temp_float = np.clip(temp_np.astype(np.float32), 0.0, 1.0)
                        combined_mask_for_frame = np.maximum(combined_mask_for_frame, align_mask_to_canvas(temp_float))
                        has_mask = True

                temp_idx = getattr(self.video_display_label, 'temp_annotation_mask_frame_idx', -1)
                temp_mask = getattr(self.video_display_label, 'temp_annotation_frame_mask', None)
                if temp_idx == frame_index and temp_mask is not None:
                    if temp_mask.dtype == bool:
                        temp_float = temp_mask.astype(np.float32)
                    else:
                        temp_float = np.clip(temp_mask.astype(np.float32), 0.0, 1.0)
                    combined_mask_for_frame = np.maximum(combined_mask_for_frame, align_mask_to_canvas(temp_float))
                    has_mask = True

            if has_mask:
                alpha_3d = combined_mask_for_frame[:, :, np.newaxis]
                blended = bgr_frame.astype(np.float32) * alpha_3d + final_preview_bgr.astype(np.float32) * (1.0 - alpha_3d)
                final_preview_bgr = np.clip(blended, 0.0, 255.0).astype(np.uint8)
            else:
                final_preview_bgr = bgr_frame.copy()

            final_preview_bgra = cv2.cvtColor(final_preview_bgr, cv2.COLOR_BGR2BGRA)
            result_pixmap = convert_cv_to_pixmap(final_preview_bgra)

            if result_pixmap and not result_pixmap.isNull():
                target_w = preview_label.width()
                if target_w < 100:
                    target_w = 280
                target_h = int(target_w * 9 / 16)

                preview_label.setMinimumHeight(target_h)
                preview_label.setMaximumHeight(target_h)

                scaled_result = result_pixmap.scaled(
                    target_w, target_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )

                preview_label.setPixmap(scaled_result)
                preview_label.repaint()
            else:
                self._clear_video_result_preview()

        except Exception:
            traceback.print_exc()
            self._clear_video_result_preview()

    @Slot()
    def cancel_current_video_operation(self):
        """Cancels any video processing threads currently active in the background context."""
        worker_to_cancel = None
        task_name = ""

        if self.is_extracting_frames and "extract" in self.active_workers:
            worker_to_cancel = "extract"
            task_name = "帧提取"
        elif self.video_segmentation_running and "propagate_video_v1991" in self.active_workers:
            worker_to_cancel = "propagate_video_v1991"
            task_name = "视频抠图"
        elif getattr(self, 'is_saving', False) and "save_video" in self.active_workers:
            worker_to_cancel = "save_video"
            task_name = "视频保存"
        elif getattr(self, 'is_loading_model', False) and "load_video_model" in self.active_workers:
            worker_to_cancel = "load_video_model"
            task_name = "视频模型加载"

        if worker_to_cancel:
            self.log_message.emit(f"用户请求取消: {task_name} (工作线程 ID: {worker_to_cancel})")
            self._cancel_worker(worker_to_cancel)
            if hasattr(self, 'video_info_label_display'):
                current_text = self.video_info_label_display.text().split('\n')[0]
                self.video_info_label_display.setText(f"{current_text}\n状态: 取消中...")
            self.update_button_states()
        else:
            self.show_status_message("没有正在进行的视频操作可取消。", 3000)

    @Slot()
    def save_video_segmentation_result(self):
        """视频导出触发核心中枢：允许随时导出（支持纯剪辑、换配音、或者半成品抠图导出）"""
        if not getattr(self, 'video_path', None) or getattr(self, 'total_frames', 0) <= 0:
            QMessageBox.warning(self, _TR("无法导出"), _TR("请先加载视频。"))
            return

        # 仅保留底层文件占用的安全保护：防止正在后台抽帧或跑AI大模型时强行读取导致崩溃
        if getattr(self, 'is_extracting_frames', False) or getattr(self, 'video_segmentation_running', False):
            QMessageBox.warning(self, _TR("无法导出"), _TR("后台正在处理视频序列数据，请等待当前进度条完成后再导出。"))
            return

        if self.is_saving or "save_video" in getattr(self, 'active_workers', {}):
            QMessageBox.information(self, _TR("忙碌"), _TR("其他保存操作正在进行中。"))
            return

        # 1. 准备对话框初始参数
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        default_name = f"{base_name}_output"
        orig_size = QSize(getattr(self, 'video_width', 1280), getattr(self, 'video_height', 720))
        fps = getattr(self, 'video_fps', VIDEO_DEFAULT_FPS)

        # 检测整个故事板是否有要求透明通道的分段
        has_transparent_bg = False
        if hasattr(self, 'virtual_timeline'):
            has_transparent_bg = any(clip.get('bg_is_transparent', False) for clip in self.virtual_timeline)

        # 2. 呼出导出设置对话框
        dialog = ModernExportDialog('video', orig_size, default_name, default_fps=fps,
                                    has_transparent_bg=has_transparent_bg, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # 3. 获取用户配置
        params = dialog.get_export_params()
        save_path = params['path']

        # 4. 同名覆盖安全校验
        if os.path.exists(save_path):
            reply = QMessageBox.question(
                self, _TR("确认覆盖"),
                _TR("文件 '{}' 已存在。\n\n您确定要覆盖此文件吗？").format(os.path.basename(save_path)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # 5. 锁定UI并展示遮罩
        self.is_saving = True
        self.update_button_states()
        if getattr(self, 'is_playing', False):
            self.pause_video()

        self.show_global_loading_overlay(_TR("正在提取并校验导出参数配置..."), 0)

        # 6. 打包渲染所需参数
        save_data = (
            self.temp_frame_dir,
            getattr(self, 'processed_masks', {}),
            self.total_frames,
            params['size'].width(),
            params['size'].height(),
            params['fps'],
            getattr(self, 'video_save_bg_color', QColor(0, 255, 0)),
            getattr(self, 'video_save_bg_image_path', None),
            getattr(self, 'video_save_bg_is_transparent', False),
            getattr(self, 'virtual_timeline', [])
        )

        options = {
            'output_format': params['format'],
            'codec_info': params.get('codec_info', {})
        }

        # 7. 投入后台工作线程执行
        worker = SaveWorker()
        self._start_worker("save_video", worker, "run_save", "segmented_video", save_data, save_path, options)

    def _save_segmented_video(self, save_data: tuple, save_path: str, options: Optional[dict]):
        if options and options.get('output_format', '').lower() == 'gif':
            self._save_video_as_gif(save_data, save_path)
            return

        import os, uuid, tempfile, cv2, numpy as np, av, traceback, gc
        from fractions import Fraction

        (
            temp_frame_dir, processed_masks, total_frames,
            target_w, target_h, actual_save_fps,
            _, _, _,
            virtual_timeline
        ) = save_data

        is_mov = save_path.lower().endswith('.mov')
        has_any_transparent_segment = any(clip.get('bg_is_transparent', False) for clip in virtual_timeline)

        # 解析来自UI传递的用户选择的编码器配置
        codec_info = options.get('codec_info', {}) if options else {}
        export_transparent = has_any_transparent_segment and codec_info.get('alpha', False)

        ext_tmp = ".mov" if export_transparent else ".mp4"
        temp_silent_video = os.path.join(tempfile.gettempdir(), f"temp_silent_{uuid.uuid4().hex[:8]}{ext_tmp}")

        try:
            self.progress.emit(0, "启动 PyAV 视频编码器...")

            container = av.open(temp_silent_video, mode='w')
            safe_rate = Fraction(actual_save_fps).limit_denominator(1000)

            # 获取具体的编码器名称和像素格式，默认为 libx264
            encoder_name = codec_info.get('encoder', 'libx264')
            pix_fmt = codec_info.get('pix_fmt', 'yuv420p')

            # 添加视频流并应用所选编码配置
            stream = container.add_stream(encoder_name, rate=safe_rate)
            stream.pix_fmt = pix_fmt

            # 配置针对不同编码器的压制参数
            if encoder_name == 'prores_ks':
                stream.options = {
                    'profile': codec_info.get('profile', '4'),
                    'vendor': 'apl0',
                    'qscale': '9'
                }
            elif encoder_name == 'libx265':
                # H.265 / HEVC 设置
                stream.options = {'crf': '23', 'preset': 'medium'}
                # 【核心修复】：彻底删除了 'x265-params': 'alpha=1'，因为预编译库不支持
            elif 'av1' in encoder_name.lower():
                # AV1 (SVT-AV1 或 AOM) 专属平衡设置
                stream.options = {'preset': '5', 'crf': '30'}
            else:
                # 默认 H.264
                stream.options = {'preset': 'ultrafast', 'crf': '18', 'threads': 'auto'}

            stream.width = target_w
            stream.height = target_h

            # 1. 映射分段配置并预缓存背景
            clip_configs = []
            current_count = 0
            for clip in virtual_timeline:
                clip_configs.append({
                    'start': current_count,
                    'end': current_count + clip['frames'] - 1,
                    'bg_is_transparent': clip.get('bg_is_transparent', False),
                    'bg_image_path': clip.get('bg_image_path', None),
                    'bg_color': clip.get('bg_color', QColor(0, 255, 0))
                })
                current_count += clip['frames']

            bg_canvas_cache = {}
            for idx, cfg in enumerate(clip_configs):
                path = cfg['bg_image_path']
                if not cfg['bg_is_transparent'] and path and os.path.exists(path):
                    bg_img = cv2.imread(path)
                    if bg_img is not None:
                        bg_canvas_cache[idx] = cv2.resize(bg_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # 2. 逐帧读取并与对应的背景配置进行融合
            for i in range(total_frames):
                if self.is_cancelled:
                    container.close()
                    if os.path.exists(temp_silent_video): os.remove(temp_silent_video)
                    self.finished.emit(False, "", "用户取消了视频导出。")
                    return

                frame_path = os.path.join(temp_frame_dir, f"{i:05d}.jpg")
                if not os.path.exists(frame_path):
                    frame_path = os.path.join(temp_frame_dir, f"{i}.jpg")

                frame_cv = cv2.imread(frame_path)
                if frame_cv is None:
                    frame_cv = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                if frame_cv.shape[:2] != (target_h, target_w):
                    frame_cv = cv2.resize(frame_cv, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                frame_masks = processed_masks.get(i, {})
                h, w = frame_cv.shape[:2]

                combined_alpha = np.zeros((h, w), dtype=np.float32)
                has_mask = False

                for mask_raw in frame_masks.values():
                    if mask_raw is not None:
                        if mask_raw.dtype == bool:
                            mask_float = mask_raw.astype(np.float32)
                        else:
                            mask_float = np.clip(mask_raw.astype(np.float32), 0.0, 1.0)

                        if mask_float.shape != (h, w):
                            mask_float = cv2.resize(mask_float, (w, h), interpolation=cv2.INTER_LINEAR)

                        combined_alpha = np.maximum(combined_alpha, mask_float)
                        has_mask = True

                # 查询当前帧匹配的背景配置
                cfg_idx = -1
                active_cfg = None
                for idx, cfg in enumerate(clip_configs):
                    if cfg['start'] <= i <= cfg['end']:
                        active_cfg = cfg
                        cfg_idx = idx
                        break
                if active_cfg is None:
                    active_cfg = {'bg_is_transparent': False, 'bg_image_path': None, 'bg_color': QColor(0, 255, 0)}

                bg_is_transparent = active_cfg['bg_is_transparent']
                bg_color = active_cfg['bg_color']

                # ==============================================================
                # 生成发给 PyAV 的最终画面数据 (区分是否需要带透明通道 RGBA vs RGB)
                # ==============================================================
                if export_transparent and bg_is_transparent:
                    # 分支 A：支持透明通道 (目前专供 ProRes 4444)
                    if has_mask:
                        alpha_channel = (combined_alpha * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        alpha_channel = np.zeros((h, w), dtype=np.uint8)

                    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                    frame_rgba = np.dstack((frame_rgb, alpha_channel))
                    av_frame = av.VideoFrame.from_ndarray(frame_rgba, format='rgba')
                else:
                    # 分支 B：不支持透明通道，或者该分段用户设置了纯色/图片背景
                    if has_mask:
                        if bg_is_transparent:
                            # 警告情况：要求透明但编码不支持，此时默认回退为绿幕
                            bg_canvas = np.full((h, w, 3), (0, 255, 0), dtype=np.uint8)
                        elif cfg_idx in bg_canvas_cache:
                            bg_canvas = bg_canvas_cache[cfg_idx].copy()
                        else:
                            bg_bgr = (bg_color.blue(), bg_color.green(), bg_color.red())
                            bg_canvas = np.full((h, w, 3), bg_bgr, dtype=np.uint8)

                        alpha_3d = combined_alpha[:, :, np.newaxis]
                        blended_frame = frame_cv.astype(np.float32) * alpha_3d + bg_canvas.astype(np.float32) * (1.0 - alpha_3d)
                        frame_cv = np.clip(blended_frame, 0.0, 255.0).astype(np.uint8)

                    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                    av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')

                av_frame.pts = i

                for packet in stream.encode(av_frame):
                    container.mux(packet)

                if i % 10 == 0:
                    self.progress.emit(int((i / total_frames) * 75),
                                       f"{encoder_name.upper()} 视频合成编码中... ({i}/{total_frames})")

            for packet in stream.encode():
                container.mux(packet)
            container.close()

            self.progress.emit(78, "构建高保真音轨...")
            success, err_msg = self._merge_audio_with_pydub(temp_silent_video, virtual_timeline, save_path)

            if os.path.exists(temp_silent_video):
                os.remove(temp_silent_video)

            if success:
                self.progress.emit(100, "视频导出成功！")
                self.finished.emit(True, save_path, "")
            else:
                self.finished.emit(False, "", f"多音轨混合失败: {err_msg}")

        except Exception as e:
            if 'temp_silent_video' in locals() and os.path.exists(temp_silent_video):
                try:
                    os.remove(temp_silent_video)
                except: pass
            traceback.print_exc()
            self.finished.emit(False, "", f"视频编码发生崩溃: {e}")
        finally:
            gc.collect()

    @Slot(bool, str, str)
    def _handle_save_video_completion(self, success, saved_path, error_message):
        # 【核心修复】：导出完成后，强制隐藏全局“系统处理中”弹窗遮罩
        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.hide()

        display_frame_idx = self.current_frame_index if self.current_frame_index != -1 and self.total_frames > 0 else 0
        if self.total_frames > 0:
            self._display_frame_wrapper(display_frame_idx)
        else:
            if hasattr(self, 'video_display_label'):
                self.video_display_label.clear_display()

        if success:
            QMessageBox.information(self, "保存成功", f"文件已成功保存至:\n{saved_path}")
            self.video_segmentation_saved = True
        else:
            QMessageBox.warning(self, "保存失败", f"无法保存文件:\n{error_message}")

        self.is_saving = False  # 确保状态重置，解锁界面按钮
        self.update_button_states()

    def load_all_assets(self):
        """Loads default presets and scans configuration lists to fetch custom files."""
        if not hasattr(self, 'stitch_asset_grid_layout') or self.stitch_asset_grid_layout is None:
            self.log_message.emit("Warning: load_all_assets called, but layout uninitialized.")
            return
        self.load_preset_assets()
        self.load_user_assets_from_config()

    def load_preset_assets(self):
        if not hasattr(self, 'stitch_asset_grid_layout') or self.stitch_asset_grid_layout is None:
            return

        while self.stitch_asset_grid_layout.count():
            child = self.stitch_asset_grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        preset_asset_dir_to_load = get_asset_path("")

        if not os.path.exists(preset_asset_dir_to_load):
            info_label = QLabel("未找到\n'assets' 文件夹")
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.stitch_asset_grid_layout.addWidget(info_label, 0, 0, 1, 2)
            return

        supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
        try:
            asset_files = [f for f in os.listdir(preset_asset_dir_to_load) if f.lower().endswith(supported_formats)]

            if not asset_files:
                info_label = QLabel("'assets' 文件夹\n中没有支持的图像")
                info_label.setWordWrap(True)
                info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.stitch_asset_grid_layout.addWidget(info_label, 0, 0, 1, 2)
                return

            for filename in sorted(asset_files):
                filepath = os.path.join(preset_asset_dir_to_load, filename)
                self.add_thumbnail_to_grid(filepath, target_grid=self.stitch_asset_grid_layout)
        except Exception as e:
            self.log_message.emit(f"Error loading preset assets: {e}")

    def add_user_assets(self):
        supported_formats = "图像文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"
        files, _ = QFileDialog.getOpenFileNames(self, "选择要添加的素材 (可多选)", "", supported_formats)

        if files:
            if not hasattr(self, 'stitch_asset_grid_layout') or self.stitch_asset_grid_layout is None:
                QMessageBox.critical(self, "错误", "素材库未正确初始化，无法添加素材。")
                return

            current_grid_layout = self.stitch_asset_grid_layout

            for file_path in files:
                is_duplicate = False
                for i in range(current_grid_layout.count()):
                    widget = current_grid_layout.itemAt(i).widget()
                    if isinstance(widget, AssetThumbnail) and os.path.normpath(widget.image_path) == os.path.normpath(
                            file_path):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    self.add_thumbnail_to_grid(file_path, target_grid=current_grid_layout)
                else:
                    self.log_message.emit(f"Asset '{os.path.basename(file_path)}' already exists, skipping.")

    def add_thumbnail_to_grid(self, image_path: str, target_grid: QGridLayout = None):
        if target_grid is None:
            target_grid = getattr(self, 'stitch_asset_grid_layout', None)
            if target_grid is None:
                return

        thumbnail = AssetThumbnail(image_path, self)
        thumbnail.delete_requested.connect(self.delete_user_asset)

        cols = 2
        total_items = target_grid.count()

        if total_items == 1:
            current_item = target_grid.itemAt(0)
            if current_item and isinstance(current_item.widget(),
                                           QLabel) and "素材库为空" in current_item.widget().text():
                target_grid.removeWidget(current_item.widget())
                current_item.widget().deleteLater()
                total_items = 0

        row = total_items // cols
        col = total_items % cols
        target_grid.addWidget(thumbnail, row, col)

    def delete_user_asset(self, image_path_to_delete: str):
        if self._is_preset_asset(image_path_to_delete):
            QMessageBox.information(self, "操作无效", "这是程序自带的预设素材，无法在此处删除。")
            return

        if not hasattr(self, 'stitch_asset_grid_layout') or self.stitch_asset_grid_layout is None:
            return

        current_grid_layout = self.stitch_asset_grid_layout
        widget_to_remove = None

        for i in range(current_grid_layout.count()):
            item = current_grid_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, AssetThumbnail) and os.path.normpath(widget.image_path) == os.path.normpath(
                        image_path_to_delete):
                    widget_to_remove = widget
                    break

        if widget_to_remove:
            current_grid_layout.removeWidget(widget_to_remove)
            widget_to_remove.deleteLater()
            self.reorganize_asset_grid(target_grid=current_grid_layout)

        if os.path.exists(self.user_assets_config_path):
            try:
                with open(self.user_assets_config_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                new_lines = [line for line in lines if
                             os.path.normpath(line.strip()) != os.path.normpath(image_path_to_delete)]

                if len(new_lines) < len(lines):
                    with open(self.user_assets_config_path, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)

            except Exception as e:
                self.log_message.emit(f"Error removing from user asset config: {e}")

    def reorganize_asset_grid(self, target_grid: QGridLayout = None):
        if target_grid is None:
            target_grid = getattr(self, 'stitch_asset_grid_layout', None)
            if target_grid is None:
                return

        items_to_readd = []
        while target_grid.count():
            item = target_grid.takeAt(0)
            if item and item.widget():
                items_to_readd.append(item.widget())
            elif item:
                del item

        cols = 2

        for i, widget in enumerate(items_to_readd):
            row = i // cols
            col = i % cols
            target_grid.addWidget(widget, row, col)

        if not items_to_readd:
            info_label = QLabel("素材库为空。\n点击下方“添加素材”按钮\n或确保 'assets' 文件夹存在。")
            info_label.setWordWrap(True)
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            target_grid.addWidget(info_label, 0, 0, 1, cols)

    @Slot(int, int)
    def set_stitch_canvas_preset_size(self, width, height):
        if hasattr(self, 'stitch_canvas_width_spin') and hasattr(self, 'stitch_canvas_height_spin'):
            self.stitch_canvas_width_spin.blockSignals(True)
            self.stitch_canvas_height_spin.blockSignals(True)

            self.stitch_canvas_width_spin.setValue(width)
            self.stitch_canvas_height_spin.setValue(height)

            self.stitch_canvas_width_spin.blockSignals(False)
            self.stitch_canvas_height_spin.blockSignals(False)

            if hasattr(self, 'stitching_canvas'):
                self.stitching_canvas.set_canvas_size(width, height)

    def _find_button_by_panel_key(self, panel_key: str) -> Optional[QToolButton]:
        for group in self.page_top_button_groups.values():
            for button in group.buttons():
                if button.property("panel_key") == panel_key:
                    return button
        return None

    def _create_layer_tool_button(self, char_icon, tooltip):
        btn = QToolButton()
        btn.setText(char_icon)
        btn.setToolTip(tooltip)
        btn.setObjectName("LayerControlButton")
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return btn

    @Slot()
    def select_stitch_bg_color(self):
        if not hasattr(self, 'stitch_solid_bg_color'):
            self.stitch_solid_bg_color = QColor(Qt.GlobalColor.white)

        current_color = self.stitch_solid_bg_color
        new_color = ModernColorDialog.getColor(current_color, self, "选择画布背景颜色")

        if new_color.isValid():
            self.stitch_solid_bg_color = new_color
            self._update_stitch_bg_color_button_style(new_color)

            if not self.stitch_bg_color_radio.isChecked():
                self.stitch_bg_color_radio.setChecked(True)
            else:
                self._on_stitch_bg_mode_changed()

    def _update_stitch_bg_color_button_style(self, color):
        if hasattr(self, 'stitch_bg_color_button'):
            self.stitch_bg_color_button.setStyleSheet(f"background-color: {color.name()};")

    @Slot()
    def on_stitch_list_selection_changed(self):
        selected_list_items = self.stitch_layers_list.selectedItems()
        selected_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected_list_items]
        self.stitching_canvas.select_items_by_ids(selected_ids)

    @Slot()
    def on_stitch_selection_changed(self):
        if not hasattr(self, 'stitching_canvas') or not self.stitching_canvas:
            return

        selected_items = self.stitching_canvas.selected_items
        primary_item = self.stitching_canvas.get_primary_selected_item()
        is_single_selection = len(selected_items) == 1

        if hasattr(self, 'stitch_props_widget'):
            self.stitch_props_widget.setEnabled(is_single_selection)

            if is_single_selection and primary_item:
                updates = {
                    'stitch_pos_x_spin': primary_item.pos.x(),
                    'stitch_pos_y_spin': primary_item.pos.y(),
                    'stitch_size_w_spin': primary_item.size.width(),
                    'stitch_size_h_spin': primary_item.size.height(),
                    'stitch_rotation_spin': primary_item.rotation
                }

                for widget_name, val in updates.items():
                    widget = getattr(self, widget_name, None)
                    if widget:
                        widget.blockSignals(True)
                        widget.setValue(val)
                        widget.blockSignals(False)

        if hasattr(self, 'stitch_layers_list'):
            self.stitch_layers_list.blockSignals(True)
            self.stitch_layers_list.clearSelection()
            selected_ids = {item.id for item in selected_items}
            for i in range(self.stitch_layers_list.count()):
                list_item = self.stitch_layers_list.item(i)
                if list_item.data(Qt.ItemDataRole.UserRole) in selected_ids:
                    list_item.setSelected(True)
            self.stitch_layers_list.blockSignals(False)

        self.update_button_states()

    def update_selected_item_property(self, prop, value):
        if not hasattr(self, 'stitching_canvas') or not self.stitching_canvas:
            return
        primary_item = self.stitching_canvas.get_primary_selected_item()

        if primary_item and not primary_item.pixmap.isNull():
            w_spin = getattr(self, 'stitch_size_w_spin', None)
            h_spin = getattr(self, 'stitch_size_h_spin', None)

            if prop == 'x':
                primary_item.pos.setX(value)
            elif prop == 'y':
                primary_item.pos.setY(value)
            elif prop == 'width':
                aspect_ratio = primary_item.pixmap.height() / primary_item.pixmap.width() if primary_item.pixmap.width() > 0 else 1
                primary_item.size.setWidth(value)
                new_height = value * aspect_ratio
                primary_item.size.setHeight(new_height)
                if h_spin:
                    h_spin.blockSignals(True)
                    h_spin.setValue(new_height)
                    h_spin.blockSignals(False)
            elif prop == 'height':
                aspect_ratio = primary_item.pixmap.width() / primary_item.pixmap.height() if primary_item.pixmap.height() > 0 else 1
                primary_item.size.setHeight(value)
                new_width = value * aspect_ratio
                primary_item.size.setWidth(new_width)
                if w_spin:
                    w_spin.blockSignals(True)
                    w_spin.setValue(new_width)
                    w_spin.blockSignals(False)
            elif prop == 'rotation':
                primary_item.rotation = value

            self.stitching_canvas.update()

    @Slot()
    def update_stitch_layers_list(self):
        if not hasattr(self, 'stitch_layers_list') or not hasattr(self,
                                                                  'stitching_canvas') or not self.stitching_canvas:
            return

        self.stitch_layers_list.blockSignals(True)
        self.stitch_layers_list.clear()

        for i, item in reversed(list(enumerate(self.stitching_canvas.items))):
            list_item = QListWidgetItem(f"{_TR('图层')} {i + 1}: {item.name}")
            list_item.setData(Qt.ItemDataRole.UserRole, item.id)
            self.stitch_layers_list.addItem(list_item)

        selected_ids = {item.id for item in self.stitching_canvas.selected_items}
        for i in range(self.stitch_layers_list.count()):
            list_item = self.stitch_layers_list.item(i)
            if list_item.data(Qt.ItemDataRole.UserRole) in selected_ids:
                list_item.setSelected(True)

        self.stitch_layers_list.blockSignals(False)
        self.update_button_states()

    @Slot()
    def save_stitched_image(self):
        """拼图画板导出：获取内容边界后支持指定输出分辨率（异步 UI 刷新防卡死）"""
        if not self.stitching_canvas or not self.stitching_canvas.items:
            QMessageBox.warning(self, _TR("无法保存"), _TR("画布上没有任何内容。"))
            return

        total_bounds = QRectF()
        max_scale_ratio = 1.0  # 【核心修复】：找出当前画布中所有素材的最大缩放比，用于还原超清原图
        for item in self.stitching_canvas.items:
            total_bounds = total_bounds.united(item.get_transformed_bounding_rect())
            if item.size.width() > 0:
                ratio = item.pixmap.width() / item.size.width()
                if ratio > max_scale_ratio:
                    max_scale_ratio = ratio

        # 将画板边界恢复至最高清素材的原始像素尺寸
        final_size = QSizeF(total_bounds.width() * max_scale_ratio, total_bounds.height() * max_scale_ratio).toSize()

        if final_size.width() <= 0 or final_size.height() <= 0:
            return

        base_name = "stitched_creation.png"

        while True:
            dialog = ModernExportDialog('stitched', final_size, base_name, parent=self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            params = dialog.get_export_params()
            save_path = params['path']

            if os.path.exists(save_path):
                reply = QMessageBox.question(
                    self, _TR("确认覆盖"),
                    _TR("文件 '{}' 已存在。\n\n您确定要覆盖此文件吗？").format(os.path.basename(save_path)),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    break
                else:
                    base_name = os.path.basename(save_path)
                    continue
            else:
                break

        # 呼出遮罩并阻断交互
        self.show_global_loading_overlay(_TR("正在渲染画布与编码超清分辨率图像..."), 0)
        self.is_saving = True
        self.update_button_states()

        # 【核心修复】：把高强度的主线程画图操作包裹，同时将计算得到的最高清分辨率注入渲染器
        def _do_stitch_and_export():
            try:
                # 按照用户设定的导出尺寸构建超清底板
                final_qimage = QImage(params['size'], QImage.Format.Format_ARGB32_Premultiplied)
                final_qimage.fill(self.stitching_canvas.background_color)
                painter = QPainter(final_qimage)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

                # 将物理画布坐标系无损映射为用户导出的超高清分辨率系
                scale_x = params['size'].width() / total_bounds.width()
                scale_y = params['size'].height() / total_bounds.height()
                painter.scale(scale_x, scale_y)

                for item in self.stitching_canvas.items:
                    painter.save()
                    relative_pos = item.pos - total_bounds.topLeft()
                    transform = QTransform()
                    transform.translate(relative_pos.x() + item.size.width() / 2,
                                        relative_pos.y() + item.size.height() / 2)
                    transform.rotate(item.rotation)
                    transform.translate(-item.size.width() / 2, -item.size.height() / 2)

                    painter.setTransform(transform, combine=True)
                    painter.drawPixmap(QRectF(QPointF(0, 0), item.size), item.pixmap, item.pixmap.rect())
                    painter.restore()
                painter.end()

                # 提交给后台压缩写盘
                self.export_thread = AsyncImageExportThread(
                    final_qimage,
                    params['size'],
                    params['path'],
                    params['format'].upper()
                )
                self.export_thread.finished_signal.connect(self._on_async_image_export_finished)
                self.export_thread.finished_signal.connect(self.export_thread.deleteLater)
                self.export_thread.start()
            except Exception as e:
                self._on_async_image_export_finished(False, str(e))

        # 同样延迟 100 毫秒，释放主线程绘图权，防止遮罩层白屏
        QTimer.singleShot(100, _do_stitch_and_export)

    @Slot()
    def save_selected_stitched_item(self):
        """升级版：保存选中的单个素材（异步 UI 刷新防卡死）"""
        if not hasattr(self, 'stitching_canvas') or not self.stitching_canvas:
            return

        primary_item = self.stitching_canvas.get_primary_selected_item()
        if not primary_item:
            QMessageBox.warning(self, _TR("无法保存"), _TR("请先在画布上单选一个素材。"))
            return

        if self.is_saving:
            QMessageBox.information(self, _TR("忙碌"), _TR("其他保存操作正在进行中。"))
            return

        base_name = primary_item.name
        suffix = ""
        if getattr(primary_item, 'is_enhanced', False):
            suffix += "_enhanced"
        if getattr(primary_item, 'has_alpha_channel', False):
            suffix += "_matted"

        default_name = f"{base_name}{suffix}.png"

        # 确保提取的是经过 EnhanceWorker 增强之后的最高清原始 Pixmap
        orig_size = primary_item.pixmap.size()

        while True:
            dialog = ModernExportDialog('image', orig_size, default_name, parent=self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            params = dialog.get_export_params()
            save_path = params['path']

            if os.path.exists(save_path):
                reply = QMessageBox.question(
                    self, _TR("确认覆盖"),
                    _TR("文件 '{}' 已存在。\n\n您确定要覆盖此文件吗？").format(os.path.basename(save_path)),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    break
                else:
                    default_name = os.path.basename(save_path)
                    continue
            else:
                break

        # 先调出 UI 遮罩，并更新状态
        self.show_global_loading_overlay(_TR("正在进行超高分辨率缩放与编码..."), 0)
        self.is_saving = True
        self.update_button_states()

        # 将极度耗时的 QImage 提取和线程启动包裹在闭包里
        def _do_export():
            try:
                # 这里如果原图被放大到了几千万像素会极大吃满CPU
                qimage_to_save = primary_item.pixmap.toImage()

                self.export_thread = AsyncImageExportThread(
                    qimage_to_save,
                    params['size'],
                    params['path'],
                    params['format'].upper()
                )
                self.export_thread.finished_signal.connect(self._on_async_image_export_finished)
                self.export_thread.finished_signal.connect(self.export_thread.deleteLater)
                self.export_thread.start()
            except Exception as e:
                self._on_async_image_export_finished(False, str(e))

        QTimer.singleShot(100, _do_export)

    @Slot(bool, str)
    def _on_async_image_export_finished(self, success, result_msg):
        """图像/拼接导出的统一回调槽函数"""
        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.hide()

        self.is_saving = False
        self.update_button_states()

        if success:
            QMessageBox.information(self, "保存成功", f"图像已成功保存至:\n{result_msg}")
            self.show_status_message("图像保存成功。", 3000)
        else:
            QMessageBox.critical(self, "保存失败", f"导出时发生错误:\n{result_msg}")

    @Slot(bool, str, str)
    def _handle_save_stitched_completion(self, success, saved_path, error_message):
        # 【核心修复】：导出完成后，强制隐藏全局“系统处理中”弹窗遮罩
        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.hide()

        self.is_saving = False  # 解锁界面按钮
        self.update_button_states()

        if success:
            QMessageBox.information(self, "保存成功", f"拼接图像已成功保存至:\n{saved_path}")
        else:
            QMessageBox.critical(self, "保存失败", f"保存拼接图像时出错:\n{error_message}")

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)

        if not hasattr(self, '_main_resize_timer'):
            self._main_resize_timer = QTimer(self)
            self._main_resize_timer.setSingleShot(True)
            self._main_resize_timer.timeout.connect(self.update_static_previews_on_resize)
        self._main_resize_timer.start(50)

    def update_static_previews_on_resize(self):
        """Forces repaints on static image previews when main layout boundaries change."""
        if hasattr(self, 'image_compare_widget') and self.image_compare_widget.original_pixmap:
            self.image_compare_widget.update()

        if hasattr(self, 'video_result_preview_label') and getattr(self, 'video_segmentation_finished',
                                                                   False) and getattr(self, 'current_frame_index',
                                                                                      -1) != -1:
            self.update_video_preview_all_targets(self.current_frame_index)

    def _update_bg_color_button_style(self):
        """刷新纯色控制小块的背景指示"""
        clip_idx = self._get_active_editing_clip_idx()
        if clip_idx == -1 or clip_idx >= len(self.virtual_timeline):
            return

        clip = self.virtual_timeline[clip_idx]
        color = clip.get('bg_color', QColor(0, 255, 0))
        color_hex = color.name()

        if hasattr(self, 'bg_color_button'):
            text_color = 'white' if color.lightnessF() < 0.5 else 'black'
            self.bg_color_button.setStyleSheet(f"background-color: {color_hex}; color: {text_color};")

    def _get_active_editing_clip_idx(self) -> int:
        """获取当前正在编辑的视频片段的索引"""
        if getattr(self, '_matting_clip_idx', -1) != -1:
            return self._matting_clip_idx
        if getattr(self, '_current_crop_clip_idx', -1) != -1:
            return self._current_crop_clip_idx
        if hasattr(self, 'storyboard_list'):
            selected_items = self.storyboard_list.selectedItems()
            if selected_items:
                return self.storyboard_list.row(selected_items[0])
        idx, _ = self._get_current_clip_info()
        return idx if idx is not None else -1

    def set_video_bg_mode(self, mode: str):
        """
        设置当前选中片段的背景模式，其他片段不受任何影响。
        """
        clip_idx = self._get_active_editing_clip_idx()
        if clip_idx == -1 or not hasattr(self, 'virtual_timeline') or clip_idx >= len(self.virtual_timeline):
            return

        # 锁定仅修改当前单一片段的背景属性
        clip = self.virtual_timeline[clip_idx]

        if mode == "transparent":
            clip['bg_is_transparent'] = True
            clip['bg_image_path'] = None
            self._update_video_bg_selector_ui("transparent")

        elif mode == "image":
            file_path, _ = QFileDialog.getOpenFileNames(
                self,
                _TR("选择背景图片"),
                "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"
            )
            if file_path and len(file_path) > 0:
                clip['bg_is_transparent'] = False
                clip['bg_image_path'] = file_path[0]
                self._update_video_bg_selector_ui("image")
            else:
                return

        elif mode == "color":
            clip['bg_is_transparent'] = False
            clip['bg_image_path'] = None
            self._update_video_bg_selector_ui("color")

        # 标记当前正在编辑的片段为“脏”，以便在退出沙盒时进行局部的重新烘焙
        if getattr(self, '_matting_clip_idx', -1) != -1:
            self._is_matting_dirty = True

        # 增量刷新：如果已经生成过抠图成果，只针对当前片段重新生成局部缓存，不改变其他片段
        if getattr(self, 'video_segmentation_finished', False):
            self._rebuild_matted_preview_cache()
        else:
            if getattr(self, 'current_frame_index', -1) != -1:
                self._display_frame_wrapper(self.current_frame_index)
                if hasattr(self, 'update_video_preview_all_targets'):
                    self.update_video_preview_all_targets(self.current_frame_index)

    def _update_video_bg_selector_ui(self, active_mode: str = None):
        """根据当前片段的背景配置刷新右侧面板属性样式"""
        clip_idx = self._get_active_editing_clip_idx()
        if clip_idx == -1 or not hasattr(self, 'virtual_timeline') or clip_idx >= len(self.virtual_timeline):
            return

        clip = self.virtual_timeline[clip_idx]
        bg_is_transparent = clip.get('bg_is_transparent', False)
        bg_image_path = clip.get('bg_image_path', None)
        bg_color = clip.get('bg_color', QColor(0, 255, 0))

        if active_mode is None:
            if bg_is_transparent:
                active_mode = "transparent"
            elif bg_image_path:
                active_mode = "image"
            else:
                active_mode = "color"

        active_style = "QToolButton { background-color: #0A84FF; border-radius: 9px; padding: 2px; } QToolButton:hover { background-color: #0070E0; }"
        inactive_style = "QToolButton { background-color: rgba(255,255,255,0.08); border-radius: 9px; padding: 2px; } QToolButton:hover { background-color: rgba(255,255,255,0.2); }"

        if hasattr(self, 'vid_bg_transparent_btn'):
            self.vid_bg_transparent_btn.setStyleSheet(
                active_style if active_mode == "transparent" else inactive_style
            )

        if hasattr(self, 'vid_bg_color_btn'):
            color_hex = bg_color.name()
            if active_mode == "color":
                self.vid_bg_color_btn.setStyleSheet(
                    f"background-color: {color_hex}; border-radius: 9px; border: 2.5px solid #0A84FF;"
                )
            else:
                self.vid_bg_color_btn.setStyleSheet(
                    f"background-color: {color_hex}; border-radius: 9px; border: 1px solid #555555;"
                )

        if hasattr(self, 'vid_bg_image_btn'):
            self.vid_bg_image_btn.setStyleSheet(
                active_style if active_mode == "image" else inactive_style
            )

        if hasattr(self, 'vid_bg_preview_label'):
            if active_mode == "image" and bg_image_path:
                pixmap = QPixmap(bg_image_path)
                if not pixmap.isNull():
                    scaled_pix = pixmap.scaled(
                        self.vid_bg_preview_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.vid_bg_preview_label.setPixmap(scaled_pix)
                    self.vid_bg_preview_label.show()
                else:
                    self.vid_bg_preview_label.hide()
            else:
                self.vid_bg_preview_label.hide()

    def _select_video_bg_color(self):
        """为当前选中的片段选择背景颜色"""
        clip_idx = self._get_active_editing_clip_idx()
        if clip_idx == -1 or not hasattr(self, 'virtual_timeline') or clip_idx >= len(self.virtual_timeline):
            return

        clip = self.virtual_timeline[clip_idx]
        current_color = clip.get('bg_color', QColor(0, 255, 0))

        new_color = ModernColorDialog.getColor(current_color, self, "选择视频背景颜色")
        if new_color.isValid():
            clip['bg_color'] = new_color
            self._update_bg_color_button_style()
            self.set_video_bg_mode("color")

    def _assemble_global_render_dir(self) -> bool:
        """
        极速无损时间线重组引擎。
        通过系统级硬链接/符号链接，将故事板中各个视频片段已经烘焙完成的局部预览 JPG 文件夹，
        按照当前的时间轴顺序极速链结到全局 `temp_render_dir` 中，完全避免全段重复重度计算。
        """
        if not hasattr(self, 'virtual_timeline') or not self.virtual_timeline:
            return False

        import os
        import shutil

        old_render_dir = getattr(self, 'temp_render_dir', None)
        self.temp_render_dir = os.path.join(TEMP_BASE_DIR, f"render_global_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.temp_render_dir, exist_ok=True)

        global_idx = 0
        success = True

        try:
            for clip in self.virtual_timeline:
                baked_dir = clip.get('baked_preview_dir', None)
                has_baked_cache = baked_dir and os.path.exists(baked_dir)

                for local_idx in range(clip['frames']):
                    target_dst = os.path.join(self.temp_render_dir, f"{global_idx:05d}.jpg")

                    if has_baked_cache:
                        # 分支 A：使用已有的局部抠图发丝烘焙缓存
                        src_frame = os.path.join(baked_dir, f"{local_idx:05d}.jpg")
                    else:
                        # 分支 B：使用未编辑片段的原视频提取帧
                        src_frame = os.path.join(self.temp_frame_dir, f"{global_idx:05d}.jpg")
                        if not os.path.exists(src_frame):
                            src_frame = os.path.join(self.temp_frame_dir, f"{global_idx}.jpg")

                    if os.path.exists(src_frame):
                        try:
                            if os.path.exists(target_dst):
                                os.remove(target_dst)
                            os.link(src_frame, target_dst)
                        except Exception:
                            try:
                                shutil.copy2(src_frame, target_dst)
                            except Exception as e:
                                print(f"链接/拷贝帧文件失败: {e}")
                                success = False
                    else:
                        # 空白保护
                        import numpy as np
                        import cv2
                        black_canvas = np.zeros((self.video_height or 720, self.video_width or 1280, 3), dtype=np.uint8)
                        cv2.imwrite(target_dst, black_canvas)

                    global_idx += 1

            # 清理失效的上一代全局预览文件夹
            if old_render_dir and os.path.exists(old_render_dir) and old_render_dir != self.temp_render_dir:
                try:
                    shutil.rmtree(old_render_dir, ignore_errors=True)
                except Exception:
                    pass

            self.video_segmentation_finished = True
            if hasattr(self, '_video_frame_cache'):
                self._video_frame_cache.clear()

            return success
        except Exception as e:
            print(f"全局预览重组链结失败: {e}")
            traceback.print_exc()
            return False

    @Slot(bool, str)
    def _on_single_clip_bake_finished_wrapper(self, success: bool, result_msg: str):
        """单视频片段局部渲染烘焙结束的回调槽。"""
        if hasattr(self, '_single_bake_thread'):
            self._single_bake_thread.quit()
            self._single_bake_thread.wait()
            self._single_bake_thread.deleteLater()
            del self._single_bake_thread
        if hasattr(self, '_single_bake_worker'):
            self._single_bake_worker.deleteLater()
            del self._single_bake_worker

        if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
            self._global_loading_overlay.hide()

        clip_idx = getattr(self, '_matting_clip_idx', -1)
        if success and clip_idx != -1:
            self._is_matting_dirty = False
            clip = self.virtual_timeline[clip_idx]
            old_baked_dir = clip.get('baked_preview_dir', None)
            clip['baked_preview_dir'] = result_msg  # 存储本片段特有的渲染帧文件夹

            if old_baked_dir and os.path.exists(old_baked_dir):
                try:
                    shutil.rmtree(old_baked_dir, ignore_errors=True)
                except Exception:
                    pass

            # 刷新片段对应的封面
            first_baked_frame_path = os.path.join(result_msg, "00000.jpg")
            if os.path.exists(first_baked_frame_path):
                baked_cv = cv2.imread(first_baked_frame_path)
                if baked_cv is not None:
                    pixmap = convert_cv_to_pixmap(baked_cv)
                    if pixmap and not pixmap.isNull():
                        clip['pixmap'] = pixmap
                        if hasattr(self, 'storyboard_list'):
                            item = self.storyboard_list.item(clip_idx)
                            if item:
                                widget = self.storyboard_list.itemWidget(item)
                                if hasattr(widget, 'bg_label'):
                                    scaled_pix = pixmap.scaled(240, 135, Qt.AspectRatioMode.KeepAspectRatio,
                                                               Qt.TransformationMode.SmoothTransformation)
                                    widget.bg_label.setPixmap(scaled_pix)

            # 极速重构链结
            self._assemble_global_render_dir()

        self._exit_dedicated_matting_mode()

    def _rebuild_matted_preview_cache(self):
        """
        智能局部增量烘焙。
        【核心修复】：背景更换触发的自动烘焙完成后，必须清空 Dirty 脏标记，
        并且同步刷新底部故事板的封面预览图！
        """
        clip_idx = self._get_active_editing_clip_idx()
        if clip_idx == -1 or not hasattr(self, 'virtual_timeline') or clip_idx >= len(self.virtual_timeline):
            return

        was_playing = getattr(self, 'is_playing', False)
        if was_playing:
            self.pause_video()

        self.show_global_loading_overlay(_TR("正在应用新背景并重构局部播放轨道..."), 0)

        clip = self.virtual_timeline[clip_idx]
        local_baked_dir = os.path.join(TEMP_BASE_DIR, f"baked_clip_{uuid.uuid4().hex[:8]}")

        global_start = sum(v['frames'] for v in self.virtual_timeline[:clip_idx])
        global_end = global_start + clip['frames'] - 1

        self._single_bake_thread = QThread(self)
        self._single_bake_worker = BakeSingleClipWorker(
            raw_frame_dir=self.temp_frame_dir,
            processed_masks=self.processed_masks,
            start_frame=global_start,
            end_frame=global_end,
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

        def on_rebuild_complete(success, result_msg):
            self._single_bake_thread.quit()
            self._single_bake_thread.wait()
            self._single_bake_thread.deleteLater()
            self._single_bake_worker.deleteLater()

            if hasattr(self, '_global_loading_overlay') and self._global_loading_overlay:
                self._global_loading_overlay.hide()

            if success:
                old_baked_dir = clip.get('baked_preview_dir', None)
                clip['baked_preview_dir'] = local_baked_dir
                if old_baked_dir and os.path.exists(old_baked_dir):
                    try:
                        shutil.rmtree(old_baked_dir, ignore_errors=True)
                    except Exception:
                        pass

                # =========================================================================
                # 【新增修复】：更换背景自动烘焙完成后，同步抓取新背景渲染图刷新故事板的封面！
                # =========================================================================
                first_baked_frame_path = os.path.join(local_baked_dir, "00000.jpg")
                if os.path.exists(first_baked_frame_path):
                    import cv2
                    from core.utils import convert_cv_to_pixmap
                    baked_cv = cv2.imread(first_baked_frame_path)
                    if baked_cv is not None:
                        pixmap = convert_cv_to_pixmap(baked_cv)
                        if pixmap and not pixmap.isNull():
                            clip['pixmap'] = pixmap
                            if hasattr(self, 'storyboard_list'):
                                item = self.storyboard_list.item(clip_idx)
                                if item:
                                    widget = self.storyboard_list.itemWidget(item)
                                    if hasattr(widget, 'bg_label'):
                                        scaled_pix = pixmap.scaled(240, 135, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                                        widget.bg_label.setPixmap(scaled_pix)

                # 清除脏标记，防止退出时再次触发全局冗余烘焙
                self._is_matting_dirty = False

                # 瞬间重组链结
                self._assemble_global_render_dir()

                if getattr(self, 'current_frame_index', -1) != -1:
                    self._display_frame_wrapper(self.current_frame_index)
                    if hasattr(self, 'update_video_preview_all_targets'):
                        self.update_video_preview_all_targets(self.current_frame_index)

                if was_playing:
                    self.play_video()
            else:
                QMessageBox.warning(self, _TR("背景更新失败"), _TR("重构播放轨道时出错: ") + result_msg)

        self._single_bake_worker.finished.connect(on_rebuild_complete)
        self._single_bake_thread.started.connect(self._single_bake_worker.run)
        self._single_bake_thread.start()

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)

    def update_button_states(self):
        """
        根据当前页面、后台任务和播放状态，计算并刷新所有控制控件的启用/禁用和勾选状态。
        """
        if hasattr(self, 'cumulative_sam_checkbox'):
            if not getattr(self.cumulative_sam_checkbox, '_user_toggled_sam', False):
                self.cumulative_sam_checkbox.blockSignals(True)
                self.cumulative_sam_checkbox.setChecked(True)
                self.cumulative_sam_checkbox.blockSignals(False)

        if hasattr(self, 'vid_matteformer_checkbox'):
            if not getattr(self.vid_matteformer_checkbox, '_user_toggled_vid', False):
                self.vid_matteformer_checkbox.blockSignals(True)
                self.vid_matteformer_checkbox.setChecked(True)
                self.vid_matteformer_checkbox.blockSignals(False)
                try:
                    self.vid_matteformer_checkbox.clicked.disconnect()
                except Exception:
                    pass
                self.vid_matteformer_checkbox.clicked.connect(
                    lambda: setattr(self.vid_matteformer_checkbox, '_user_toggled_vid', True)
                )

        def set_enabled(widget_name, state):
            widget = getattr(self, widget_name, None)
            if widget and isinstance(widget, (QWidget, QAction)):
                try:
                    if widget.parent() is not None or isinstance(widget, QMainWindow) or isinstance(widget, QAction):
                        widget.setEnabled(bool(state))
                except RuntimeError:
                    pass

        is_enhancing_task_running = getattr(self, 'is_enhancing', False) or any(
            k.startswith("enhance_") for k in getattr(self, 'active_workers', {})
        )
        is_predicting_task_running = getattr(self, 'is_predicting', False) or "predict" in getattr(self,
                                                                                                   'active_workers', {})
        is_saving_task_running = getattr(self, 'is_saving', False) or any(
            k.startswith("save_") for k in getattr(self, 'active_workers', {})
        )
        is_video_task_running = getattr(self, 'is_extracting_frames', False) or getattr(self,
                                                                                        'video_segmentation_running',
                                                                                        False)
        is_busy = getattr(self, 'is_loading_model',
                          False) or is_enhancing_task_running or is_predicting_task_running or is_saving_task_running or is_video_task_running

        current_page_idx = self.stacked_widget.currentIndex()

        if current_page_idx == self.CREATIVE_WORKSHOP_INDEX:
            in_seg_mode = getattr(self, 'is_in_segmentation_overlay_mode', False)
            canvas_exists = hasattr(self, 'stitching_canvas') and self.stitching_canvas
            seg_label = getattr(self, 'segmentation_overlay_label', None)

            canvas_has_items = canvas_exists and bool(self.stitching_canvas.items)
            has_selection = canvas_has_items and bool(self.stitching_canvas.selected_items)
            is_single_selection = has_selection and len(self.stitching_canvas.selected_items) == 1

            can_undo_seg = bool(in_seg_mode and seg_label and len(seg_label.mask_history) > 1)
            can_redo_seg = bool(in_seg_mode and seg_label and len(seg_label.redo_stack) > 0)

            set_enabled('workshop_main_top_bar', not in_seg_mode)
            set_enabled('segmentation_overlay_top_bar', in_seg_mode)
            set_enabled('segmentation_tool_palette', in_seg_mode)

            set_enabled('load_stitch_image_button', not in_seg_mode and not is_busy)
            set_enabled('save_stitched_button', not in_seg_mode and canvas_has_items and not is_busy)
            set_enabled('clear_stitching_canvas_button', not in_seg_mode and canvas_has_items and not is_busy)

            set_enabled('workshop_asset_library_button', not in_seg_mode and not is_busy)
            set_enabled('workshop_item_tools_button', not in_seg_mode and not is_busy and has_selection)
            set_enabled('workshop_canvas_settings_button', not in_seg_mode and not is_busy)
            set_enabled('workshop_layers_button', not in_seg_mode and not is_busy and canvas_has_items)

            set_enabled('workshop_enhance_button',
                        not in_seg_mode and is_single_selection and not is_busy and globals().get(
                            'REALESRGAN_AVAILABLE', True))
            set_enabled('segment_selected_button',
                        not in_seg_mode and is_single_selection and not is_busy and getattr(self,
                                                                                            'image_predictor_loaded',
                                                                                            False))
            set_enabled('save_selected_item_button', not in_seg_mode and is_single_selection and not is_busy)

            set_enabled('save_segmentation_button', in_seg_mode and not is_busy)
            set_enabled('apply_segmentation_button', in_seg_mode and not is_busy)
            set_enabled('cancel_segmentation_button', in_seg_mode and not is_busy)

            set_enabled('seg_undo_tool', can_undo_seg and not is_busy)
            set_enabled('seg_redo_tool', can_redo_seg and not is_busy)
            set_enabled('seg_sam_mode_tool', in_seg_mode and not is_busy)
            set_enabled('seg_paint_mode_tool', in_seg_mode and not is_busy)
            set_enabled('seg_preview_tool',
                        in_seg_mode and not is_busy and seg_label and getattr(seg_label, 'current_mask',
                                                                              None) is not None and np.any(
                            getattr(seg_label, 'current_mask', [False])))
            set_enabled('seg_reset_view_tool', in_seg_mode and not is_busy)

            if hasattr(self, 'undo_action'):
                self.undo_action.setEnabled(can_undo_seg and not is_busy)
            if hasattr(self, 'redo_action'):
                self.redo_action.setEnabled(can_redo_seg and not is_busy)
            if hasattr(self, 'delete_action'):
                self.delete_action.setEnabled(not in_seg_mode and has_selection and not is_busy)

            primary_item = getattr(self.stitching_canvas, 'get_primary_selected_item',
                                   lambda: None)() if canvas_exists else None

            can_combine = not in_seg_mode and canvas_exists and len(self.stitching_canvas.selected_items) > 1
            set_enabled('combine_selected_button', can_combine and not is_busy)

            can_uncombine = not in_seg_mode and is_single_selection and primary_item and getattr(primary_item,
                                                                                                 'source_items',
                                                                                                 None) is not None
            set_enabled('uncombine_selected_button', can_uncombine and not is_busy)

            enhance_btn_enabled = not in_seg_mode and is_single_selection and not is_busy and globals().get(
                'REALESRGAN_AVAILABLE', True)
            set_enabled('enhance_selected_button', enhance_btn_enabled)
            set_enabled('enhance_scale_combo', enhance_btn_enabled)
            set_enabled('tile_mode_combo', enhance_btn_enabled)
            set_enabled('enhance_model_combo', enhance_btn_enabled)

            if hasattr(self, 'tile_mode_combo'):
                is_custom_tile = self.tile_mode_combo.currentText() == "自定义"
                set_enabled('custom_tile_size_spinbox', enhance_btn_enabled and is_custom_tile)

            set_enabled('res_original_radio', in_seg_mode and not is_busy)
            set_enabled('res_768_radio', in_seg_mode and not is_busy)
            set_enabled('res_1280_radio', in_seg_mode and not is_busy)
            set_enabled('res_1920_radio', in_seg_mode and not is_busy)
            set_enabled('res_custom_radio', in_seg_mode and not is_busy)

            if hasattr(self, 'res_custom_radio'):
                is_custom_res = self.res_custom_radio.isChecked()
                set_enabled('custom_max_dim_spinbox', in_seg_mode and not is_busy and is_custom_res)

        elif current_page_idx == self.VIDEO_SEG_PAGE_INDEX:
            deps_ok = getattr(self, 'SAM2_VIDEO_PREDICTOR_AVAILABLE', True) and getattr(self, 'TORCH_AVAILABLE',
                                                                                        True) and getattr(self,
                                                                                                          'PILLOW_AVAILABLE',
                                                                                                          True)
            # 同时支持直连快速抠图模型 (ImagePredictor) 或时序大模型 (VideoPredictor) 两种后端加载校验
            is_direct_mat = hasattr(self,
                                    'vid_direct_matting_checkbox') and self.vid_direct_matting_checkbox.isChecked()
            predictor_loaded = getattr(self, 'video_predictor_loaded', False) or (
                        is_direct_mat and getattr(self, 'image_predictor_loaded', False))
            pred_ready = deps_ok and predictor_loaded and not getattr(self, 'sam_video_load_failed', False)

            vid_loaded = getattr(self, 'video_path', None) is not None and getattr(self, 'total_frames', 0) > 0
            can_operate = pred_ready and not is_busy
            vid_label = getattr(self, 'video_display_label', None)
            is_video_compare_mode = vid_label and getattr(vid_label, '_is_in_compare_mode', False)

            can_interact_points = vid_loaded and can_operate and not getattr(self, 'is_playing', False)
            can_manage_layers = vid_loaded and can_operate and not getattr(self, 'is_playing', False)

            # =========================================================================
            # 【核心修复】：播放停止或添加对象后动态恢复并激活画布标注能力
            # =========================================================================
            if vid_label:
                is_matting_page = getattr(self, 'vid_editor_stack', None) and \
                                  self.vid_editor_stack.currentWidget() == getattr(self, 'vid_dedicated_matting_page',
                                                                                   None)
                # 只有在抠图沙盒页面中，且视频属于非播放非繁忙状态，解锁交互属性
                allow_interact = is_matting_page and can_interact_points
                vid_label.set_allow_interaction(allow_interact)
                vid_label.update_cursor()
            # =========================================================================

            has_current_target = getattr(self, 'current_target_id', -1) != -1 and getattr(self, 'current_target_id',
                                                                                          -1) in getattr(self,
                                                                                                         'target_points',
                                                                                                         {})
            can_start_seg = can_manage_layers and any(
                d.get('points') or d.get('box') for d in getattr(self, 'target_points', {}).values())
            can_save = vid_loaded and getattr(self, 'video_segmentation_finished', False) and bool(
                getattr(self, 'processed_masks', {})) and not is_saving_task_running
            can_play_ctrl = vid_loaded and not getattr(self, 'video_segmentation_running', False)
            can_compare = vid_loaded and getattr(self, 'video_segmentation_finished', False) and not getattr(self,
                                                                                                             'video_segmentation_running',
                                                                                                             False)

            if not can_compare and is_video_compare_mode:
                if hasattr(self, 'video_compare_mode_button') and self.video_compare_mode_button.isChecked():
                    self.video_compare_mode_button.blockSignals(True)
                    self.video_compare_mode_button.setChecked(False)
                    self.video_compare_mode_button.blockSignals(False)
                    if vid_label:
                        vid_label.set_compare_mode(False)

            set_enabled('video_help_button_top', True)
            set_enabled('load_video_button_top', can_operate)
            set_enabled('video_compare_mode_button', can_compare)
            set_enabled('vid_objects_button', can_manage_layers)
            set_enabled('vid_process_button', can_interact_points)
            set_enabled('vid_output_button', vid_loaded)

            use_matanyone = True
            if hasattr(self, 'vid_matteformer_checkbox'):
                use_matanyone = self.vid_matteformer_checkbox.isChecked()

            max_allowed_objs = globals().get('MAX_VIDEO_OBJS', 10)
            set_enabled('add_target_button',
                        can_interact_points and len(getattr(self, 'target_points', {})) < max_allowed_objs)

            set_enabled('delete_current_target_button', can_manage_layers and has_current_target)

            has_undo = bool(getattr(self, 'video_undo_stack', []))
            has_redo = bool(getattr(self, 'video_redo_stack', []))

            set_enabled('vid_undo_button', can_manage_layers and has_undo)
            set_enabled('vid_redo_button', can_manage_layers and has_redo)
            set_enabled('vid_mat_undo_button', can_manage_layers and has_undo)
            set_enabled('vid_mat_redo_button', can_manage_layers and has_redo)

            set_enabled('start_video_seg_button', can_start_seg)
            set_enabled('cancel_video_seg_button',
                        getattr(self, 'is_extracting_frames', False) or getattr(self, 'video_segmentation_running',
                                                                                False) or (
                                getattr(self, 'is_saving', False) and "save_video" in getattr(self, 'active_workers',
                                                                                              {})))

            can_save_out = vid_loaded and not is_saving_task_running
            set_enabled('save_video_seg_button', can_save_out)
            set_enabled('bg_color_button', can_save_out)

            set_enabled('video_thumbnail_scrubber', can_play_ctrl and not getattr(self, 'is_playing', False))
            set_enabled('video_frame_spinbox', can_play_ctrl and not getattr(self, 'is_playing', False))
            set_enabled('play_pause_button', can_play_ctrl)
            set_enabled('stop_button', can_play_ctrl and (
                    getattr(self, 'is_playing', False) or getattr(self, 'current_frame_index', -1) != 0))

            if hasattr(self, 'play_pause_button'):
                icon_name = "pause-fill.svg" if getattr(self, 'is_playing', False) else "play-fill.svg"
                self.play_pause_button.setIcon(self._create_svg_icon(icon_name, color="#E0E0E0"))

            if hasattr(self, 'undo_action'):
                self.undo_action.setEnabled(can_manage_layers and has_undo)
            if hasattr(self, 'redo_action'):
                self.redo_action.setEnabled(can_manage_layers and has_redo)
            if hasattr(self, 'delete_action'):
                self.delete_action.setEnabled(can_manage_layers and has_current_target)

    def _is_preset_asset(self, path_to_check: str) -> bool:
        """Determines if the passed absolute path correlates to standard system resources."""
        try:
            preset_asset_dir_absolute = get_asset_path("")
            norm_preset_dir = os.path.normpath(preset_asset_dir_absolute)
            norm_image_path = os.path.normpath(path_to_check)
            norm_image_dir = os.path.dirname(norm_image_path)
            return (norm_image_dir == norm_preset_dir)
        except Exception as e:
            self.log_message.emit(f"Error checking preset asset status: {e}")
            return False

    def load_user_assets_from_config(self):
        """Scans the user asset configuration file and populates the library grids."""
        if not os.path.exists(getattr(self, 'user_assets_config_path', '')):
            return

        if not hasattr(self, 'stitch_asset_grid_layout') or self.stitch_asset_grid_layout is None:
            return

        try:
            with open(self.user_assets_config_path, 'r', encoding='utf-8') as f:
                user_paths = [line.strip() for line in f if line.strip()]

            current_grid_layout = self.stitch_asset_grid_layout

            for path in user_paths:
                if os.path.exists(path):
                    is_duplicate = False
                    for i in range(current_grid_layout.count()):
                        widget = current_grid_layout.itemAt(i).widget()
                        if isinstance(widget, AssetThumbnail) and os.path.normpath(
                                widget.image_path) == os.path.normpath(path):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        self.add_thumbnail_to_grid(path, target_grid=current_grid_layout)
        except Exception as e:
            self.log_message.emit(f"Error loading user assets: {e}")

    def closeEvent(self, event: QCloseEvent):
        """Reclaims active worker threads and synchronizes config state variables before exit."""
        self.log_message.emit("Closing application...")

        unsaved_vid = False
        unsaved_creative = False

        if (getattr(self, 'video_path', None) and getattr(self, 'video_segmentation_finished', False) and
                not getattr(self, 'video_segmentation_saved', True) and bool(getattr(self, 'processed_masks', {}))):
            unsaved_vid = True

        if hasattr(self, 'stitching_canvas') and self.stitching_canvas and self.stitching_canvas.items:
            unsaved_creative = True

        running_workers = list(getattr(self, 'active_workers', {}).keys())

        if unsaved_vid or unsaved_creative or running_workers:
            message = _TR("退出前请注意:\n\n")
            if unsaved_vid:
                message += _TR("  • 视频抠图有未保存的结果。\n")
            if unsaved_creative:
                message += _TR("  • 创意工坊有未保存的创作。\n")
            if running_workers:
                message += f"  • {_TR('后台任务正在运行:')} {', '.join(_TR(WORKER_ID_TO_CN.get(wid, wid)) for wid in running_workers)}\n"
            message += _TR("\n您确定要强制退出吗？ (正在运行的任务将尝试取消)")

            reply = QMessageBox.question(
                self, _TR('确认退出'), message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        self._save_settings()

        if hasattr(self, '_refinement_threads_pool'):
            for r_thread in list(self._refinement_threads_pool):
                if r_thread.isRunning():
                    r_thread.wait(500)

        if hasattr(self, 'stitch_asset_grid_layout') and self.stitch_asset_grid_layout is not None:
            user_asset_paths = []
            for i in range(self.stitch_asset_grid_layout.count()):
                widget = self.stitch_asset_grid_layout.itemAt(i).widget()
                if isinstance(widget, AssetThumbnail) and not self._is_preset_asset(widget.image_path):
                    user_asset_paths.append(widget.image_path)
            try:
                if hasattr(self, 'user_assets_config_path'):
                    with open(self.user_assets_config_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(user_asset_paths))
            except Exception as e:
                self.log_message.emit(f"Error saving user assets configuration: {e}")

        if running_workers:
            for wid in running_workers:
                self._cancel_worker(wid)

        for timer_attr in ['_main_resize_timer', '_refinement_update_timer', 'playback_timer']:
            timer = getattr(self, timer_attr, None)
            if timer and isinstance(timer, QTimer) and timer.isActive():
                timer.stop()

        self.reset_video_state()

        if os.path.exists(TEMP_BASE_DIR):
            try:
                if not os.listdir(TEMP_BASE_DIR):
                    os.rmdir(TEMP_BASE_DIR)
            except Exception:
                pass

        if getattr(self, 'image_predictor', None):
            del self.image_predictor
            self.image_predictor = None

        if getattr(self, 'video_predictor', None):
            del self.video_predictor
            self.video_predictor = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        event.accept()

    def _cancel_worker(self, worker_id):
        """Dispatches cancellation signals to registered thread operations."""
        worker_id_cn = worker_id
        if hasattr(self, 'WORKER_ID_TO_CN'):
            worker_id_cn = self.WORKER_ID_TO_CN.get(worker_id, worker_id)

        if worker_id in getattr(self, 'active_workers', {}):
            thread, worker = self.active_workers[worker_id]

            if hasattr(worker, 'cancel'):
                try:
                    worker.cancel()
                except RuntimeError:
                    pass

            if worker_id == "propagate_video_v1991" and getattr(self, 'current_propagate_video_progress_dialog', None):
                try:
                    self.current_propagate_video_progress_dialog.cancel()
                except RuntimeError:
                    pass
            elif worker_id == "enhance":
                enhance_pd = getattr(self, f"current_{worker_id}_progress", None)
                if enhance_pd and isinstance(enhance_pd, QProgressDialog):
                    try:
                        enhance_pd.cancel()
                    except RuntimeError:
                        pass
            elif worker_id == "save_segment":
                save_seg_pd = getattr(self, f"current_{worker_id}_progress", None)
                if save_seg_pd and isinstance(save_seg_pd, QProgressDialog):
                    try:
                        save_seg_pd.cancel()
                    except RuntimeError:
                        pass

            if worker_id in ["extract", "propagate_video_v1991", "save_video", "load_video_model", "init_video_state"]:
                if hasattr(self, 'video_info_label_display'):
                    try:
                        current_text_line1 = self.video_info_label_display.text().split('\n')[0]
                        self.video_info_label_display.setText(f"{current_text_line1}\n状态: 已取消")
                    except RuntimeError:
                        pass

    @Slot()
    def _toggle_application_language(self):
        """Cycles through language contexts and updates matching widgets on execution."""
        from config.settings import get_app_lang, set_app_lang

        current = get_app_lang()
        new_lang = "en" if current == "zh" else "zh"
        set_app_lang(new_lang)

        self.lang_toggle_btn.setText("EN / 中" if new_lang == "en" else "中 / EN")
        self.lang_toggle_btn.setStyleSheet("""
            QPushButton { background-color: #1A73E8; color: #FFFFFF; border-radius: 12px; padding: 4px 14px; font-weight: bold; font-size: 12px; border: none; }
        """ if new_lang == "en" else """
            QPushButton { background-color: #2C2C2E; color: #A0A0A5; border-radius: 12px; padding: 4px 14px; font-weight: bold; font-size: 12px; border: 1px solid #3A3A3C; }
        """)

        self._translate_ui_node(self)

        if hasattr(self, 'floating_panel_container') and self.floating_panel_container:
            self._translate_ui_node(self.floating_panel_container)

        if hasattr(self, 'asset_library_panel_floating') and self.asset_library_panel_floating:
            self._translate_ui_node(self.asset_library_panel_floating)

        if hasattr(self, 'batch_matting_page') and self.batch_matting_page:
            self._translate_ui_node(self.batch_matting_page)
            self.batch_matting_page.update_ui_state()

        if hasattr(self, 'video_display_label') and self.video_display_label:
            if getattr(self.video_display_label, 'current_pixmap', None) is None:
                self.video_display_label.setText(_TR("拖放视频/GIF文件至此，\n或使用“加载”按钮，\n支持缩放/平移。"))

        if hasattr(self, 'segmentation_overlay_label') and self.segmentation_overlay_label:
            if getattr(self.segmentation_overlay_label, 'original_cv_image', None) is None:
                self.segmentation_overlay_label.setText(_TR("拖放图像文件至此\n或使用按钮加载"))

        if hasattr(self, 'image_compare_widget_enhance') and self.image_compare_widget_enhance:
            if getattr(self.image_compare_widget_enhance, 'original_pixmap', None) is None:
                self.image_compare_widget_enhance.update()

        if hasattr(self, 'update_stitch_layers_list'):
            self.update_stitch_layers_list()
        if hasattr(self, '_refresh_video_objects_list'):
            self._refresh_video_objects_list()
        if hasattr(self, '_update_current_target_label'):
            self._update_current_target_label()

        self.show_status_message("Language switched to English" if new_lang == "en" else "已切换为中文", 2000)

    def _translate_ui_node(self, node):
        """Recursively parses widget hierarchies to translate text variables."""
        if not node:
            return

        if isinstance(node, QWidget):
            orig_tooltip = node.property("orig_tooltip")
            if orig_tooltip is None:
                curr_tooltip = node.toolTip()
                if curr_tooltip:
                    node.setProperty("orig_tooltip", curr_tooltip)
                    orig_tooltip = curr_tooltip
            if orig_tooltip:
                node.setToolTip(_TR(orig_tooltip))

            if hasattr(node, "placeholderText") and hasattr(node, "setPlaceholderText"):
                orig_placeholder = node.property("orig_placeholder")
                if orig_placeholder is None:
                    curr_placeholder = node.placeholderText()
                    if curr_placeholder:
                        node.setProperty("orig_placeholder", curr_placeholder)
                        orig_placeholder = curr_placeholder
                if orig_placeholder:
                    node.setPlaceholderText(_TR(orig_placeholder))

        if isinstance(node, (QLabel, QPushButton, QToolButton, QCheckBox, QRadioButton)):
            orig_text = node.property("orig_text")
            if orig_text is None:
                orig_text = node.text()
                if orig_text and not orig_text.isnumeric() and ":" not in orig_text and not orig_text.endswith(
                        "px") and not orig_text.endswith("%"):
                    node.setProperty("orig_text", orig_text)
            if orig_text:
                node.setText(_TR(orig_text))

        elif isinstance(node, QGroupBox):
            orig_title = node.property("orig_title")
            if orig_title is None:
                orig_title = node.title()
                if orig_title:
                    node.setProperty("orig_title", orig_title)
            if orig_title:
                node.setTitle(_TR(orig_title))

        elif isinstance(node, QComboBox):
            node.blockSignals(True)
            for i in range(node.count()):
                orig_item = node.itemData(i, Qt.ItemDataRole.UserRole)
                if orig_item is None:
                    orig_item = node.itemText(i)
                    if orig_item:
                        node.setItemData(i, orig_item, Qt.ItemDataRole.UserRole)
                if orig_item:
                    node.setItemText(i, _TR(orig_item))
            node.blockSignals(False)

        elif isinstance(node, QTabWidget):
            tab_cache = node.property("orig_tab_texts")
            if tab_cache is None:
                tab_cache = {}

            changed = False
            for i in range(node.count()):
                if i not in tab_cache:
                    tab_cache[i] = node.tabText(i)
                    changed = True

                orig_text = tab_cache[i]
                if orig_text:
                    node.setTabText(i, _TR(orig_text))

            if changed:
                node.setProperty("orig_tab_texts", tab_cache)

        elif isinstance(node, QListWidget):
            for i in range(node.count()):
                item = node.item(i)
                orig_item_text = item.data(Qt.ItemDataRole.UserRole + 999)
                if orig_item_text is None:
                    orig_item_text = item.text()
                    if orig_item_text:
                        item.setData(Qt.ItemDataRole.UserRole + 999, orig_item_text)
                if orig_item_text:
                    item.setText(_TR(orig_item_text))

                item_widget = node.itemWidget(item)
                if item_widget:
                    self._translate_ui_node(item_widget)

        if type(node).__name__ == "GradientTitleLabel":
            if hasattr(node, 'text_parts'):
                new_parts = []
                for i, (text, color_or_grad) in enumerate(node.text_parts):
                    orig_t = node.property(f"orig_text_{i}")
                    if orig_t is None:
                        orig_t = text
                        node.setProperty(f"orig_text_{i}", orig_t)
                    new_parts.append((_TR(orig_t), color_or_grad))
                node.set_text_parts(new_parts)

        for action in node.findChildren(QAction):
            orig_act_text = action.property("orig_text")
            if orig_act_text is None:
                orig_act_text = action.text()
                if orig_act_text:
                    action.setProperty("orig_text", orig_act_text)
            if orig_act_text:
                action.setText(_TR(orig_act_text))

            orig_act_tooltip = action.property("orig_tooltip")
            if orig_act_tooltip is None:
                curr_act_tooltip = action.toolTip()
                if curr_act_tooltip:
                    action.setProperty("orig_tooltip", curr_act_tooltip)
                    orig_act_tooltip = curr_act_tooltip
            if orig_act_tooltip:
                action.setToolTip(_TR(orig_act_tooltip))

        for child in node.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            self._translate_ui_node(child)

