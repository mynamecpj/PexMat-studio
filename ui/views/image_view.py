import os
import numpy as np
from typing import Optional
import cv2
import traceback
import collections
import math
import torch

from config.settings import (DEFAULT_REFINE_SHIFT, DEFAULT_REFINE_GUIDED_FILTER_ENABLED,
    DEFAULT_REFINE_GUIDED_FILTER_RADIUS, DEFAULT_REFINE_GUIDED_FILTER_EPS_SCALED, MIN_ZOOM, MAX_UNDO_HISTORY,
    DEFAULT_REFINE_SMOOTH, DEFAULT_REFINE_FEATHER, MAX_ZOOM, ZOOM_FACTOR, DEFAULT_MASK_ALPHA_IMAGE,
    STITCHING_DEFAULT_CANVAS_WIDTH, STITCHING_DEFAULT_CANVAS_HEIGHT)
from core.utils import convert_cv_to_pixmap, _static_apply_mask_refinements, create_checkerboard_pixmap, get_asset_path

from PySide6.QtWidgets import (QApplication, QLabel, QWidget, QMessageBox, QSizePolicy, QCheckBox, QHBoxLayout,
                               QVBoxLayout, QButtonGroup, QToolButton, QPushButton, QFrame, QGraphicsDropShadowEffect,
                               QGraphicsOpacityEffect, QFileDialog, QTabWidget, QStackedWidget)
from PySide6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QCursor, QMouseEvent, QDragEnterEvent, QDropEvent,
    QResizeEvent, QWheelEvent, QBrush, QPaintEvent, QPainterPath, QTransform, QMovie, QFont, QImageReader)
from PySide6.QtCore import (Qt, QPoint, QPointF, QRect, QRectF, QSize, QTimer, QEvent, Signal, QSizeF, Slot,
                            QPropertyAnimation, QEasingCurve, QThread, QCoreApplication, QParallelAnimationGroup)

import ui.components.panels
from ui.views.canvas_view import StitchingCanvas, FloatingPillToolbar

# ==========================================
# Modern floating card and frameless button style (Dark refactored version)
# ==========================================
MODERN_TOP_BAR_BUTTON_STYLE = """
    QToolButton {
        border: none;
        background-color: transparent;
        border-radius: 10px; 
        padding: 6px 8px; 
        min-width: 64px;
        color: #E0E0E0; 
        font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
        font-weight: 500;
        font-size: 12px;
    }
    QToolButton:hover {
        background-color: #333333; 
        color: #FFFFFF;
    }
    QToolButton:pressed {
        background-color: #404040;
    }
    QToolButton:checked {
        background-color: rgba(26, 115, 232, 0.2); 
        color: #60A5FA; 
        font-weight: bold;
    }
"""

FLOATING_CARD_STYLE = """
    QWidget#FloatingCard {
        background-color: #262626;
        border-radius: 14px;
        border: 1px solid #333333;
    }
"""

MODERN_PUSHBUTTON_STYLE = """
    QPushButton {
        background-color: #333333;
        color: #E0E0E0;
        border: 1px solid #404040;
        border-radius: 6px;
        padding: 6px 16px;
        font-size: 13px;
        font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    }
    QPushButton:hover {
        background-color: #404040; 
        border: 1px solid #555555;
        color: #FFFFFF;
    }
    QPushButton:pressed {
        background-color: #2A2A2A;
        border: 1px solid #1A73E8;
    }
"""

class FloatingToast(QLabel):
    """Modern fade-in/fade-out floating toast notification."""

    def __init__(self, parent, text, duration_ms=2000):
        super().__init__(text, parent)
        self.duration_ms = duration_ms

        self.setStyleSheet("""
            QLabel {
                background-color: rgba(17, 24, 39, 0.85);
                color: #FFFFFF;
                border-radius: 12px;
                padding: 12px 24px;
                font-family: -apple-system, "Microsoft YaHei", sans-serif;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.adjustSize()

        parent_rect = parent.rect()
        x = (parent_rect.width() - self.width()) // 2
        y = (parent_rect.height() - self.height()) // 2 - 40
        self.move(x, y)

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)

        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(300)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.setEasingCurve(QEasingCurve.Type.OutQuad)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.fade_out)
        self.timer.setSingleShot(True)

    def show_toast(self):
        self.show()
        self.raise_()
        self.anim.start()
        self.timer.start(self.duration_ms)

    def fade_out(self):
        self.anim.setDirection(QPropertyAnimation.Direction.Backward)
        self.anim.finished.connect(self.close)
        self.anim.start()


class ImageViewMixin:
    """Image processing and creative workshop UI construction and logic."""

    def _create_modern_separator(self, is_horizontal=False):
        sep = QWidget()
        if is_horizontal:
            sep.setFixedHeight(1)
            sep.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            sep.setFixedWidth(1)
            sep.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sep.setStyleSheet("background-color: #333333; margin: 8px 0px;")
        return sep

    def _add_drop_shadow(self, widget, radius=15, alpha=15, offset_y=3):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(radius)
        shadow.setColor(QColor(0, 0, 0, alpha))
        shadow.setOffset(0, offset_y)
        widget.setGraphicsEffect(shadow)

    @Slot()
    def auto_segment_action(self):
        if getattr(self, 'is_predicting', False):
            toast = FloatingToast(self.workshop_content_container, "⏳ 正在拼命计算中，请勿重复点击...")
            toast.show_toast()
            return

        if not getattr(self, 'image_predictor_loaded', False):
            QMessageBox.warning(self, "模型未就绪", "图像分割模型尚未加载完成，请稍候。")
            return

        img_label = getattr(self, 'segmentation_overlay_label', None)
        if not img_label or img_label.original_cv_image is None:
            QMessageBox.warning(self, "无图像", "请先在画布中选择素材并进入抠图模式。")
            return

        if getattr(img_label, '_has_auto_matted', False):
            self.log_message.emit("🔄 用户强制重置并重新运行一键抠图...")

        self.show_status_message("正在启动一键自动抠图 (智能大模型引擎)...", 0)

        if hasattr(self, 'matteformer_checkbox'):
            self.matteformer_checkbox.blockSignals(True)
            self.matteformer_checkbox.setChecked(False)
            self.matteformer_checkbox.blockSignals(False)

        img_label.points = []
        img_label.input_box = None

        img_label.auto_predict_mask()

    @Slot()
    def _toggle_workshop_left_panel_animated(self):
        """Creative Workshop: Left asset library collapse and expand animation."""
        current_width = self.left_sidebar.width()
        is_collapsed = current_width < 100

        if is_collapsed:
            # Expand left panel
            target_width = 260
            self.workshop_collapse_btn.setText("＜")
            self.workshop_left_spacer.hide()
            self.asset_library_container.show()
            self.workshop_title_label.show()
        else:
            # Collapse left panel (leave enough width for home and collapse buttons)
            target_width = 80
            self.workshop_collapse_btn.setText("＞")
            self.asset_library_container.hide()
            self.workshop_title_label.hide()
            self.workshop_left_spacer.show()

        # Start smooth width tween animation
        anim_panel_min = QPropertyAnimation(self.left_sidebar, b"minimumWidth")
        anim_panel_max = QPropertyAnimation(self.left_sidebar, b"maximumWidth")

        self._workshop_panel_anim = QParallelAnimationGroup(self)
        for anim in [anim_panel_min, anim_panel_max]:
            anim.setDuration(350)
            anim.setEasingCurve(QEasingCurve.Type.OutQuart)
            anim.setStartValue(current_width)
            anim.setEndValue(target_width)
            self._workshop_panel_anim.addAnimation(anim)

        self._workshop_panel_anim.start()

    def setup_creative_workshop_page(self):
        self.creative_workshop_page = QWidget()
        self.creative_workshop_page.setObjectName("CreativeWorkshopContainer")
        # Overall canvas background is extremely dark (the bottommost background to eliminate color difference).
        self.creative_workshop_page.setStyleSheet("QWidget#CreativeWorkshopContainer { background-color: #181818; }")

        # Compatibility layer (hide old version buttons).
        self.workshop_asset_library_button = QToolButton()
        self.workshop_asset_library_button.hide()
        self.workshop_item_tools_button = QToolButton()
        self.workshop_item_tools_button.hide()
        self.workshop_canvas_settings_button = QToolButton()
        self.workshop_canvas_settings_button.hide()
        self.workshop_layers_button = QToolButton()
        self.workshop_layers_button.hide()
        self.seg_settings_button = QToolButton()
        self.seg_settings_button.hide()
        self.seg_refine_button = QToolButton()
        self.seg_refine_button.hide()
        self.seg_color_button = QToolButton()
        self.seg_color_button.hide()
        self.seg_preview_tool = QToolButton()
        self.seg_preview_tool.hide()
        self.seg_reset_view_tool = QToolButton()
        self.seg_reset_view_tool.hide()

        page_main_layout = QHBoxLayout(self.creative_workshop_page)
        page_main_layout.setContentsMargins(0, 0, 0, 0)
        # Add 16px spacing to allow the middle canvas to automatically have breathing room from the left and right panels.
        page_main_layout.setSpacing(16)

        # ==========================================
        # 1. Left resource bar (transparent background).
        # ==========================================
        self.left_sidebar = QWidget()
        # [Modification]: Cancel the fixed setFixedWidth, change to animatable max/min width.
        self.left_sidebar.setMinimumWidth(260)
        self.left_sidebar.setMaximumWidth(260)
        self.left_sidebar.setStyleSheet("background-color: transparent;")

        left_layout = QVBoxLayout(self.left_sidebar)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Top: Home button & Title & Collapse button.
        left_top_bar = QWidget()
        left_top_bar.setFixedHeight(50)
        left_top_layout = QHBoxLayout(left_top_bar)
        left_top_layout.setContentsMargins(12, 0, 8, 0)

        self.home_button_stitch_page = QToolButton()
        self.home_button_stitch_page.setIcon(self._create_svg_icon("house.svg", size=20, color=QColor("#E0E0E0")))
        # [Modification]: Reduce padding to ensure the home button and collapse button can fit side by side after collapsing.
        self.home_button_stitch_page.setStyleSheet("""
                    QToolButton { border: none; background: transparent; border-radius: 6px; padding: 6px; }
                    QToolButton:hover { background-color: #333333; }
                """)
        self.home_button_stitch_page.clicked.connect(lambda: self.switch_page_with_slide(self.WELCOME_PAGE_INDEX))

        self.workshop_title_label = QLabel("创意工坊")
        self.workshop_title_label.setStyleSheet("color: #FFFFFF; font-weight: bold; font-size: 15px; border: none;")

        # [Addition]: Collapse/Expand button.
        self.workshop_collapse_btn = QToolButton()
        self.workshop_collapse_btn.setText("＜")
        self.workshop_collapse_btn.setFixedSize(28, 28)
        self.workshop_collapse_btn.setStyleSheet(
            "color: #1A73E8; font-size: 16px; font-weight: bold; border-radius: 4px; background-color: transparent;")
        self.workshop_collapse_btn.clicked.connect(self._toggle_workshop_left_panel_animated)

        left_top_layout.addWidget(self.home_button_stitch_page)
        left_top_layout.addSpacing(4)
        left_top_layout.addWidget(self.workshop_title_label)
        left_top_layout.addStretch()
        left_top_layout.addWidget(self.workshop_collapse_btn)
        left_layout.addWidget(left_top_bar)

        # Left asset container.
        self.asset_library_container = QWidget()
        self.asset_library_layout = QVBoxLayout(self.asset_library_container)
        self.asset_library_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.asset_library_container, 1)

        # [Addition]: Spacer used to push the top layout up after collapsing, preventing buttons from falling to the middle of the screen.
        self.workshop_left_spacer = QWidget()
        self.workshop_left_spacer.hide()
        left_layout.addWidget(self.workshop_left_spacer, 1)

        page_main_layout.addWidget(self.left_sidebar)

        # ==========================================
        # 2. Central canvas area.
        # ==========================================
        self.center_workspace = QWidget()
        self.center_workspace.setStyleSheet("background-color: transparent;")
        center_layout = QVBoxLayout(self.center_workspace)
        # Leave 16px spacing at the top and bottom so the canvas isn't stretched to the edges, creating a "floating workbench" feel.
        center_layout.setContentsMargins(0, 16, 0, 16)

        self.workspace_stack = QStackedWidget()
        self.stitching_canvas = StitchingCanvas(parent=self.workspace_stack)
        self.stitching_canvas.main_app_window = self
        self.workspace_stack.addWidget(self.stitching_canvas)

        self.segmentation_overlay_label = ImageLabel(main_window_ref=self, parent=self.workspace_stack)
        self.workspace_stack.addWidget(self.segmentation_overlay_label)

        center_layout.addWidget(self.workspace_stack)
        page_main_layout.addWidget(self.center_workspace, 1)

        # ==========================================
        # 3. Bottom pill toolbar.
        # ==========================================
        self.bottom_pill_toolbar = FloatingPillToolbar(self.center_workspace)

        # [Core Modification]: Icon size increased from 20 to 26.
        icon_size = 26

        self.load_stitch_image_button = QToolButton()
        self.load_stitch_image_button.setIcon(
            self._create_svg_icon("file-earmark-plus.svg", size=icon_size, color=QColor("#E0E0E0")))
        self.load_stitch_image_button.clicked.connect(self.load_stitch_image_action)

        self.segment_selected_button = QToolButton()
        self.segment_selected_button.setIcon(
            self._create_svg_icon("person-bounding-box.svg", size=icon_size, color=QColor("#E0E0E0")))
        self.segment_selected_button.clicked.connect(
            lambda: self._enter_segmentation_mode(self.stitching_canvas.get_primary_selected_item()))

        self.workshop_enhance_button = QToolButton()
        self.workshop_enhance_button.setIcon(
            self._create_svg_icon("badge-4k.svg", size=icon_size, color=QColor("#E0E0E0")))

        self.batch_seg_workshop_btn = QToolButton()
        self.batch_seg_workshop_btn.setIcon(
            self._create_svg_icon("images.svg", size=icon_size, color=QColor("#E0E0E0")))
        self.batch_seg_workshop_btn.setToolTip("跳转批量抠图模式")
        self.batch_seg_workshop_btn.clicked.connect(
            lambda: self.switch_page(getattr(self, 'BATCH_MATTING_INDEX', 3))
        )

        self.clear_stitching_canvas_button = QToolButton()
        self.clear_stitching_canvas_button.setIcon(
            self._create_svg_icon("trash.svg", size=icon_size, color=QColor("#E0E0E0")))
        self.clear_stitching_canvas_button.clicked.connect(self.stitching_canvas.delete_selected_item)

        self.save_selected_item_button = QToolButton()
        self.save_selected_item_button.setIcon(
            self._create_svg_icon("download.svg", size=icon_size, color=QColor("#E0E0E0")))
        self.save_selected_item_button.clicked.connect(self.save_selected_stitched_item)

        self.save_stitched_button = QToolButton()
        self.save_stitched_button.setIcon(
            self._create_svg_icon("clipboard.svg", size=icon_size, color=QColor("#E0E0E0")))
        self.save_stitched_button.clicked.connect(self.save_stitched_image)

        # [Exclusive CSS]: Increase padding to make the button larger, thereby supporting the entire bottom pill black frame.
        BIG_PILL_BUTTON_STYLE = """
                    QToolButton {
                        border: none;
                        background-color: transparent;
                        border-radius: 12px; 
                        padding: 10px 16px; 
                    }
                    QToolButton:hover {
                        background-color: rgba(255, 255, 255, 0.1); 
                    }
                    QToolButton:pressed {
                        background-color: rgba(255, 255, 255, 0.15);
                    }
                """

        for btn in [self.load_stitch_image_button,
                    self.segment_selected_button,
                    self.batch_seg_workshop_btn,
                    self.workshop_enhance_button,
                    self.clear_stitching_canvas_button,
                    self.bottom_pill_toolbar.add_separator(),
                    self.save_selected_item_button,
                    self.save_stitched_button]:
            if isinstance(btn, QToolButton):
                btn.setIconSize(QSize(icon_size, icon_size))
                btn.setStyleSheet(BIG_PILL_BUTTON_STYLE)
                self.bottom_pill_toolbar.add_widget(btn)

        def reposition_pill():
            w = self.bottom_pill_toolbar.sizeHint().width()
            h = self.bottom_pill_toolbar.sizeHint().height()
            x = (self.center_workspace.width() - w) // 2
            y = self.center_workspace.height() - h - 30
            self.bottom_pill_toolbar.setGeometry(x, y, w, h)

        self.center_workspace.resizeEvent = lambda e: (QWidget.resizeEvent(self.center_workspace, e), reposition_pill())

        # ==========================================
        # 4. Right property bar (transparent background).
        # ==========================================
        self.right_sidebar = QWidget()
        self.right_sidebar.setFixedWidth(320)
        self.right_sidebar.setStyleSheet("background-color: transparent;")

        right_layout = QVBoxLayout(self.right_sidebar)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.right_properties_stack = QStackedWidget()
        right_layout.addWidget(self.right_properties_stack)
        page_main_layout.addWidget(self.right_sidebar)

        self.stacked_widget.addWidget(self.creative_workshop_page)

        self.workshop_main_top_bar = QWidget()
        self.workshop_main_top_bar.hide()
        self.segmentation_overlay_top_bar = QWidget()
        self.segmentation_overlay_top_bar.hide()
        self.segmentation_tool_palette = QWidget()
        self.segmentation_tool_palette.hide()

        self._populate_modern_panels()

        self.workshop_enhance_button.clicked.connect(
            lambda: self.right_properties_stack.setCurrentWidget(self.enhance_props_widget))
        self.stitching_canvas.set_canvas_size(STITCHING_DEFAULT_CANVAS_WIDTH, STITCHING_DEFAULT_CANVAS_HEIGHT)
        self.stitching_canvas.set_background_color(QColor(Qt.GlobalColor.transparent))
        QTimer.singleShot(0, self._connect_dynamic_panel_signals)

    def _populate_modern_panels(self):
        """Stuff all sub-panels into the fixed layouts on the left and right, and force them to display."""

        # ==========================================
        # 1. Fill the clean left asset library.
        # ==========================================
        dummy_asset_frame = ui.components.panels.AssetPanelFrame(self)
        self._populate_asset_library_panel(dummy_asset_frame)

        # Extract the built clean content.
        asset_content = dummy_asset_frame.inner_frame
        asset_content.setStyleSheet("background: transparent; border: none;")
        self.asset_library_layout.addWidget(asset_content)

        # ==========================================
        # 2. Build the right canvas property stack (Canvas + Selected Item + Layers).
        # ==========================================
        self.canvas_props_widget = QWidget()
        canvas_props_layout = QVBoxLayout(self.canvas_props_widget)
        canvas_props_layout.setContentsMargins(16, 16, 16, 16)
        canvas_props_layout.setSpacing(16)

        self._build_stitch_canvas_settings_panel(canvas_props_layout)
        self._build_workshop_item_tools_panel(canvas_props_layout)
        self._build_stitch_layers_panel(canvas_props_layout)

        self.right_properties_stack.addWidget(self.canvas_props_widget)

        # ==========================================
        # 3. Matting property stack.
        # ==========================================
        self.segment_props_widget = QWidget()
        segment_props_layout = QVBoxLayout(self.segment_props_widget)
        segment_props_layout.setContentsMargins(16, 16, 16, 16)
        segment_props_layout.setSpacing(16)

        seg_tools_layout = QHBoxLayout()

        self.seg_sam_mode_tool = QToolButton()
        self.seg_sam_mode_tool.setIcon(self._create_svg_icon("magic.svg", size=20, color=QColor("#E0E0E0")))
        self.seg_sam_mode_tool.setCheckable(True)
        self.seg_sam_mode_tool.setChecked(True)
        self.seg_sam_mode_tool.setToolTip("点/框选抠图")

        self.seg_paint_mode_tool = QToolButton()
        self.seg_paint_mode_tool.setIcon(self._create_svg_icon("brush.svg", size=20, color=QColor("#E0E0E0")))
        self.seg_paint_mode_tool.setCheckable(True)
        self.seg_paint_mode_tool.setToolTip("画笔修补")

        seg_mode_group = QButtonGroup(self)
        seg_mode_group.setExclusive(True)
        seg_mode_group.addButton(self.seg_sam_mode_tool)
        seg_mode_group.addButton(self.seg_paint_mode_tool)

        self.seg_auto_matting_tool = QToolButton()
        self.seg_auto_matting_tool.setIcon(
            self._create_svg_icon("person-bounding-box.svg", size=20, color=QColor("#1A73E8")))
        self.seg_auto_matting_tool.setToolTip("一键智能主体抠图")
        self.seg_auto_matting_tool.clicked.connect(self.auto_segment_action)

        self.seg_undo_tool = QToolButton()
        self.seg_undo_tool.setIcon(self._create_svg_icon("arrow-90deg-left.svg", size=20, color=QColor("#E0E0E0")))
        self.seg_undo_tool.setToolTip("撤销")

        self.seg_redo_tool = QToolButton()
        self.seg_redo_tool.setIcon(self._create_svg_icon("arrow-90deg-right.svg", size=20, color=QColor("#E0E0E0")))
        self.seg_redo_tool.setToolTip("重做")

        for btn in [self.seg_sam_mode_tool, self.seg_paint_mode_tool, self.seg_auto_matting_tool, self.seg_undo_tool,
                    self.seg_redo_tool]:
            btn.setStyleSheet(MODERN_TOP_BAR_BUTTON_STYLE)
            seg_tools_layout.addWidget(btn)

        seg_tools_layout.addStretch(1)
        segment_props_layout.addLayout(seg_tools_layout)

        self._build_segment_overlay_settings_panel(segment_props_layout)
        self._build_segment_overlay_refine_panel(segment_props_layout)

        for btn in self.segment_props_widget.findChildren(QPushButton):
            if "一键" in btn.text() or "自动" in btn.text():
                btn.hide()

        segment_props_layout.addStretch(1)

        exit_seg_layout = QHBoxLayout()
        self.apply_segmentation_button = QPushButton("完成并返回")
        self.apply_segmentation_button.setStyleSheet(
            MODERN_PUSHBUTTON_STYLE + "QPushButton{background:#1A73E8; color:white;}")
        self.cancel_segmentation_button = QPushButton("取消")
        self.cancel_segmentation_button.setStyleSheet(MODERN_PUSHBUTTON_STYLE)
        exit_seg_layout.addWidget(self.cancel_segmentation_button)
        exit_seg_layout.addWidget(self.apply_segmentation_button)
        segment_props_layout.addLayout(exit_seg_layout)

        self.right_properties_stack.addWidget(self.segment_props_widget)

        # ==========================================
        # 4. High-definition upscaling stack.
        # ==========================================
        self.enhance_props_widget = QWidget()
        enhance_props_layout = QVBoxLayout(self.enhance_props_widget)
        enhance_props_layout.setContentsMargins(16, 16, 16, 16)
        enhance_props_layout.setSpacing(16)

        self._build_stitch_enhance_panel(enhance_props_layout)

        self.back_to_canvas_btn = QPushButton("返回画板")
        self.back_to_canvas_btn.setStyleSheet(MODERN_PUSHBUTTON_STYLE)
        self.back_to_canvas_btn.clicked.connect(
            lambda: self.right_properties_stack.setCurrentWidget(self.canvas_props_widget))
        enhance_props_layout.addWidget(self.back_to_canvas_btn)

        self.right_properties_stack.addWidget(self.enhance_props_widget)

    @Slot()
    def _fix_dynamic_panels_style(self):
        for widget in self.findChildren(QWidget):
            if isinstance(widget, QToolButton) or isinstance(widget, QPushButton):
                continue

            panel_key = widget.property("panel_key")
            obj_name = widget.objectName() if widget.objectName() else ""

            is_target_panel = (
                    panel_key in ["stitch_layers", "workshop_item_tools"] or
                    "layer" in obj_name.lower() or
                    "property" in obj_name.lower() or
                    "item_tools" in obj_name.lower()
            )

            if is_target_panel:
                widget.setMinimumWidth(320)
                for child in widget.findChildren(QWidget):
                    if (child.inherits("QListWidget") or child.inherits("QScrollArea") or
                            child.inherits("QTreeView") or child.inherits("QTableWidget")):
                        child.setMinimumWidth(300)

    @Slot(bool)
    def toggle_segment_compare_mode(self, checked: bool):
        if hasattr(self, 'segment_label') and self.segment_label:
            self.segment_label.set_compare_mode(checked)
            self.update_button_states()

    @Slot(int)
    def update_segment_split_ratio(self, value: int):
        if hasattr(self, 'segment_label') and self.segment_label:
            ratio = value / 1000.0
            if self.segment_label._split_ratio != ratio:
                self.segment_label._split_ratio = ratio
                self.segment_label.update()

    @Slot(QToolButton, bool, QToolButton)
    def on_interaction_mode_changed_from_tool_custom(self, triggered_button: QToolButton, checked: bool,
                                                     other_button: QToolButton):
        # Compatible with old and new canvas naming.
        img_label = getattr(self, 'segmentation_overlay_label', getattr(self, 'segment_label', None))
        if not img_label:
            return

        # [Core Fix 2]: Removed the code that forced the button to bounce back due to img_label._is_previewing_mask.
        # Release UI-level switching restrictions.

        new_mode = img_label.interaction_mode

        # Compatible with different button naming (the old version might be called sam_mode_top_button, the new one seg_sam_mode_tool).
        sam_btn = getattr(self, 'seg_sam_mode_tool', getattr(self, 'sam_mode_top_button', None))

        if checked:
            new_mode = 'sam' if triggered_button == sam_btn else 'paint'
            if other_button.isChecked():
                other_button.blockSignals(True)
                other_button.setChecked(False)
                other_button.blockSignals(False)
        else:
            if not other_button.isChecked():
                other_button.blockSignals(True)
                other_button.setChecked(True)
                other_button.blockSignals(False)
                new_mode = 'sam' if other_button == sam_btn else 'paint'
            else:
                new_mode = 'sam' if (sam_btn and sam_btn.isChecked()) else 'paint'

        if img_label.interaction_mode != new_mode:
            img_label.set_interaction_mode(new_mode)

        self._update_interaction_specific_controls()
        self.update_button_states()


class DraggableImageLabel(QLabel):
    def __init__(self, parent=None, load_callback=None):
        super().__init__(parent);
        self.setAcceptDrops(True);
        self.load_callback = load_callback
        self.setObjectName("DraggableImageLabel");
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200);
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setText("拖放图像文件至此\n或使用按钮加载");
        self.current_pixmap_orig = None;
        self.setScaledContents(False)
        self.setProperty("acceptingDrops", False)
        self._is_dragging_over = False

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime = event.mimeData()
        if mime.hasUrls():
            supported = tuple(['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.jfif', '.webp', '.gif'])
            urls = mime.urls()
            if any(u.isLocalFile() and u.toLocalFile().lower().endswith(supported) for u in urls):
                event.acceptProposedAction();
                self._is_dragging_over = True
                self.update()
                return
        event.ignore();
        self._is_dragging_over = False
        self.update()

    def dragLeaveEvent(self, event):
        self._is_dragging_over = False
        self.update()
        event.accept()

    def dropEvent(self, event: QDropEvent):
        self._is_dragging_over = False
        self.update()
        urls = event.mimeData().urls();
        supported_formats = (
            '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.jfif', '.webp', '.gif')
        for url in urls:
            if url.isLocalFile():
                fpath = url.toLocalFile()
                if fpath.lower().endswith(supported_formats):
                    if self.load_callback: self.load_callback(fpath)
                    event.acceptProposedAction();
                    return
        event.ignore()

    def clear(self):
        self.current_pixmap_orig = None;
        self.setText("拖放图像文件至此\n或使用按钮加载")
        self.update()

    def setPixmapKeepingAspect(self, pixmap):
        if pixmap and not pixmap.isNull():
            self.current_pixmap_orig = pixmap
            self.setText("")
        else:
            self.clear()
        self.update()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            clip_path = QPainterPath()
            clip_path.addRoundedRect(self.rect(), 16, 16)
            painter.setClipPath(clip_path)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            painter.fillRect(self.rect(), QColor("#1C1C1C"))

            if self.current_pixmap_orig and not self.current_pixmap_orig.isNull():
                pixmap = self.current_pixmap_orig
                scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)

                x = (self.width() - scaled_pixmap.width()) / 2
                y = (self.height() - scaled_pixmap.height()) / 2
                dest_rect = QRect(x, y, scaled_pixmap.width(), scaled_pixmap.height())

                painter.drawPixmap(dest_rect, scaled_pixmap)
            else:
                painter.setPen(QColor("#777777"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())

            if self._is_dragging_over:
                painter.setBrush(QBrush(QColor(26, 115, 232, 40)))
                drag_pen = QPen(QColor(26, 115, 232, 200), 3)
                drag_pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(drag_pen)
                painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 14, 14)

            border_pen = QPen(QColor("#333333"))
            border_pen.setWidthF(2)
            border_pen.setCosmetic(True)
            painter.setPen(border_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            border_rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
            painter.drawRoundedRect(border_rect, 16, 16)

        except Exception as e:
            print(f"DraggableImageLabel paintEvent 错误: {e}")
        finally:
            painter.end()


class ImageLabel(QLabel):
    predict_request = Signal(object, object, object, bool,object)
    refinement_started = Signal(str)
    refinement_finished = Signal()

    def __init__(self, main_window_ref, parent=None):
        if parent is None:
            parent = main_window_ref
        super().__init__(parent)

        self.main_window = main_window_ref
        self.parent_window = self.main_window

        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("SegmentLabel")
        self.setProperty("isPreviewing", False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)
        self._is_dragging_over = False

        self.interaction_mode = 'sam'
        self.points = []
        self.input_box = None
        self.drawing_rect = False
        self.start_pos_widget = None
        self.current_rect_widget = None
        self.painting = False
        self.paint_mode = 'foreground'
        self.brush_size = 15
        self.last_paint_pos_img = None
        self.live_stroke_points = []
        self.live_stroke_mode = 'foreground'
        self.paint_render_mode = 'live'
        self._mask_state_before_paint = None
        self._live_stroke_bbox = QRectF()

        self.mask_history = collections.deque(maxlen=MAX_UNDO_HISTORY)
        self.redo_stack = collections.deque(maxlen=MAX_UNDO_HISTORY)

        self.original_cv_image_full_res = None
        self.original_scale_factor = 1.0
        self.original_cv_image = None
        self.base_pixmap = None
        self.current_mask = None

        self.refine_smooth = DEFAULT_REFINE_SMOOTH
        self.refine_feather = DEFAULT_REFINE_FEATHER
        self.refine_shift = DEFAULT_REFINE_SHIFT
        self.refine_matteformer_enabled = False

        self._cached_refined_mask: Optional[np.ndarray] = None
        self._cached_overlay_pixmap: Optional[QPixmap] = None
        self._cached_cutout_pixmap: Optional[QPixmap] = None

        self._full_image_buffer_edit: Optional[QPixmap] = None
        self._full_image_buffer_preview: Optional[QPixmap] = None

        self._dirty_rect_for_refinement: Optional[QRectF] = None
        self._pending_dirty_rect: Optional[QRectF] = None

        self._stroke_buffer = QPixmap()

        self.zoom_scale = 1.0
        self.view_offset_img = QPointF(0.0, 0.0)
        self._is_panning = False
        self._last_pan_pos_widget = QPoint()
        self._is_previewing_mask = False
        self._checker_pixmap = None
        self.last_mouse_pos_widget = None
        self._is_user_zoomed = False

        self._default_cursor = self.cursor()
        self._cross_cursor = QCursor(Qt.CursorShape.CrossCursor)
        self._paint_cursor = QCursor(Qt.CursorShape.PointingHandCursor)
        self._pan_open_cursor = QCursor(Qt.CursorShape.OpenHandCursor)
        self._pan_closed_cursor = QCursor(Qt.CursorShape.ClosedHandCursor)
        self._wait_cursor = QCursor(Qt.CursorShape.WaitCursor)

        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.clear()

    def set_image(self, cv_image_working, initial_mask=None, original_full_res_image=None, scale_factor=1.0):
        if cv_image_working is None:
            self.clear_all()
            return

        self.original_cv_image = cv_image_working.copy()
        # 【核心修复 1】：增加“纯净工作底图”的终极备份，永远不被算法污染
        self.pure_original_cv_image = cv_image_working.copy()

        if original_full_res_image is not None:
            self.original_cv_image_full_res = original_full_res_image.copy()
            # 【核心修复 1】：增加“纯净高分底图”的终极备份
            self.pure_original_cv_image_full_res = original_full_res_image.copy()
        else:
            self.original_cv_image_full_res = cv_image_working.copy()
            self.pure_original_cv_image_full_res = cv_image_working.copy()

        self.original_scale_factor = scale_factor

        h, w = self.original_cv_image.shape[:2]
        self.base_pixmap = convert_cv_to_pixmap(self.original_cv_image)
        if self.base_pixmap.isNull():
            QMessageBox.critical(self, "内部错误", "无法将OpenCV图像转换为QPixmap。")
            self.clear_all()
            return

        self._clear_all_caches()
        self.clear_selection(clear_image=False)
        self._clear_history()
        self._checker_pixmap = None

        QTimer.singleShot(0, self.parent_window.reset_mask_refinement_controls)

        if initial_mask is not None and initial_mask.shape == (h, w) and initial_mask.dtype == bool:
            self.current_mask = initial_mask.copy()
        else:
            self.current_mask = np.zeros((h, w), dtype=bool)

        self._ensure_mask_exists()

        self._is_user_zoomed = False
        self.fit_image_to_view()
        QTimer.singleShot(0, self.parent_window.update_button_states)

    def set_compare_mode(self, enabled: bool):
        if self._is_in_compare_mode != enabled:
            self._is_in_compare_mode = enabled
            self._buffer_dirty = True
            if enabled:
                self.clear_selection(clear_image=False)
            self.update()
            self.update_cursor()

    @Slot()
    def auto_predict_mask(self):
        if getattr(self.parent_window, 'is_predicting', False) or getattr(self, 'is_auto_matting_active', False):
            self.parent_window.show_status_message("正在处理中，请勿重复点击。")
            return

        if self.original_cv_image is None or not self._ensure_mask_exists():
            self.parent_window.show_status_message("没有可用的图像数据。")
            return

        # =======================================================================
        # 【核心修复 2】：第二次及以后点击一键抠图，强制从纯净备份中复原底片！
        # =======================================================================
        if getattr(self, '_has_auto_matted', False):
            self.parent_window.log_message.emit("🔄 重新触发一键抠图，正在恢复纯净无污染底板...")
            if hasattr(self, 'pure_original_cv_image_full_res') and self.pure_original_cv_image_full_res is not None:
                self.original_cv_image_full_res = self.pure_original_cv_image_full_res.copy()
            if hasattr(self, 'pure_original_cv_image') and self.pure_original_cv_image is not None:
                self.original_cv_image = self.pure_original_cv_image.copy()
                self.base_pixmap = convert_cv_to_pixmap(self.original_cv_image)
        # =======================================================================

        predictor = getattr(self.parent_window, 'image_predictor', None)

        # 此时取出的 img_rgb 保证是 100% 原汁原味的照片
        img_rgb = cv2.cvtColor(self.original_cv_image_full_res, cv2.COLOR_BGR2RGB)

        self.points = []
        self.input_box = None

        self.is_auto_matting_active = True

        is_cumulative = False
        cumulative_cb = getattr(self.parent_window, 'cumulative_sam_checkbox', None)
        if cumulative_cb is not None:
            is_cumulative = cumulative_cb.isChecked()

        app_device = self.parent_window.get_current_device().type if hasattr(self.parent_window,
                                                                             'get_current_device') else "cpu"
        device_str = f"GPU({app_device.upper()})" if app_device in ['cuda', 'mps'] else "CPU"

        self.parent_window.show_status_message(f"正在启动一键自动抠图| 推理设备: {device_str}...", 0)
        self.refinement_started.emit(f"正在一键抠图 ({device_str})")

        QApplication.processEvents()

        try:
            self.predict_request.emit(predictor, (None, None), "AUTO", is_cumulative, img_rgb)
        except TypeError as e:
            self.parent_window.log_message.emit(f"信号发送错误: {e}")
            self.refinement_finished.emit()

        self.update_display()

    def _deferred_auto_predict_execution(self):
        predictor = getattr(self.parent_window, 'image_predictor', None)

        img_rgb = cv2.cvtColor(self.original_cv_image_full_res, cv2.COLOR_BGR2RGB)

        is_cumulative = False
        cumulative_cb = getattr(self.parent_window, 'cumulative_sam_checkbox', None)
        if cumulative_cb is not None:
            is_cumulative = cumulative_cb.isChecked()

        try:
            self.predict_request.emit(predictor, (None, None), "AUTO", is_cumulative, img_rgb)
        except TypeError as e:
            self.parent_window.log_message.emit(f"信号发送错误: {e}")
            self.refinement_finished.emit()

        self.update_display()

    def update_cursor(self):
        is_overlay_mode = getattr(self, 'is_overlay_mode', False)

        if self._is_panning:
            self.setCursor(self._pan_closed_cursor)
            return

        if self._is_previewing_mask:
            self.setCursor(self._default_cursor)
            return

        if self.base_pixmap is None or self.base_pixmap.isNull():
            self.setCursor(self._default_cursor)
            return

        widget_pos = self.mapFromGlobal(QCursor.pos())

        if is_overlay_mode:
            if self.rect().contains(widget_pos):
                if self.interaction_mode == 'sam':
                    self.setCursor(self._cross_cursor)
                elif self.interaction_mode == 'paint':
                    self.setCursor(Qt.CursorShape.BlankCursor)
                else:
                    self.setCursor(self._default_cursor)
            else:
                self.setCursor(self._default_cursor)
            return

        if not self.rect().contains(widget_pos):
            self.setCursor(self._default_cursor)
            return

        cursor_over_img = False
        img_pos_f = self.map_widget_to_image_coords(widget_pos)
        if img_pos_f and self.original_cv_image is not None:
            h, w = self.original_cv_image.shape[:2]
            cursor_over_img = 0 <= img_pos_f.x() < w and 0 <= img_pos_f.y() < h

        new_cursor = self._default_cursor
        if cursor_over_img:
            if self.interaction_mode == 'sam':
                new_cursor = self._cross_cursor
            elif self.interaction_mode == 'paint':
                new_cursor = Qt.CursorShape.BlankCursor
        else:
            new_cursor = self._pan_open_cursor

        if self.interaction_mode == 'sam' and self.drawing_rect:
            new_cursor = self._cross_cursor

        self.setCursor(new_cursor)

    def _check_and_trigger_refinement_after_mask_change(self, immediate=False):
        if not self.parent_window:
            return

        matteformer_checkbox = getattr(self.parent_window, 'matteformer_checkbox', None)
        if matteformer_checkbox and matteformer_checkbox.isChecked():
            dirty_rect = self._dirty_rect_for_refinement
            self._dirty_rect_for_refinement = None

            self.parent_window._pending_refinement_values['dirty_rect'] = dirty_rect

            if immediate:
                if hasattr(self.parent_window, '_refinement_update_timer'):
                    self.parent_window._refinement_update_timer.stop()
                self.parent_window._apply_pending_refinements()
            else:
                self.parent_window._on_refinement_changed()

    def get_cached_refined_mask(self) -> Optional[np.ndarray]:
        if self._cached_refined_mask is None:
            self.parent_window.log_message.emit("应用时发现精炼蒙版缓存无效，正在重新计算一次...")

            if not self._ensure_mask_exists() or self.current_mask is None:
                return None

            current_refine_params = self.get_current_refinement_params_as_dict()

            self._cached_refined_mask = _static_apply_mask_refinements(
                self.current_mask,
                self.original_cv_image,
                current_refine_params,
                main_window_ref=self.parent_window
            )

        return self._cached_refined_mask

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self._checker_pixmap = None
        self._buffer_dirty = True

        if not self._is_user_zoomed or self.base_pixmap is None or self.base_pixmap.isNull():
            self.fit_image_to_view()
            return

        old_size = event.oldSize()
        new_size = self.size()
        if old_size.width() <= 0 or old_size.height() <= 0 or old_size == new_size:
            self.update()
            return

        old_center_widget = QPoint(old_size.width() // 2, old_size.height() // 2)
        center_content_point = self.map_widget_to_image_coords(old_center_widget)
        if center_content_point is None:
            self.fit_image_to_view()
            return

        new_center_widget = QPoint(new_size.width() // 2, new_size.height() // 2)
        new_offset_x = center_content_point.x() - (new_center_widget.x() / self.zoom_scale)
        new_offset_y = center_content_point.y() - (new_center_widget.y() / self.zoom_scale)
        self.view_offset_img = QPointF(new_offset_x, new_offset_y)
        self.update()

    def fit_image_to_view(self):
        if self.base_pixmap is None or self.base_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            self.zoom_scale = 1.0
            self.view_offset_img = QPointF(0, 0)
        else:
            widget_w, widget_h = self.width(), self.height()
            image_w, image_h = self.base_pixmap.width(), self.base_pixmap.height()
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
        self._buffer_dirty = True
        self.update()

    def reset_view(self):
        self.fit_image_to_view()

    def wheelEvent(self, event: QWheelEvent):
        if self.base_pixmap is None or self.base_pixmap.isNull():
            event.ignore();
            return
        delta_steps = event.angleDelta().y() / 120.0
        if abs(delta_steps) < 0.1:
            event.ignore();
            return

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
        if abs(new_scale - old_scale) < 1e-6:
            event.ignore();
            return

        self._is_user_zoomed = True
        self.zoom_scale = new_scale
        new_offset_x = img_pos_f_before.x() - (widget_pos.x() / self.zoom_scale)
        new_offset_y = img_pos_f_before.y() - (widget_pos.y() / self.zoom_scale)
        self.view_offset_img = QPointF(new_offset_x, new_offset_y)

        self._buffer_dirty = True
        self.update()
        self.update_cursor()
        event.accept()

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime = event.mimeData()
        if mime.hasUrls():
            supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.jfif', '.webp', '.gif')
            if any(u.isLocalFile() and u.toLocalFile().lower().endswith(supported) for u in mime.urls()):
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
        supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.jfif', '.webp', '.gif')
        for url in event.mimeData().urls():
            if url.isLocalFile():
                fpath = url.toLocalFile()
                if fpath.lower().endswith(supported):
                    if hasattr(self.parent_window, '_load_image_for_segmentation'):
                        if self.parent_window.stacked_widget.currentIndex() == self.parent_window.CREATIVE_WORKSHOP_INDEX:
                            self.parent_window._load_image_for_segmentation(fpath, target_label=self)
                        else:
                            self.parent_window.switch_page(self.parent_window.IMAGE_SEG_PAGE_INDEX)
                            QApplication.processEvents()
                            self.parent_window._load_image_for_segmentation(fpath)
                    event.acceptProposedAction()
                    return
        event.ignore()


    def set_paint_render_mode(self, mode):
        if mode in ['live', 'precise'] and self.paint_render_mode != mode:
            self.paint_render_mode = mode

            if self.painting:
                self.painting = False
                self.live_stroke_points = []
                if not self._stroke_buffer.isNull():
                    self._stroke_buffer.fill(Qt.transparent)

                if self._mask_state_before_paint is not None:
                    self.current_mask = self._mask_state_before_paint
                    self._mask_state_before_paint = None

                self._clear_all_caches()
                self.update_display()

            self.update()

    def clear_selection(self, clear_image=False):
        self.points = []
        self.input_box = None
        self.drawing_rect = False
        self.start_pos_widget = None
        self.current_rect_widget = None
        self.painting = False
        self.last_paint_pos_img = None
        self.live_stroke_points = []
        self._mask_state_before_paint = None

        self.is_auto_matting_active = False
        self._has_auto_matted = False

        self._clear_all_caches()

        if clear_image:
            self.original_cv_image = None
            self.base_pixmap = None
            self.current_mask = None
            self.original_cv_image_full_res = None
            self.original_scale_factor = 1.0
            self._clear_history()
            self._is_previewing_mask = False
            self._checker_pixmap = None
            self._is_user_zoomed = False

            if hasattr(self.main_window, 'reset_refinement_sliders'):
                QTimer.singleShot(0, self.main_window.reset_refinement_sliders)

            self.fit_image_to_view()
            self.clear()
            self.update_cursor()

            if hasattr(self.main_window, 'cumulative_sam_checkbox'):
                QTimer.singleShot(0, lambda: self.main_window.cumulative_sam_checkbox.setChecked(False))

        elif self.original_cv_image is not None and not self._is_previewing_mask:
            self.update_display()

        if hasattr(self.main_window, 'update_button_states'):
            QTimer.singleShot(0, self.main_window.update_button_states)

    def clear_all(self):
        self.clear_selection(clear_image=True)

    def clear(self):
        if self.original_cv_image is None:
            self.setText("将显示高清后图像")
        else:
            self.setText("")
        self._buffer_dirty = True
        self.update()

    def reset_mask_and_history(self):
        if self.original_cv_image is None or self.original_cv_image_full_res is None:
            self.clear_all()
            return True

        try:
            self.clear_selection(clear_image=False)

            # =======================================================================
            # 【核心修复 3】：当用户点击“清空/重置”图标时，同样恢复被消染破坏的原图
            # =======================================================================
            if hasattr(self, 'pure_original_cv_image_full_res') and self.pure_original_cv_image_full_res is not None:
                self.original_cv_image_full_res = self.pure_original_cv_image_full_res.copy()
            if hasattr(self, 'pure_original_cv_image') and self.pure_original_cv_image is not None:
                self.original_cv_image = self.pure_original_cv_image.copy()
                self.base_pixmap = convert_cv_to_pixmap(self.original_cv_image)
            # =======================================================================

            h, w = self.original_cv_image.shape[:2]
            initial_mask_state = None
            if len(self.original_cv_image_full_res.shape) == 3 and self.original_cv_image_full_res.shape[2] == 4:
                alpha_channel_full_res = self.original_cv_image_full_res[:, :, 3]
                if np.any(alpha_channel_full_res > 1):
                    initial_mask_full_res = (alpha_channel_full_res > 1)
                    initial_mask_state = cv2.resize(initial_mask_full_res.astype(np.uint8), (w, h),
                                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    initial_mask_state = np.zeros((h, w), dtype=bool)
            else:
                initial_mask_state = np.zeros((h, w), dtype=bool)

            self.current_mask = initial_mask_state.copy()
            self._clear_history()

            self.mask_history.append((self.current_mask.copy(), None))
            self.update_display()
            return True
        except Exception as e:
            traceback.print_exc()
            self.clear_all()
            return False

    def _clear_history(self):
        self.mask_history.clear()
        self.redo_stack.clear()
        self._clear_all_caches()

    def _push_mask_history(self, previous_mask_state):
        if self._is_previewing_mask: return

        last_bool_mask = self.mask_history[-1][0] if (
                    len(self.mask_history) > 0 and isinstance(self.mask_history[-1], tuple)) else (
            self.mask_history[-1] if len(self.mask_history) > 0 else None)
        if last_bool_mask is not None and np.array_equal(previous_mask_state, last_bool_mask):
            return

        float_state = self._cached_refined_mask.copy() if getattr(self, '_cached_refined_mask',
                                                                  None) is not None else None
        self.mask_history.append((previous_mask_state, float_state))

        if self.redo_stack:
            self.redo_stack.clear()

        self._clear_visual_caches()

    def undo_last_action(self):
        if self._is_previewing_mask or len(self.mask_history) <= 1: return
        try:
            if self.painting:
                self.painting = False
                self.live_stroke_points = []
                self._mask_state_before_paint = None

            float_state = self._cached_refined_mask.copy() if getattr(self, '_cached_refined_mask', None) is not None else None
            self.redo_stack.append((self.current_mask.copy(), float_state))
            self.mask_history.pop()

            last_history = self.mask_history[-1]
            if isinstance(last_history, tuple):
                self.current_mask = last_history[0].copy()
                self._cached_refined_mask = last_history[1].copy() if last_history[1] is not None else None
            else:
                self.current_mask = last_history.copy()
                self._cached_refined_mask = None

            self._clear_visual_caches()
            self.clear_selection(clear_image=False)
            self.update_display()
        except Exception as e:
            traceback.print_exc()
        finally:
            if self.parent_window:
                QTimer.singleShot(0, self.parent_window.update_button_states)

    def redo_last_action(self):
        if self._is_previewing_mask or not self.redo_stack: return
        try:
            if self.painting:
                self.painting = False
                self.live_stroke_points = []
                self._mask_state_before_paint = None

            state_to_restore = self.redo_stack.pop()

            float_state = self._cached_refined_mask.copy() if getattr(self, '_cached_refined_mask', None) is not None else None
            self.mask_history.append((self.current_mask.copy(), float_state))

            if isinstance(state_to_restore, tuple):
                self.current_mask = state_to_restore[0].copy()
                self._cached_refined_mask = state_to_restore[1].copy() if state_to_restore[1] is not None else None
            else:
                self.current_mask = state_to_restore.copy()
                self._cached_refined_mask = None

            self._clear_visual_caches()
            self.clear_selection(clear_image=False)
            self.update_display()
        except Exception as e:
            traceback.print_exc()
        finally:
            if self.parent_window:
                QTimer.singleShot(0, self.parent_window.update_button_states)

    def map_widget_to_image_coords(self, widget_pos: QPoint):
        if self.base_pixmap is None or self.base_pixmap.isNull() or abs(self.zoom_scale) < 1e-6: return None
        return QPointF((widget_pos.x() / self.zoom_scale) + self.view_offset_img.x(),
                       (widget_pos.y() / self.zoom_scale) + self.view_offset_img.y())

    def map_image_to_widget_coords(self, image_pos_f: QPointF):
        if self.base_pixmap is None or self.base_pixmap.isNull(): return None
        return QPoint(int(round((image_pos_f.x() - self.view_offset_img.x()) * self.zoom_scale)),
                      int(round((image_pos_f.y() - self.view_offset_img.y()) * self.zoom_scale)))

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            if getattr(self, 'base_pixmap', None) and not self.base_pixmap.isNull():
                self._is_panning = True
                self._last_pan_pos_widget = event.position().toPoint()
                self.setCursor(self._pan_closed_cursor)
                event.accept()
                return
            event.ignore()
            return

        # [Preview mode interception has been completely removed]
        if getattr(self, 'base_pixmap', None) is None or self.base_pixmap.isNull() or getattr(self, '_is_panning', False):
            event.ignore()
            return

        widget_pos = event.position().toPoint()
        image_pos_f = self.map_widget_to_image_coords(widget_pos)
        if image_pos_f is None:
            event.ignore()
            return

        image_coords_int = (int(round(image_pos_f.x())), int(round(image_pos_f.y())))

        if getattr(self, 'interaction_mode', None) == 'paint':
            if not self._ensure_mask_exists():
                event.ignore()
                return

            self._mask_state_before_paint = self.current_mask.copy()
            self.painting = True
            self.paint_mode = 'foreground' if event.button() == Qt.MouseButton.LeftButton else 'background'
            self.last_paint_pos_img = image_coords_int

            self._live_stroke_bbox = QRectF()

            if getattr(self, 'points', None) or getattr(self, 'input_box', None) is not None:
                self.points = []
                self.input_box = None

            if getattr(self, 'paint_render_mode', None) == 'live':
                self.live_stroke_mode = self.paint_mode
                if self._stroke_buffer.size() != self.base_pixmap.size():
                    self._stroke_buffer = QPixmap(self.base_pixmap.size())
                self._stroke_buffer.fill(Qt.transparent)
                p = QPainter(self._stroke_buffer)
                self._render_stroke_segment(p, image_coords_int, image_coords_int, is_point=True)
                p.end()
                self.update()
            else:
                if self.paint_on_mask(image_coords_int):
                    self.update_display()
            event.accept()
            return

        elif getattr(self, 'interaction_mode', None) == 'sam':
            self.handle_sam_press(event, image_coords_int)
            event.accept()
            return

        event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent):
        current_pos_widget = event.position().toPoint()
        self.last_mouse_pos_widget = current_pos_widget

        if self._is_panning and (event.buttons() & Qt.MouseButton.MiddleButton):
            delta = current_pos_widget - self._last_pan_pos_widget
            if not delta.isNull():
                self.view_offset_img -= QPointF(delta.x() / self.zoom_scale, delta.y() / self.zoom_scale)
                self._last_pan_pos_widget = current_pos_widget
                self._buffer_dirty = True
                self.update()
            event.accept()
            return

        if self.interaction_mode == 'paint':
            if self.painting:
                image_pos_f = self.map_widget_to_image_coords(current_pos_widget)
                if image_pos_f and self.last_paint_pos_img:
                    current_pos_img = (int(round(image_pos_f.x())), int(round(image_pos_f.y())))

                    if hasattr(self, '_live_stroke_bbox'):
                        point_to_add = QPointF(current_pos_img[0], current_pos_img[1])
                        if self._live_stroke_bbox.isNull():
                            self._live_stroke_bbox = QRectF(point_to_add, QSizeF(1, 1))
                        else:
                            self._live_stroke_bbox = self._live_stroke_bbox.united(QRectF(point_to_add, QSizeF(1, 1)))

                    if self.paint_render_mode == 'live':
                        p = QPainter(self._stroke_buffer)
                        self._render_stroke_segment(p, self.last_paint_pos_img, current_pos_img)
                        p.end()
                        self.update()
                    else:
                        self.paint_line_on_mask(self.last_paint_pos_img, current_pos_img)
                        self.update_display()
                    self.last_paint_pos_img = current_pos_img
            else:
                self.update()
            event.accept()
            return

        if self.interaction_mode == 'sam' and self.drawing_rect:
            self.handle_sam_move(event)
            event.accept()
            return

        self.update_cursor()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton and self._is_panning:
            self._is_panning = False
            self.update_cursor()
            event.accept()
            return

        # [Completely removed if not self._is_previewing_mask] No longer have or self._is_previewing_mask !!!
        if self.base_pixmap is None or self.base_pixmap.isNull():
            event.ignore()
            return

        if self.interaction_mode == 'paint' and self.painting:
            self.painting = False

            if self.paint_render_mode == 'live':
                self.handle_paint_release_live(event)
            else:
                self.handle_paint_release_precise(event)

            self.last_paint_pos_img = None
            self.update_cursor()
            event.accept()
            return

        if self.interaction_mode == 'sam':
            _, handled_sam, predict_triggered = self.handle_sam_release(event)
            if predict_triggered:
                self.predict_mask()  # <--- This line can finally be executed!
            if handled_sam:
                self.update_cursor()
                event.accept()
                return

        self.update_cursor()
        event.ignore()

    def _render_stroke_segment(self, painter: QPainter, p1_img, p2_img, is_point=False):
        painter.setRenderHint(QPainter.Antialiasing, True)
        stroke_color = self.parent_window.selected_mask_color if self.live_stroke_mode == 'foreground' else QColor(255,
                                                                                                                   0, 0,
                                                                                                                   120)

        stroke_pen = QPen(stroke_color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                          Qt.PenJoinStyle.RoundJoin)
        painter.setPen(stroke_pen)

        if is_point:
            painter.drawPoint(QPointF(p1_img[0], p1_img[1]))
        else:
            painter.drawLine(QPointF(p1_img[0], p1_img[1]), QPointF(p2_img[0], p2_img[1]))

    def enterEvent(self, event: QEvent):
        self.update_cursor()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent):
        self.setCursor(self._default_cursor)
        self.last_mouse_pos_widget = None
        if self.interaction_mode == 'paint' and not self._is_previewing_mask:
            self.update()
        super().leaveEvent(event)

    def handle_sam_press(self, event, image_coords_int):
        handled, causes_change = False, False
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.drawing_rect = True;
                self.start_pos_widget = event.position().toPoint()
                self.current_rect_widget = QRect(self.start_pos_widget, self.start_pos_widget)
                if self.points: causes_change = True; self.points = []
                if self.input_box is not None: causes_change = True; self.input_box = None
                handled = True;
                self._buffer_dirty = True;
                self.update()
            elif image_coords_int:
                if self.input_box is not None: self.input_box = None; causes_change = True
                self.points.append((image_coords_int[0], image_coords_int[1], 1))
                handled, causes_change = True, True
                self._buffer_dirty = True;
                self.update()
        elif event.button() == Qt.MouseButton.RightButton and image_coords_int:
            if self.input_box is not None: self.input_box = None; causes_change = True
            self.points.append((image_coords_int[0], image_coords_int[1], 0))
            handled, causes_change = True, True
            self._buffer_dirty = True;
            self.update()
        return handled, causes_change

    def handle_sam_move(self, event):
        if self.drawing_rect and self.start_pos_widget and (event.buttons() & Qt.MouseButton.LeftButton):
            self.current_rect_widget = QRect(self.start_pos_widget, event.position().toPoint()).normalized()
            self.update_display()

    def handle_sam_release(self, event):
        update_needed, handled, predict_triggered = False, False, False
        if self.drawing_rect and event.button() == Qt.MouseButton.LeftButton:
            handled = True
            final_rect_widget = QRect(self.start_pos_widget, event.position().toPoint()).normalized()
            if final_rect_widget.isValid() and final_rect_widget.width() > 2 and final_rect_widget.height() > 2:
                tl_f, br_f = self.map_widget_to_image_coords(
                    final_rect_widget.topLeft()), self.map_widget_to_image_coords(final_rect_widget.bottomRight())
                if tl_f and br_f and self.original_cv_image is not None:
                    x1, y1, x2, y2 = int(round(tl_f.x())), int(round(tl_f.y())), int(round(br_f.x())), int(
                        round(br_f.y()))
                    h, w = self.original_cv_image.shape[:2]
                    img_box = [max(0, min(x1, x2)), max(0, min(y1, y2)), min(w - 1, max(x1, x2)),
                               min(h - 1, max(y1, y2))]
                    if img_box[2] > img_box[0] and img_box[3] > img_box[1]:
                        self.input_box, self.points = np.array([img_box]), []
                        predict_triggered = True
                        previous_mask_state_for_history = self.current_mask.copy()
                        self._push_mask_history(previous_mask_state_for_history)
            self.drawing_rect, self.start_pos_widget, self.current_rect_widget = False, None, None
            self._buffer_dirty = True;
            self.update_display()
        elif event.button() in [Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton] and self.points:
            predict_triggered, handled = True, True
            previous_mask_state_for_history = self.current_mask.copy()
            self._push_mask_history(previous_mask_state_for_history)
        return update_needed, handled, predict_triggered

    def handle_paint_press_live(self, event, image_coords_int):
        if not self._ensure_mask_exists(): return False, False
        handled, causes_change_to_selection = False, False

        if self.points or self.input_box is not None:
            self.points, self.input_box = [], None
            causes_change_to_selection = True
            self.update_display()

        current_paint_mode = None
        if event.button() == Qt.MouseButton.LeftButton:
            current_paint_mode = 'foreground'
        elif event.button() == Qt.MouseButton.RightButton:
            current_paint_mode = 'background'

        if current_paint_mode:
            self.painting = True
            self.paint_mode = current_paint_mode
            self.live_stroke_mode = current_paint_mode
            self.last_paint_pos_img = image_coords_int
            self.live_stroke_points = [image_coords_int]
            if not self._stroke_buffer.isNull():
                self._stroke_buffer.fill(Qt.transparent)
            handled = True
            self.update()

        return handled, causes_change_to_selection

    def handle_paint_release_live(self, event: QMouseEvent):
        button_matches = (event.button() == Qt.MouseButton.LeftButton and self.live_stroke_mode == 'foreground') or \
                         (event.button() == Qt.MouseButton.RightButton and self.live_stroke_mode == 'background')

        if not button_matches:
            if not self._stroke_buffer.isNull():
                self._stroke_buffer.fill(Qt.transparent)
            self._mask_state_before_paint = None
            self.update_display()
            return

        mask_was_changed = False

        if not self._stroke_buffer.isNull() and self._ensure_mask_exists():
            stroke_qimage = self._stroke_buffer.toImage().convertToFormat(QImage.Format_Alpha8)
            ptr = stroke_qimage.constBits()
            bytes_per_line = stroke_qimage.bytesPerLine()
            h, w = stroke_qimage.height(), stroke_qimage.width()

            temp_array = np.array(ptr, copy=True).reshape(h, bytes_per_line)
            stroke_mask_np = np.ascontiguousarray(temp_array[:, :w]) > 0

            if stroke_mask_np.shape != self.current_mask.shape:
                stroke_mask_np = cv2.resize(stroke_mask_np.astype(np.uint8),
                                            (self.current_mask.shape[1], self.current_mask.shape[0]),
                                            interpolation=cv2.INTER_NEAREST).astype(bool)

            if self.live_stroke_mode == 'foreground':
                self.current_mask |= stroke_mask_np
                if getattr(self, '_cached_refined_mask', None) is not None:
                    self._cached_refined_mask[stroke_mask_np] = 1.0
            else:
                self.current_mask &= ~stroke_mask_np
                if getattr(self, '_cached_refined_mask', None) is not None:
                    self._cached_refined_mask[stroke_mask_np] = 0.0

            if self._mask_state_before_paint is not None:
                mask_was_changed = not np.array_equal(self.current_mask, self._mask_state_before_paint)
            else:
                mask_was_changed = True

        self.live_stroke_points = []
        if not self._stroke_buffer.isNull():
            self._stroke_buffer.fill(Qt.transparent)

        if mask_was_changed:
            current_dirty_rect = None
            if hasattr(self, '_live_stroke_bbox') and not self._live_stroke_bbox.isNull():
                brush_radius = self.brush_size / 2.0
                paint_dirty_rect = self._live_stroke_bbox.adjusted(-brush_radius, -brush_radius, brush_radius, brush_radius)
                current_dirty_rect = paint_dirty_rect
                if self._dirty_rect_for_refinement is None:
                    self._dirty_rect_for_refinement = paint_dirty_rect
                else:
                    self._dirty_rect_for_refinement = self._dirty_rect_for_refinement.united(paint_dirty_rect)

            if self._mask_state_before_paint is not None:
                self._push_mask_history(self._mask_state_before_paint)

            self._clear_visual_caches()
            self._fast_draw_then_trigger_refinement(current_dirty_rect)

            if self.parent_window:
                QTimer.singleShot(0, self.parent_window.update_button_states)
        else:
            self.update_display()

        self._mask_state_before_paint = None

    def handle_paint_press_precise(self, event, image_coords_int):
        if not self._ensure_mask_exists(): return False, False
        handled, causes_change_to_mask = False, False

        if self.points or self.input_box is not None:
            self.points = []
            self.input_box = None
            causes_change_to_mask = True
            self.update_display()

        self._mask_state_before_paint = self.current_mask.copy()

        current_paint_mode = 'foreground' if event.button() == Qt.MouseButton.LeftButton else (
            'background' if event.button() == Qt.MouseButton.RightButton else None)

        if current_paint_mode:
            self.painting = True
            self.paint_mode = current_paint_mode
            handled = True
            self.last_paint_pos_img = image_coords_int
            if self.paint_on_mask(image_coords_int):
                causes_change_to_mask = True
            self.update()

        return handled, causes_change_to_mask

    def handle_paint_move_precise(self, event, image_coords_int):
        is_correct_button = (self.paint_mode == 'foreground' and (event.buttons() & Qt.MouseButton.LeftButton)) or \
                            (self.paint_mode == 'background' and (event.buttons() & Qt.MouseButton.RightButton))

        if self.painting and is_correct_button and image_coords_int:
            changed = False
            if self.last_paint_pos_img:
                changed = self.paint_line_on_mask(self.last_paint_pos_img, image_coords_int)
            else:
                changed = self.paint_on_mask(image_coords_int)

            self.last_paint_pos_img = image_coords_int

            if changed:
                self.update_display()

    def handle_paint_release_precise(self, event: QMouseEvent):
        button_matches = (event.button() == Qt.MouseButton.LeftButton and self.paint_mode == 'foreground') or \
                         (event.button() == Qt.MouseButton.RightButton and self.paint_mode == 'background')

        if not button_matches:
            if self._mask_state_before_paint is not None:
                self.current_mask = self._mask_state_before_paint
            self._mask_state_before_paint = None
            self._clear_visual_caches()
            self.update_display()
            return

        mask_was_changed = False
        if self._mask_state_before_paint is not None and self.current_mask is not None:
            if not np.array_equal(self.current_mask, self._mask_state_before_paint):
                mask_was_changed = True

        if mask_was_changed:
            current_dirty_rect = None
            if hasattr(self, '_live_stroke_bbox') and not self._live_stroke_bbox.isNull():
                brush_radius = self.brush_size / 2.0
                paint_dirty_rect = self._live_stroke_bbox.adjusted(-brush_radius, -brush_radius, brush_radius, brush_radius)
                current_dirty_rect = paint_dirty_rect
                if self._dirty_rect_for_refinement is None:
                    self._dirty_rect_for_refinement = paint_dirty_rect
                else:
                    self._dirty_rect_for_refinement = self._dirty_rect_for_refinement.united(paint_dirty_rect)

            if self._mask_state_before_paint is not None:
                self._push_mask_history(self._mask_state_before_paint)

            self._clear_visual_caches()
            self._fast_draw_then_trigger_refinement(current_dirty_rect)

            if self.parent_window:
                QTimer.singleShot(0, self.parent_window.update_button_states)
        else:
            self.update_display()

        self._mask_state_before_paint = None

    def _ensure_mask_exists(self):
        if self.original_cv_image is None: return False
        h, w = self.original_cv_image.shape[:2]
        mask_valid = (self.current_mask is not None and self.current_mask.shape == (
        h, w) and self.current_mask.dtype == bool)

        if not mask_valid:
            self.current_mask = np.zeros((h, w), dtype=bool)
            self._clear_history()
            self.mask_history.append((self.current_mask.copy(), None))
            if self.parent_window: QTimer.singleShot(0, self.parent_window.update_button_states)
            return True
        elif len(self.mask_history) == 0:
            float_state = self._cached_refined_mask.copy() if getattr(self, '_cached_refined_mask',
                                                                      None) is not None else None
            self.mask_history.append((self.current_mask.copy(), float_state))
            if self.parent_window: QTimer.singleShot(0, self.parent_window.update_button_states)
        return True

    def paint_on_mask(self, center_coords_img):
        if not self._ensure_mask_exists() or center_coords_img is None: return False
        x, y = center_coords_img
        radius = max(1, int(round(self.brush_size / 2.0)))
        paint_val = (self.paint_mode == 'foreground')

        ymin, ymax = max(0, y - radius), min(self.current_mask.shape[0], y + radius + 1)
        xmin, xmax = max(0, x - radius), min(self.current_mask.shape[1], x + radius + 1)

        if ymin >= ymax or xmin >= xmax: return False

        yi, xi = np.ogrid[ymin:ymax, xmin:xmax]
        dist_sq = (xi - x) ** 2 + (yi - y) ** 2
        circle_mask_local = dist_sq <= radius ** 2

        self.current_mask[ymin:ymax, xmin:xmax][circle_mask_local] = paint_val

        if getattr(self, '_cached_refined_mask', None) is not None:
            float_val = 1.0 if paint_val else 0.0
            self._cached_refined_mask[ymin:ymax, xmin:xmax][circle_mask_local] = float_val

        self._clear_visual_caches()
        return True

    def paint_line_on_mask(self, start_coords_img, end_coords_img):
        if not self._ensure_mask_exists() or start_coords_img is None or end_coords_img is None: return False
        x1, y1 = start_coords_img
        x2, y2 = end_coords_img
        thick = max(1, int(round(self.brush_size)))
        paint_val = (self.paint_mode == 'foreground')

        stroke_candidate_mask_u8 = np.zeros_like(self.current_mask, dtype=np.uint8)
        cv2.line(stroke_candidate_mask_u8, (x1, y1), (x2, y2), 1, thickness=thick, lineType=cv2.LINE_AA)
        actual_stroke_mask_bool = stroke_candidate_mask_u8.astype(bool)

        self.current_mask[actual_stroke_mask_bool] = paint_val

        if getattr(self, '_cached_refined_mask', None) is not None:
            float_val = 1.0 if paint_val else 0.0
            self._cached_refined_mask[actual_stroke_mask_bool] = float_val

        self._clear_visual_caches()
        return True

    def predict_mask(self):
        if self.parent_window.is_predicting: return
        predictor = getattr(self.parent_window, 'image_predictor', None)
        predictor_ready = predictor is not None and self.parent_window.image_set_in_predictor
        if not (
                predictor_ready and self.original_cv_image is not None and self.interaction_mode == 'sam' and self._ensure_mask_exists()): return

        has_pts, has_box = bool(self.points), self.input_box is not None
        if not has_pts and not has_box: self.update_display(); return

        in_pts, in_labels, in_box = None, None, None
        if has_box:
            in_box = self.input_box
        elif has_pts:
            in_pts = np.array([[p[0], p[1]] for p in self.points], dtype=np.float32)
            in_labels = np.array([p[2] for p in self.points], dtype=np.int32)

        is_cumulative = getattr(self.parent_window, 'cumulative_sam_checkbox', QCheckBox()).isChecked()

        mask_tensor = None
        has_auto_matting = getattr(self, '_has_auto_matted', False) and getattr(self, '_cached_refined_mask',
                                                                                None) is not None

        if not has_auto_matting and is_cumulative and getattr(self, 'current_mask', None) is not None and np.any(
                self.current_mask):
            mask_np_256 = cv2.resize(self.current_mask.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_logits = (mask_np_256 * 40.0) - 20.0
            mask_tensor = torch.from_numpy(mask_logits).unsqueeze(0).unsqueeze(0).to(predictor.device,
                                                                                     dtype=torch.float32)

        self.predict_request.emit(predictor, (in_pts, in_labels), in_box, is_cumulative, mask_tensor)
        self.points = []
        self.input_box = None
        self.update_display()

    def _fast_draw_then_trigger_refinement(self, dirty_rect=None):
        matteformer_cb = getattr(self.parent_window, 'matteformer_checkbox', None)
        if matteformer_cb and matteformer_cb.isChecked():
            self._is_computing_refinement_in_bg = True
            self.update_display(dirty_rect=dirty_rect)

            QApplication.processEvents()

            self._is_computing_refinement_in_bg = False
            self._check_and_trigger_refinement_after_mask_change()
        else:
            self.update_display(dirty_rect=dirty_rect)

    def handle_prediction_result(self, new_mask_from_sam, was_cumulative):
        self.refinement_finished.emit()

        decon_rgb = None
        if isinstance(new_mask_from_sam, tuple):
            new_mask_from_sam, decon_rgb = new_mask_from_sam

        if new_mask_from_sam is None or not self._ensure_mask_exists():
            self.is_auto_matting_active = False
            return

        is_auto = getattr(self, 'is_auto_matting_active', False)

        if is_auto and np.issubdtype(new_mask_from_sam.dtype, np.floating):
            if decon_rgb is not None:
                decon_bgr = cv2.cvtColor(decon_rgb, cv2.COLOR_RGB2BGR)
                self.original_cv_image_full_res = decon_bgr.copy()
                h_work, w_work = self.original_cv_image.shape[:2]
                self.original_cv_image = cv2.resize(decon_bgr, (w_work, h_work), interpolation=cv2.INTER_AREA)
                self.base_pixmap = convert_cv_to_pixmap(self.original_cv_image)

            h_work, w_work = self.original_cv_image.shape[:2]
            if new_mask_from_sam.shape[:2] != (h_work, w_work):
                working_mask = cv2.resize(new_mask_from_sam, (w_work, h_work), interpolation=cv2.INTER_AREA)
            else:
                working_mask = new_mask_from_sam

            bool_mask = working_mask > 0.5
            prev_mask_for_history = self.current_mask.copy()
            self._push_mask_history(prev_mask_for_history)

            self.current_mask = bool_mask
            self._cached_refined_mask = working_mask.copy()
            self._last_valid_refined_mask = self._cached_refined_mask.copy()

            self.is_auto_matting_active = False
            self._has_auto_matted = True

            self.update_display()
            if self.parent_window:
                QTimer.singleShot(0, self.parent_window.update_button_states)
            return

        if new_mask_from_sam.dtype != bool:
            new_mask_from_sam = new_mask_from_sam > 0

        if new_mask_from_sam.shape != self.current_mask.shape:
            return

        prev_mask_for_history = self.current_mask.copy()

        if getattr(self, '_has_auto_matted', False) and getattr(self, '_cached_refined_mask', None) is not None:
            protected_region = self._cached_refined_mask > 0.01
            new_mask_from_sam[protected_region] = False
            self.parent_window.log_message.emit("🛡️ 触发绝对防御：已拦截 SAM2 对现有一键抠图区域的污染。")

        if was_cumulative == "SMART_ERASE":
            potential_new_mask = np.logical_and(self.current_mask, ~new_mask_from_sam)
        else:
            potential_new_mask = np.logical_or(self.current_mask,
                                               new_mask_from_sam) if was_cumulative else new_mask_from_sam

        mask_was_changed = not np.array_equal(self.current_mask, potential_new_mask)

        if mask_was_changed:
            self._push_mask_history(prev_mask_for_history)

            if getattr(self, '_cached_refined_mask', None) is not None:
                added = potential_new_mask & ~self.current_mask
                removed = self.current_mask & ~potential_new_mask
                self._cached_refined_mask[added] = 1.0
                self._cached_refined_mask[removed] = 0.0

            self.current_mask = potential_new_mask

            sam_dirty_rect = None
            rows, cols = np.where(new_mask_from_sam)
            if rows.size > 0:
                x, y, w, h = cv2.boundingRect(np.column_stack((cols, rows)))
                sam_dirty_rect = QRectF(x, y, w, h)
                if getattr(self, '_dirty_rect_for_refinement', None) is None:
                    self._dirty_rect_for_refinement = sam_dirty_rect
                else:
                    self._dirty_rect_for_refinement = self._dirty_rect_for_refinement.united(sam_dirty_rect)

            if hasattr(self, '_clear_visual_caches'):
                self._clear_visual_caches()
            else:
                self._cached_overlay_pixmap = None
                self._cached_cutout_pixmap = None
                self._full_image_buffer_edit = None
                self._full_image_buffer_preview = None

            self.update_display()

            QApplication.processEvents()
            self._check_and_trigger_refinement_after_mask_change(immediate=True)

        else:
            self.update_display()

        self.is_auto_matting_active = False

        if self.parent_window:
            QTimer.singleShot(0, self.parent_window.update_button_states)

    def _clear_visual_caches(self):
        """
        [Physical Fix]: Completely refactor the visual cache clearing logic.
        When the user interacts with the brush or SAM 2, only clear the Pixmap cache used for display rendering.
        [Must Retain] The global high-precision floating-point mask cache (_cached_refined_mask) to achieve true local ROI incremental accumulation calculation.
        """
        # -----------------------------------------------------------------
        # Core crash prevention and incremental optimization: Absolutely DO NOT clear self._cached_refined_mask = None here!
        # This ensures that local calculations can use the previous refinement results for feathering and blending.
        # -----------------------------------------------------------------
        self._cached_overlay_pixmap = None
        self._cached_cutout_pixmap = None
        self._full_image_buffer_edit = None
        self._full_image_buffer_preview = None

        if hasattr(self, '_stroke_buffer') and not self._stroke_buffer.isNull():
            self._stroke_buffer.fill(Qt.GlobalColor.transparent)

    def _clear_all_caches(self):
        if getattr(self, '_cached_refined_mask', None) is not None:
            self._last_valid_refined_mask = self._cached_refined_mask.copy()

        self._cached_refined_mask = None
        if hasattr(self, '_cached_4k_auto_mask'):
            delattr(self, '_cached_4k_auto_mask')

        self._cached_overlay_pixmap = None
        self._cached_cutout_pixmap = None
        self._full_image_buffer_edit = None
        self._full_image_buffer_preview = None

        if hasattr(self, '_stroke_buffer') and not self._stroke_buffer.isNull():
            self._stroke_buffer.fill(Qt.GlobalColor.transparent)

    def get_current_refinement_params_as_dict(self):
        return {
            'refine_smooth': self.refine_smooth,
            'refine_feather': self.refine_feather,
            'refine_shift': self.refine_shift,
            'refine_matteformer_enabled': self.refine_matteformer_enabled
        }

    def set_refinement_params_batch(self, params: dict, dirty_rect: Optional[QRectF] = None):
        if self._is_previewing_mask:
            return False

        changed = False

        for pname, new_val in params.items():
            if hasattr(self, pname):
                curr_val = getattr(self, pname)

                is_different = False
                if isinstance(new_val, float):
                    if not math.isclose(curr_val, new_val, rel_tol=1e-9):
                        is_different = True
                elif curr_val != new_val:
                    is_different = True

                if is_different:
                    setattr(self, pname, new_val)
                    changed = True
            else:
                setattr(self, pname, new_val)
                changed = True

        if changed or dirty_rect is not None:
            if changed:
                self.parent_window.log_message.emit(f"优化参数已改变，准备全局重绘: {params}")
                self._clear_all_caches()
            else:
                self.parent_window.log_message.emit(f"收到新增抠图区域，准备局部精修...")

            self.update_display(dirty_rect=dirty_rect)
            return True

        return False

    def set_mask_preview_mode(self, previewing: bool):
        if self._is_previewing_mask == previewing:
            return

        self.parent_window.log_message.emit(f"切换抠图预览模式 -> {'开启' if previewing else '关闭'}")
        self._is_previewing_mask = previewing

        if previewing:
            self.points, self.input_box, self.drawing_rect, self.painting, self.live_stroke_points = [], None, False, False, []
            if not self._stroke_buffer.isNull():
                self._stroke_buffer.fill(Qt.GlobalColor.transparent)

        self.update()
        self.update_cursor()

    def _create_checkerboard_pixmap(self, size: QSize):
        if size.isEmpty(): return QPixmap()
        if self._checker_pixmap and self._checker_pixmap.size() == size: return self._checker_pixmap
        self._checker_pixmap = create_checkerboard_pixmap(size)
        return self._checker_pixmap

    def _calculate_render_rects(self):
        if not self.base_pixmap or self.base_pixmap.isNull(): return QRect(), QRect()
        widget_rect = self.rect()
        orig_w, orig_h = self.base_pixmap.width(), self.base_pixmap.height()
        view_w_img, view_h_img = widget_rect.width() / self.zoom_scale, widget_rect.height() / self.zoom_scale
        src_rect_f = QRectF(self.view_offset_img.x(), self.view_offset_img.y(), view_w_img, view_h_img)
        src_rect_clip_f = src_rect_f.intersected(QRectF(0, 0, orig_w, orig_h))
        if not src_rect_clip_f.isValid(): return QRect(), QRect()
        dest_rect_f = QRectF(
            (src_rect_clip_f.left() - self.view_offset_img.x()) * self.zoom_scale,
            (src_rect_clip_f.top() - self.view_offset_img.y()) * self.zoom_scale,
            src_rect_clip_f.width() * self.zoom_scale,
            src_rect_clip_f.height() * self.zoom_scale
        )
        return dest_rect_f.toRect(), src_rect_clip_f.toRect()

    def _rebuild_back_buffer(self):
        dirty_rect = getattr(self, '_pending_dirty_rect', None)
        self._pending_dirty_rect = None

        is_full_rebuild = (dirty_rect is None or self._back_buffer.isNull() or self._cached_refined_mask is None)

        if is_full_rebuild:
            self.parent_window.log_message.emit("执行完整重绘...")
            if not self.base_pixmap:
                self._back_buffer = QPixmap(self.size())
                # --- [Dark Mode] Empty background filled with #1C1C1C ---
                self._back_buffer.fill(QColor("#1C1C1C"))
                painter = QPainter(self._back_buffer)
                painter.setPen(QColor("#777777"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "拖放图像文件至此...")
                painter.end()
                self._buffer_dirty = False
                return

            if self._ensure_mask_exists() and self.current_mask is not None:
                current_refine_params = self.get_current_refinement_params_as_dict()
                self._cached_refined_mask = _static_apply_mask_refinements(
                    self.current_mask, self.original_cv_image,
                    current_refine_params,
                    main_window_ref=self.parent_window
                )
            else:
                self._cached_refined_mask = None

            self._cached_overlay_pixmap = None
            self._cached_cutout_pixmap = None
        else:
            self.parent_window.log_message.emit(f"执行局部更新，脏区域: {dirty_rect}")

            padding = 64
            h, w = self.original_cv_image.shape[:2]
            compute_rect = dirty_rect.adjusted(-padding, -padding, padding, padding).toRect()
            compute_rect = compute_rect.intersected(QRect(0, 0, w, h))

            if compute_rect.isEmpty():
                self.parent_window.log_message.emit("计算区域为空，跳过局部更新。")
                self._buffer_dirty = False
                return

            x, y, cw, ch = compute_rect.getRect()
            image_tile = self.original_cv_image[y:y + ch, x:x + cw]
            base_mask_tile = self.current_mask[y:y + ch, x:x + cw]

            current_refine_params = self.get_current_refinement_params_as_dict()
            refined_mask_tile = _static_apply_mask_refinements(
                base_mask_tile,
                image_tile,
                current_refine_params,
                main_window_ref=self.parent_window
            )

            self._cached_refined_mask[y:y + ch, x:x + cw] = refined_mask_tile

            self._cached_overlay_pixmap = None
            self._cached_cutout_pixmap = None

        self._back_buffer = QPixmap(self.size())
        # --- [Dark Mode] Repaint background filled with #1C1C1C ---
        self._back_buffer.fill(QColor("#1C1C1C"))

        painter = QPainter(self._back_buffer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        dest_rect, src_rect_clip = self._calculate_render_rects()
        if not dest_rect.isValid():
            painter.end()
            self._buffer_dirty = False
            return

        if self._is_previewing_mask:
            checker_pixmap = self._create_checkerboard_pixmap(self.base_pixmap.size())
            if not checker_pixmap.isNull():
                painter.drawPixmap(dest_rect, checker_pixmap, src_rect_clip)

            if self._cached_cutout_pixmap is None:
                self._cached_cutout_pixmap = self._generate_cutout_pixmap(self._cached_refined_mask)
            if self._cached_cutout_pixmap and not self._cached_cutout_pixmap.isNull():
                painter.drawPixmap(dest_rect, self._cached_cutout_pixmap, src_rect_clip)
        else:
            painter.drawPixmap(dest_rect, self.base_pixmap, src_rect_clip)

            if self._cached_refined_mask is not None and np.any(self._cached_refined_mask > 0):
                if self._cached_overlay_pixmap is None:
                    self._cached_overlay_pixmap = self._generate_overlay_pixmap(self._cached_refined_mask)
                if self._cached_overlay_pixmap and not self._cached_overlay_pixmap.isNull():
                    painter.drawPixmap(dest_rect, self._cached_overlay_pixmap, src_rect_clip)

            if self.interaction_mode == 'sam' and self.points:
                painter.save()
                transform = QTransform()
                transform.translate(dest_rect.left(), dest_rect.top())
                transform.scale(self.zoom_scale, self.zoom_scale)
                transform.translate(-src_rect_clip.left(), -src_rect_clip.top())
                painter.setTransform(transform)
                pt_rad = max(2., 5. / self.zoom_scale)
                for x_img, y_img, lbl in self.points:
                    pt_color = QColor(34, 177, 76) if lbl == 1 else QColor(237, 28, 36)
                    painter.setPen(QPen(Qt.GlobalColor.black, 1.0 / self.zoom_scale))
                    painter.setBrush(QColor(pt_color.red(), pt_color.green(), pt_color.blue(), 200))
                    painter.drawEllipse(QPointF(x_img, y_img), pt_rad, pt_rad)
                painter.restore()

        painter.end()
        self._buffer_dirty = False

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        active_buffer = None
        if getattr(self, '_is_previewing_mask', False):
            if getattr(self, '_full_image_buffer_preview', None) is None:
                self._rebuild_preview_buffer()
            active_buffer = self._full_image_buffer_preview
        else:
            if getattr(self, '_full_image_buffer_edit', None) is None:
                self._rebuild_edit_buffer()
            active_buffer = self._full_image_buffer_edit

        if active_buffer is None or active_buffer.isNull():
            painter.setPen(QColor("#777777"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
        else:
            dest_rect, src_rect = self._calculate_render_rects()
            if dest_rect.isValid() and src_rect.isValid():
                painter.drawPixmap(dest_rect, active_buffer, src_rect)

            # [Completely removed if not self._is_previewing_mask]
            # Ensure points, bounding boxes, and brush indicators are visible even on transparent images!

            if self.interaction_mode == 'sam' and self.points:
                painter.save()
                transform = QTransform()
                transform.translate(dest_rect.left(), dest_rect.top())
                transform.scale(self.zoom_scale, self.zoom_scale)
                transform.translate(-src_rect.left(), -src_rect.top())
                painter.setTransform(transform)
                pt_rad = max(2., 5. / self.zoom_scale)
                for x_img, y_img, lbl in self.points:
                    pt_color = QColor(34, 177, 76) if lbl == 1 else QColor(237, 28, 36)
                    painter.setPen(QPen(Qt.GlobalColor.black, 1.0 / self.zoom_scale))
                    painter.setBrush(QColor(pt_color.red(), pt_color.green(), pt_color.blue(), 200))
                    painter.drawEllipse(QPointF(x_img, y_img), pt_rad, pt_rad)
                painter.restore()

            if self.interaction_mode == 'paint' and self.last_mouse_pos_widget:
                image_pos_f = self.map_widget_to_image_coords(self.last_mouse_pos_widget)
                if image_pos_f and self.original_cv_image is not None:
                    h, w = self.original_cv_image.shape[:2]
                    if 0 <= image_pos_f.x() < w and 0 <= image_pos_f.y() < h:
                        widget_brush_rad = max(1., (self.brush_size / 2.) * self.zoom_scale)
                        preview_pen = QPen(QColor(79, 70, 229, 200), 1.5)
                        painter.setPen(preview_pen)
                        painter.setBrush(Qt.NoBrush)
                        painter.drawEllipse(QPointF(self.last_mouse_pos_widget), widget_brush_rad, widget_brush_rad)

            if self.interaction_mode == 'sam' and self.drawing_rect and self.current_rect_widget:
                box_pen = QPen(QColor(79, 70, 229), 2.0)
                box_pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(box_pen)
                painter.drawRect(self.current_rect_widget)

            if self.painting and self.paint_render_mode == 'live' and not self._stroke_buffer.isNull():
                painter.drawPixmap(dest_rect, self._stroke_buffer, src_rect)

        if getattr(self, '_is_dragging_over', False):
            painter.setBrush(QBrush(QColor(79, 70, 229, 40)))
            drag_pen = QPen(QColor(79, 70, 229, 200), 3)
            drag_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(drag_pen)
            rect_f = QRectF(self.rect()).adjusted(2, 2, -2, -2)
            painter.drawPath(self._get_rounded_path(rect_f, 10.0))

        painter.end()

    def _get_rounded_path(self, rect: QRectF, radius: float) -> QPainterPath:
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        return path

    def _rebuild_full_image_buffer(self):
        pass

    def update_display(self, dirty_rect: Optional[QRectF] = None):
        self._full_image_buffer_edit = None
        self._full_image_buffer_preview = None
        self._pending_dirty_rect = dirty_rect
        self.update()

    def _ensure_and_get_refined_mask(self):
        """
        Get the refined mask at the current working resolution.
        Fixed: Automatically perform incremental blending to merge historical high-precision soft mask with the current new brush strokes during the asynchronous calculation transition in the main thread. This thoroughly prevents flashing and sudden harsh mask transitions during model calculation.
        """
        is_main_thread = (QThread.currentThread() == QCoreApplication.instance().thread())

        if is_main_thread:
            if getattr(self, '_cached_refined_mask', None) is not None:
                return self._cached_refined_mask

            # Core improvement: When background asynchronous calculation is not finished, use historical high-quality soft mask (_last_valid_refined_mask) to incrementally blend with the latest brush modification area (current_mask), ensuring transition frames remain smooth and free of hard edge aliasing.
            if getattr(self, '_last_valid_refined_mask', None) is not None and getattr(self, 'current_mask', None) is not None:
                if self.current_mask.shape == self._last_valid_refined_mask.shape:
                    temp_mask = self.current_mask.astype(np.float32)
                    overlap = self.current_mask & (self._last_valid_refined_mask > 0.01)
                    temp_mask[overlap] = self._last_valid_refined_mask[overlap]
                    return temp_mask

            return getattr(self, 'current_mask', None)

        # Background sub-thread branch (autonomously rebuilt by the calculation thread).
        dirty_rect = getattr(self, '_pending_dirty_rect', None)
        self._pending_dirty_rect = None

        is_full_rebuild = (self._cached_refined_mask is None) or (dirty_rect is None)

        if is_full_rebuild:
            if self._ensure_mask_exists() and self.current_mask is not None:
                params = self.get_current_refinement_params_as_dict()
                self._cached_refined_mask = _static_apply_mask_refinements(
                    self.current_mask, self.original_cv_image, params, self.parent_window
                )
                self._last_valid_refined_mask = self._cached_refined_mask.copy() if self._cached_refined_mask is not None else None
        else:
            padding = 64
            h, w = self.original_cv_image.shape[:2]
            compute_rect = dirty_rect.adjusted(-padding, -padding, padding, padding).toRect().intersected(
                QRect(0, 0, w, h))

            if not compute_rect.isEmpty():
                x, y, cw, ch = compute_rect.getRect()
                image_tile = self.original_cv_image[y:y + ch, x:x + cw]
                base_mask_tile = self.current_mask[y:y + ch, x:x + cw]
                params = self.get_current_refinement_params_as_dict()

                refined_mask_tile = _static_apply_mask_refinements(
                    base_mask_tile, image_tile, params, self.parent_window
                )

                if refined_mask_tile is not None and refined_mask_tile.shape == (ch, cw):
                    self._cached_refined_mask[y:y + ch, x:x + cw] = refined_mask_tile
                self._last_valid_refined_mask = self._cached_refined_mask.copy() if self._cached_refined_mask is not None else None

        return self._cached_refined_mask

    def _rebuild_edit_buffer(self):
        """
        Rebuild the combined buffer of image and semi-transparent color mask in edit mode.
        Fixed: Completely removed the condition forcing the display of rough binary mask during background calculation, unified to output high-precision progressive soft mask.
        """
        if self.base_pixmap is None:
            self._full_image_buffer_edit = None
            return

        # Core fix: Even if running in the background, fetch the high-quality soft mask after smooth transition blending to ensure the image is not rough.
        refined_mask = self._ensure_and_get_refined_mask()

        self._full_image_buffer_edit = QPixmap(self.base_pixmap.size())
        self._full_image_buffer_edit.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self._full_image_buffer_edit)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        painter.drawPixmap(0, 0, self.base_pixmap)

        if refined_mask is not None and np.any(refined_mask > 0):
            if self._cached_overlay_pixmap is None:
                self._cached_overlay_pixmap = self._generate_overlay_pixmap(refined_mask)
            if self._cached_overlay_pixmap:
                painter.drawPixmap(0, 0, self._cached_overlay_pixmap)

        painter.end()

    def _rebuild_preview_buffer(self):
        """
        Rebuild the matting effect buffer in preview mode (checkerboard transparent background).
        Fixed: Completely removed the condition forcing the display of rough binary mask during background calculation, unified to output high-precision progressive soft mask.
        """
        if self.base_pixmap is None:
            self._full_image_buffer_preview = None
            return

        # Core fix: Even if running in the background, fetch the high-quality soft mask after smooth transition blending to ensure the image is not rough.
        refined_mask = self._ensure_and_get_refined_mask()

        self._full_image_buffer_preview = QPixmap(self.base_pixmap.size())
        self._full_image_buffer_preview.fill(Qt.GlobalColor.transparent)
        painter = QPainter(self._full_image_buffer_preview)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        checker = self._create_checkerboard_pixmap(self.base_pixmap.size())
        painter.drawPixmap(0, 0, checker)

        if refined_mask is not None and np.any(refined_mask > 0):
            if self._cached_cutout_pixmap is None:
                self._cached_cutout_pixmap = self._generate_cutout_pixmap(refined_mask)
            if self._cached_cutout_pixmap:
                painter.drawPixmap(0, 0, self._cached_cutout_pixmap)

        painter.end()

    def _generate_cutout_pixmap(self, refined_mask_param):
        alpha_mask_u8 = None
        if refined_mask_param is None: return QPixmap()
        h_mask, w_mask = refined_mask_param.shape[:2]
        if refined_mask_param.dtype == bool:
            alpha_mask_u8 = refined_mask_param.astype(np.uint8) * 255
        elif refined_mask_param.dtype == np.uint8:
            alpha_mask_u8 = refined_mask_param
        elif np.issubdtype(refined_mask_param.dtype, np.floating):
            alpha_mask_u8 = np.clip(refined_mask_param * 255., 0, 255).astype(np.uint8)
        else:
            return QPixmap()

        if np.any(alpha_mask_u8 > 0) and self.original_cv_image is not None:
            img_src = self.original_cv_image;
            bgr_img_full = None
            try:
                if len(img_src.shape) == 2:
                    bgr_img_full = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR)
                elif len(img_src.shape) == 3 and img_src.shape[2] == 4:
                    bgr_img_full = cv2.cvtColor(img_src, cv2.COLOR_BGRA2BGR)
                elif len(img_src.shape) == 3 and img_src.shape[2] == 3:
                    bgr_img_full = img_src
            except cv2.error as e:
                return QPixmap()
            if bgr_img_full is None: return QPixmap()

            h_img, w_img = bgr_img_full.shape[:2]
            if h_img != h_mask or w_img != w_mask:
                try:
                    alpha_mask_u8 = cv2.resize(alpha_mask_u8, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                except cv2.error:
                    return QPixmap()

            if not bgr_img_full.flags['C_CONTIGUOUS']: bgr_img_full = np.ascontiguousarray(bgr_img_full)
            cutout_bgra = cv2.cvtColor(bgr_img_full, cv2.COLOR_BGR2BGRA);
            cutout_bgra[:, :, 3] = alpha_mask_u8
            return convert_cv_to_pixmap(cutout_bgra)
        else:
            return QPixmap()

    def _generate_overlay_pixmap(self, refined_mask_param):
        alpha_mask_u8 = None
        if refined_mask_param is None: return QPixmap()
        h_mask, w_mask = refined_mask_param.shape[:2]
        if refined_mask_param.dtype == bool:
            alpha_mask_u8 = refined_mask_param.astype(np.uint8) * 255
        elif refined_mask_param.dtype == np.uint8:
            alpha_mask_u8 = refined_mask_param
        elif np.issubdtype(refined_mask_param.dtype, np.floating):
            alpha_mask_u8 = np.clip(refined_mask_param * 255., 0, 255).astype(np.uint8)
        else:
            alpha_mask_u8 = np.zeros((h_mask, w_mask), dtype=np.uint8)

        if np.any(alpha_mask_u8 > 0):
            overlay_cv = np.zeros((h_mask, w_mask, 4), dtype=np.uint8)
            fill_color = self.parent_window.selected_mask_color
            fill_bgr = (fill_color.blue(), fill_color.green(), fill_color.red())
            mask_bool_fill = alpha_mask_u8 > 0
            overlay_cv[mask_bool_fill, 0:3] = fill_bgr
            overlay_cv[:, :, 3] = (alpha_mask_u8.astype(np.float32) / 255. * DEFAULT_MASK_ALPHA_IMAGE).astype(np.uint8)
            return convert_cv_to_pixmap(overlay_cv)
        else:
            return QPixmap()

    def set_interaction_mode(self, mode):
        # [Core Fix 1]: Removed the interception of self._is_previewing_mask.
        # Allow freely switching between SAM / Paint in transparent preview mode.
        if mode not in ['sam', 'paint'] or self.interaction_mode == mode:
            return

        self.interaction_mode = mode

        if mode == 'paint' and (self.points or self.input_box):
            self.points = []
            self.input_box = None
            self.update_display()
        elif mode == 'sam' and self.painting:
            self.painting = False
            self.live_stroke_points = []
            self._mask_state_before_paint = None
            self.update()

        self.update_cursor()

        if hasattr(self, 'parent_window') and self.parent_window:
            QTimer.singleShot(0, self.parent_window.update_button_states)

    def set_brush_size(self, size):
        if self._is_previewing_mask: return
        new_size = max(1, size)
        if new_size != self.brush_size:
            self.brush_size = new_size
            self.update()


class ImageCompareWidget(QWidget):
    """A control that compares two images by dragging a slider, supporting zooming and panning."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap: QPixmap | None = None
        self.enhanced_pixmap: QPixmap | None = None

        self._is_dragging_over = False
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._split_ratio = 0.5
        self._is_dragging_handle = False
        self._triangle_size = 10
        self._triangle_rect = QRect()
        self._slider_visible = False

        self.zoom_scale = 1.0
        self.view_offset = QPointF(0, 0)
        self._is_panning = False
        self._last_pan_pos = QPoint()
        self._is_user_zoomed = False

        self.load_callback = None

    def set_load_callback(self, callback):
        self.load_callback = callback

    def set_images(self, original: QPixmap, enhanced: QPixmap | None = None):
        self.original_pixmap = original
        self.enhanced_pixmap = enhanced
        self.reset_view()

        self._slider_visible = (enhanced is not None)
        if self._slider_visible:
            self._split_ratio = 0.5

        self.update()

    def clear_content(self):
        self.original_pixmap = None
        self.enhanced_pixmap = None
        self.reset_view()
        self.update()

    def get_enhanced_pixmap(self) -> QPixmap | None:
        return self.enhanced_pixmap

    def reset_view(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            padding_factor = 0.98
            scale_w = (self.width() * padding_factor) / self.original_pixmap.width()
            scale_h = (self.height() * padding_factor) / self.original_pixmap.height()
            self.zoom_scale = min(scale_w, scale_h)

            scaled_w = self.original_pixmap.width() * self.zoom_scale
            scaled_h = self.original_pixmap.height() * self.zoom_scale

            self.view_offset = QPointF(
                (self.width() - scaled_w) / 2,
                (self.height() - scaled_h) / 2
            )
        else:
            self.zoom_scale = 1.0
            self.view_offset = QPointF(0, 0)

        self._is_user_zoomed = False
        self.update()

    def map_widget_to_image_coords(self, widget_pos: QPoint) -> QPointF:
        if self.zoom_scale < 1e-6:
            return QPointF()
        return (QPointF(widget_pos) - self.view_offset) / self.zoom_scale

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime = event.mimeData()
        if mime.hasUrls():
            supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.jfif', '.webp', '.gif')
            if any(u.isLocalFile() and u.toLocalFile().lower().endswith(supported) for u in mime.urls()):
                event.acceptProposedAction();
                self._is_dragging_over = True;
                self.update()
                return
        event.ignore();
        self._is_dragging_over = False;
        self.update()

    def dragLeaveEvent(self, event: QEvent):
        self._is_dragging_over = False;
        self.update();
        event.accept()

    def dropEvent(self, event: QDropEvent):
        self._is_dragging_over = False;
        self.update()
        urls = event.mimeData().urls();
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.jfif', '.webp', '.gif')
        for url in urls:
            if url.isLocalFile():
                fpath = url.toLocalFile()
                if fpath.lower().endswith(supported_formats):
                    if self.load_callback: self.load_callback(fpath)
                    event.acceptProposedAction();
                    return
        event.ignore()

    def wheelEvent(self, event: QWheelEvent):
        if not self.original_pixmap:
            event.ignore();
            return

        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1 / 1.15
        old_scale = self.zoom_scale
        new_scale = old_scale * zoom_factor
        self.zoom_scale = max(MIN_ZOOM, min(new_scale, MAX_ZOOM))
        if abs(self.zoom_scale - old_scale) < 1e-6:
            event.ignore();
            return

        self._is_user_zoomed = True
        mouse_pos = event.position()
        self.view_offset = mouse_pos - (mouse_pos - self.view_offset) * (self.zoom_scale / old_scale)
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            if self.original_pixmap and not self.original_pixmap.isNull():
                self._is_panning = True
                self._last_pan_pos = event.position().toPoint()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            event.ignore()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._slider_visible and self._triangle_rect.contains(event.position().toPoint()):
                self._is_dragging_handle = True
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                event.accept()
                return

        event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent):
        current_pos = event.position().toPoint()

        if self._is_panning:
            delta = current_pos - self._last_pan_pos
            self.view_offset += QPointF(delta)
            self._last_pan_pos = current_pos
            self.update()
            return

        if self._is_dragging_handle:
            if not self.original_pixmap: return
            scaled_w = self.original_pixmap.width() * self.zoom_scale
            scaled_h = self.original_pixmap.height() * self.zoom_scale
            display_rect = QRectF(self.view_offset, QSizeF(scaled_w, scaled_h)).toRect()

            if display_rect.width() > 0:
                new_ratio = (current_pos.x() - display_rect.left()) / display_rect.width()
                self._split_ratio = max(0.0, min(1.0, new_ratio))
                self.update()
        else:
            self.update_cursor(current_pos)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
        elif event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging_handle = False

        self.update_cursor(event.position().toPoint())
        super().mouseReleaseEvent(event)

    def update_cursor(self, pos):
        if self._is_panning:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._is_dragging_handle:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self._slider_visible and self._triangle_rect.contains(pos):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self.original_pixmap:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.original_pixmap:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            scaled_w = self.original_pixmap.width() * self.zoom_scale
            scaled_h = self.original_pixmap.height() * self.zoom_scale
            dest_rect = QRectF(self.view_offset, QSizeF(scaled_w, scaled_h))

            painter.drawPixmap(dest_rect, self.original_pixmap, self.original_pixmap.rect())

            if self.enhanced_pixmap and self._slider_visible:
                split_pos_x = dest_rect.left() + dest_rect.width() * self._split_ratio

                source_rect_enh = QRectF(
                    0, 0,
                    self.enhanced_pixmap.width() * self._split_ratio,
                    self.enhanced_pixmap.height()
                )
                target_rect_enh = QRectF(
                    dest_rect.left(), dest_rect.top(),
                    split_pos_x - dest_rect.left(),
                    dest_rect.height()
                )

                if source_rect_enh.isValid() and target_rect_enh.isValid():
                    painter.drawPixmap(target_rect_enh, self.enhanced_pixmap, source_rect_enh)

                painter.setPen(QPen(QColor(255, 255, 255, 200), 2.5))
                painter.drawLine(int(split_pos_x), int(dest_rect.top()), int(split_pos_x), int(dest_rect.bottom()))
                painter.setPen(QPen(QColor(0, 0, 0, 100), 1.0))
                painter.drawLine(int(split_pos_x), int(dest_rect.top()), int(split_pos_x), int(dest_rect.bottom()))

                triangle_y_center = dest_rect.center().y()
                triangle_path = QPainterPath()
                triangle_path.moveTo(split_pos_x - self._triangle_size, triangle_y_center)
                triangle_path.lineTo(split_pos_x, triangle_y_center - self._triangle_size)
                triangle_path.lineTo(split_pos_x + self._triangle_size, triangle_y_center)
                triangle_path.lineTo(split_pos_x, triangle_y_center + self._triangle_size)
                triangle_path.closeSubpath()

                self._triangle_rect = triangle_path.boundingRect().toRect().adjusted(-5, -5, 5, 5)

                painter.setPen(QPen(QColor(0, 0, 0, 50), 1))
                painter.setBrush(QColor(255, 255, 255, 220))
                painter.drawPath(triangle_path)
        else:
            painter.setPen(QColor("#777777"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "拖放图像文件至此\n或使用按钮加载")

        if self._is_dragging_over:
            painter.setBrush(QBrush(QColor(79, 70, 229, 40)))
            drag_pen = QPen(QColor(79, 70, 229, 200), 3)
            drag_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(drag_pen)
            painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 10, 10)

        painter.end()