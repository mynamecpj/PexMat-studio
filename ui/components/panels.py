import functools
from typing import Optional

from PySide6.QtWidgets import (QFrame, QWidget, QVBoxLayout, QGraphicsDropShadowEffect, QMainWindow, QToolBar,
                               QStackedWidget, QToolButton, QPushButton, QHBoxLayout, QLabel, QGroupBox, QCheckBox,
                               QGridLayout, QRadioButton, QSlider, QSpacerItem, QSizePolicy, QComboBox, QSpinBox,
                               QListWidget, QDoubleSpinBox, QButtonGroup, QScrollArea)
from PySide6.QtGui import QPainter, QPen, QColor, QMouseEvent, QPaintEvent, QPainterPath
from PySide6.QtCore import Qt, QRect, Signal, Slot, QParallelAnimationGroup, QPropertyAnimation, QEasingCurve, QTimer, \
    QSize, QPoint

# Extracted from inline imports
import ui.views.image_view

from config.settings import (C_WIDGET_BG, C_DIVIDER_COLOR, REFINEMENT_UPDATE_DEBOUNCE_MS, C_SECONDARY_BUTTON_TEXT,
                             DEFAULT_REFINE_SMOOTH, ENHANCE_MODELS, DEFAULT_ENHANCE_MODEL_NAME,
                             STITCHING_DEFAULT_CANVAS_WIDTH, STITCHING_DEFAULT_CANVAS_HEIGHT, MAX_REFINE_SMOOTH,
                             MAX_REFINE_FEATHER, DEFAULT_REFINE_FEATHER, MAX_REFINE_SHIFT_SLIDER,
                             DEFAULT_REFINE_SHIFT_SLIDER, DEFAULT_REFINE_SHIFT, DEFAULT_REFINE_GUIDED_FILTER_ENABLED,
                             MAX_REFINE_GUIDED_FILTER_RADIUS, DEFAULT_REFINE_GUIDED_FILTER_RADIUS,
                             MAX_REFINE_GUIDED_FILTER_EPS_SCALED, DEFAULT_REFINE_GUIDED_FILTER_EPS_SCALED,
                             C_TEXT_PRIMARY, _TR)

# ==========================================
# Modern frameless tool button style (Dark mode)
# ==========================================
MODERN_TOOLBUTTON_STYLE = """
    QToolButton {
        border: none;
        background-color: transparent;
        border-radius: 8px;
        padding: 8px 6px;
        color: #E0E0E0; 
        font-weight: 500;
        font-size: 13px;
    }
    QToolButton:hover {
        background-color: rgba(255, 255, 255, 0.08); 
        color: #FFFFFF;
    }
    QToolButton:pressed, QToolButton:checked {
        background-color: rgba(255, 255, 255, 0.12);
        color: #60A5FA;
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

# ==========================================
# Exclusive combo box style (Dark mode)
# ==========================================
MODERN_COMBOBOX_STYLE = """
    QComboBox {
        background-color: #333333;
        border: 1px solid #404040;
        border-radius: 6px;
        padding: 4px 8px;
        color: #E0E0E0;
    }
    QComboBox:hover {
        border: 1px solid #1A73E8;
    }
    QComboBox::drop-down {
        border: none;
        width: 24px;
    }
    QComboBox QAbstractItemView {
        background-color: #262626;
        border: 1px solid #404040;
        border-radius: 6px;
        selection-background-color: #1A73E8; 
        selection-color: #FFFFFF;            
        outline: none;
    }
    QComboBox QAbstractItemView::item {
        padding: 8px 8px;
        color: #E0E0E0;
    }
    QComboBox QAbstractItemView::item:hover {
        background-color: #333333;           
        color: #FFFFFF;
    }
"""

# ==========================================
# Modern spin box style (Dark mode + SVG up/down arrow icons)
# ==========================================
MODERN_SPINBOX_STYLE = """
    QSpinBox, QDoubleSpinBox {
        background-color: #333333;
        border: 1px solid #404040;
        border-radius: 6px;
        padding: 4px 24px 4px 8px; 
        color: #E0E0E0;
        min-height: 22px;
    }
    QSpinBox:hover, QDoubleSpinBox:hover {
        border: 1px solid #1A73E8;
    }
    QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #3B82F6;
        background-color: #262626;
        outline: none;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #404040;
        border-bottom: 1px solid transparent;
        border-top-right-radius: 6px;
        background-color: #2A2A2A;
        image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNFMEUwRTAiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cG9seWxpbmUgcG9pbnRzPSIxOCAxNSAxMiA5IDYgMTUiPjwvcG9seWxpbmU+PC9zdmc+);
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
        background-color: #404040;
    }
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
        background-color: #1A73E8;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 20px;
        border-left: 1px solid #404040;
        border-top: 1px solid #404040;
        border-bottom-right-radius: 6px;
        background-color: #2A2A2A;
        image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMiIgaGVpZ2h0PSIxMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNFMEUwRTAiIHN0cm9rZS13aWR0aD0iMyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cG9seWxpbmUgcG9pbnRzPSI2IDkgMTIgMTUgMTggOSI+PC9wb2x5bGluZT48L3N2Zz4=);
    }
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background-color: #404040;
    }
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
        background-color: #1A73E8;
    }
    QSpinBox::disabled, QDoubleSpinBox::disabled {
        background-color: #2A2A2A;
        color: #555555;
        border: 1px solid #333333;
    }
    QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled,
    QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {
        background-color: #2A2A2A;
        image: none;
    }
"""


class SlidingPanelFrame(QFrame):
    """
    A setting panel component specifically designed to slide out from the right.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_color = QColor("#262626")
        self.border_color = QColor("#333333")
        self.border_radius = 16.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(0, 0, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, self.border_radius, self.border_radius)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.fillPath(path, self.bg_color)

        pen = QPen(self.border_color, 1)
        painter.setPen(pen)
        painter.drawPath(path)


class AssetPanelInnerFrame(QFrame):
    """
    The base drawing component for the left asset library.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_color = QColor("#262626")
        self.border_color = QColor("#333333")
        self.border_radius = 16.0

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, self.border_radius, self.border_radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.fillPath(path, self.bg_color)
        pen = QPen(self.border_color, 1)
        painter.setPen(pen)
        painter.drawPath(path)


class AssetPanelFrame(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AssetLibraryPanel")

        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.setFixedWidth(230)

        self._dragging_mouse_press_pos = None
        self._dragging_widget_press_pos = None
        self.setCursor(Qt.ArrowCursor)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(0)

        self.inner_frame = AssetPanelInnerFrame(self)
        self.main_layout.addWidget(self.inner_frame)

        asset_panel_shadow = QGraphicsDropShadowEffect(self.inner_frame)
        asset_panel_shadow.setBlurRadius(20)
        asset_panel_shadow.setColor(QColor(0, 0, 0, 150))
        asset_panel_shadow.setOffset(0, 4)
        self.inner_frame.setGraphicsEffect(asset_panel_shadow)

        self.content_layout = QVBoxLayout(self.inner_frame)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

    def mousePressEvent(self, event: QMouseEvent):
        child = self.childAt(event.position().toPoint())
        if child is not None and child != self and child != self.inner_frame:
            super().mousePressEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_mouse_press_pos = event.globalPosition().toPoint()
            self._dragging_widget_press_pos = self.pos()
            self.setCursor(Qt.SizeAllCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging_mouse_press_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            delta = event.globalPosition().toPoint() - self._dragging_mouse_press_pos
            self.move(self._dragging_widget_press_pos + delta)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_mouse_press_pos is not None:
            self._dragging_mouse_press_pos = None
            self._dragging_widget_press_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class FloatingPanelContainer(QWidget):
    closed = Signal()
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FloatingPanelContainer")
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self._content_widget = None
        self.pending_content_widget = None
        self.is_opening = False

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        self.content_frame = QFrame(self)
        self.content_frame.setObjectName("FloatingPanelContentFrame")
        self.main_layout.addWidget(self.content_frame)

        shadow = QGraphicsDropShadowEffect(self.content_frame)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 4)
        self.content_frame.setGraphicsEffect(shadow)

        frame_internal_layout = QVBoxLayout(self.content_frame)
        frame_internal_layout.setContentsMargins(0, 0, 0, 0)
        frame_internal_layout.setSpacing(0)

        self.draggable_area = QWidget()
        self.draggable_area.setFixedHeight(25)
        self.draggable_area.setCursor(Qt.SizeAllCursor)
        frame_internal_layout.addWidget(self.draggable_area)

        self.actual_content_container = QWidget()
        self.content_layout = QVBoxLayout(self.actual_content_container)
        self.content_layout.setContentsMargins(10, 5, 10, 10)
        frame_internal_layout.addWidget(self.actual_content_container, 1)

        self.animation_group = QParallelAnimationGroup(self)
        self.geom_animation = QPropertyAnimation(self, b"geometry")
        self.opacity_animation = QPropertyAnimation(self, b"windowOpacity")

        self.geom_animation.setDuration(380)
        self.geom_animation.setEasingCurve(QEasingCurve.Type.InOutExpo)
        self.opacity_animation.setDuration(250)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.animation_group.addAnimation(self.geom_animation)
        self.animation_group.addAnimation(self.opacity_animation)
        self.animation_group.finished.connect(self._on_animation_finished)

        self._dragging_mouse_press_pos = None
        self._dragging_widget_press_pos = None

    def set_content_widget(self, widget: QWidget | None):
        current_content = None
        if self.content_layout.count() > 0:
            child_item = self.content_layout.takeAt(0)
            if child_item and child_item.widget():
                current_content = child_item.widget()
                current_content.setParent(None)
                current_content.hide()

        self._content_widget = widget
        if self._content_widget:
            if self._content_widget.parentWidget() is not None and self._content_widget.parentWidget() != self.actual_content_container:
                self._content_widget.setParent(self.actual_content_container)

            self.content_layout.addWidget(self._content_widget)
            self._content_widget.show()
            self.log_message.emit(
                f"FloatingPanel: Set content to {self._content_widget.objectName() if self._content_widget else 'None'}")
        elif current_content:
            self.log_message.emit(
                f"FloatingPanel: Content cleared (was {current_content.objectName() if current_content else 'Unknown'})")

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.draggable_area.underMouse():
            self._dragging_mouse_press_pos = event.globalPosition().toPoint()
            self._dragging_widget_press_pos = self.pos()
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging_mouse_press_pos is not None:
            delta = event.globalPosition().toPoint() - self._dragging_mouse_press_pos
            self.move(self._dragging_widget_press_pos + delta)
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_mouse_press_pos = None
            self._dragging_widget_press_pos = None
            event.accept()
        else:
            event.ignore()

    def show_animated(self, start_rect: QRect, target_rect: QRect):
        self.log_message.emit(f"FloatingPanel: show_animated from {start_rect} to {target_rect}")
        self.is_opening = True
        if hasattr(self, 'pending_content_widget') and self.pending_content_widget:
            self.set_content_widget(self.pending_content_widget)
            self.pending_content_widget = None

        self.setGeometry(start_rect)
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()

        self.geom_animation.setStartValue(start_rect)
        self.geom_animation.setEndValue(target_rect)
        self.opacity_animation.setStartValue(0.0)
        self.opacity_animation.setEndValue(1.0)
        self.animation_group.start()

    def hide_animated(self, target_rect: QRect):
        self.log_message.emit(f"FloatingPanel: hide_animated to {target_rect}")
        self.is_opening = False
        self.geom_animation.setStartValue(self.geometry())
        self.geom_animation.setEndValue(target_rect)
        self.opacity_animation.setStartValue(self.windowOpacity())
        self.opacity_animation.setEndValue(0.0)
        self.animation_group.start()

    @Slot()
    def _on_animation_finished(self):
        if not self.is_opening:
            self.hide()
            if self._content_widget:
                self._content_widget.hide()
            self.closed.emit()
        else:
            self.setWindowOpacity(1.0)

    def set_pending_content(self, widget: QWidget):
        self.pending_content_widget = widget


class PanelMixin:
    """
    Responsible for managing the construction and animation of all sidebars, floating panels, inspector panels, and their internal components.
    """

    def _init_floating_panels_and_timers(self):
        self.playback_timer = QTimer(self)
        if hasattr(self, 'advance_frame_playback'):
            self.playback_timer.timeout.connect(self.advance_frame_playback)

        self._refinement_update_timer = QTimer(self)
        self._refinement_update_timer.setSingleShot(True)
        self._refinement_update_timer.setInterval(REFINEMENT_UPDATE_DEBOUNCE_MS)
        self._refinement_update_timer.timeout.connect(self._apply_pending_refinements)
        self._pending_refinement_values = {}

        self.asset_library_panel_floating = AssetPanelFrame(self)
        self.asset_library_panel_floating.hide()

        self.asset_panel_animation_floating = QPropertyAnimation(self.asset_library_panel_floating, b"geometry")
        self.asset_panel_animation_floating.setDuration(300)
        self.asset_panel_animation_floating.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.floating_panel_container = FloatingPanelContainer(self)
        self.floating_panel_container.hide()
        self.floating_panel_container.closed.connect(self._open_new_panel_after_close)

        self.current_active_panel_button = None
        self.current_panel_content_widget = None
        self.page_specific_panels = {}
        self.page_top_button_groups = {}
        self._pending_new_panel_args = None
        self._was_right_panel_visible_on_deactivate = False
        self._was_left_panel_visible_on_deactivate = False

    def _create_animated_inspector_panel(self):
        inspector_bar = QToolBar("Inspector")
        inspector_bar.setObjectName("InspectorToolBar")
        inspector_bar.setOrientation(Qt.Orientation.Vertical)
        inspector_bar.setMovable(False)
        inspector_bar.setFloatable(False)
        inspector_bar.setIconSize(QSize(24, 24))
        inspector_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        inspector_bar.setFixedWidth(75)

        settings_stack = QStackedWidget()
        settings_stack.setObjectName("RightPropertiesPanel")
        settings_stack.setFixedWidth(0)

        animation = QPropertyAnimation(settings_stack, b"minimumWidth")
        animation.setDuration(250)
        animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        return inspector_bar, settings_stack, animation

    def _on_top_panel_button_clicked(self, trigger_button, panel_key):
        """Perfectly route the old sliding panel logic to the new fixed three-column layout."""
        if not hasattr(self, 'right_properties_stack'):
            return

        if panel_key == "stitch_enhance":
            self.right_properties_stack.setCurrentWidget(self.enhance_props_widget)

            if hasattr(self, 'stitching_canvas'):
                item = self.stitching_canvas.get_primary_selected_item()
                if item and hasattr(self, 'image_compare_widget_enhance'):
                    self.image_compare_widget_enhance.set_images(original=item.pixmap, enhanced=None)

        elif panel_key in ["workshop_item_tools", "stitch_canvas_settings", "stitch_layers"]:
            self.right_properties_stack.setCurrentWidget(self.canvas_props_widget)

        elif panel_key in ["segment_overlay_settings", "segment_overlay_refine", "segment_overlay_color"]:
            self.right_properties_stack.setCurrentWidget(self.segment_props_widget)

    def _show_actual_panel(self, trigger_button: QToolButton, content_widget: QWidget):
        pass

    def hide_floating_panel(self):
        pass

    @Slot()
    def _open_new_panel_after_close(self):
        if hasattr(self, '_pending_new_panel_args') and self._pending_new_panel_args:
            trigger_button, content_widget = self._pending_new_panel_args
            self._show_actual_panel(trigger_button, content_widget)
            self._pending_new_panel_args = None

    def _create_panel_content(self, panel_key: str):
        if panel_key in getattr(self, 'page_specific_panels', {}):
            return self.page_specific_panels[panel_key]

        panel_content_container = QWidget()
        panel_content_container.setStyleSheet("background-color: transparent;")
        panel_master_layout = QVBoxLayout(panel_content_container)
        panel_master_layout.setContentsMargins(0, 0, 0, 0)
        panel_master_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { background-color: transparent; border: none; outline: none; }")

        inner_widget = QWidget()
        inner_widget.setObjectName("InnerPanelContent")
        inner_layout = QVBoxLayout(inner_widget)
        inner_layout.setContentsMargins(12, 12, 12, 12)
        inner_layout.setSpacing(12)

        panel_builders = {
            "video_objects": self._build_video_objects_panel,
            "video_process": self._build_video_process_panel,
            "video_output": self._build_video_output_panel,
            "workshop_item_tools": self._build_workshop_item_tools_panel,
            "stitch_enhance": self._build_stitch_enhance_panel,
            "stitch_canvas_settings": self._build_stitch_canvas_settings_panel,
            "stitch_layers": self._build_stitch_layers_panel,
            "segment_overlay_settings": self._build_segment_overlay_settings_panel,
            "segment_overlay_refine": self._build_segment_overlay_refine_panel,
            "segment_overlay_color": self._build_segment_overlay_color_panel,
        }

        builder_func = panel_builders.get(panel_key)
        if builder_func:
            builder_func(inner_layout)
            inner_layout.addStretch(1)
        else:
            panel_content_container.deleteLater()
            return None

        scroll_area.setWidget(inner_widget)
        panel_master_layout.addWidget(scroll_area)

        if not hasattr(self, 'page_specific_panels'):
            self.page_specific_panels = {}
        self.page_specific_panels[panel_key] = panel_content_container
        return panel_content_container

    def create_card_widget(self, title_text: str):
        """
        Compact advanced card, automatically binds original text to support translation switching.
        """
        card_container = QWidget()
        card_container.setStyleSheet("""
            QWidget { background-color: #262626; border-radius: 8px; }
        """)
        card_layout = QVBoxLayout(card_container)
        card_layout.setContentsMargins(12, 10, 12, 12)
        card_layout.setSpacing(10)

        title_label = QLabel()
        title_label.setProperty("orig_text", title_text)
        title_label.setText(_TR(title_text))
        title_label.setStyleSheet("""
            QLabel { color: #D0D0D0; font-size: 12px; font-weight: bold; background: transparent; border: none; }
        """)

        content_area = QWidget()
        content_area.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        card_layout.addWidget(title_label)
        card_layout.addWidget(content_area)

        return card_container, content_layout

    def _build_video_objects_panel(self, parent_layout: QVBoxLayout):
        object_management_card, object_management_layout = self.create_card_widget("对象图层管理")
        object_management_layout.setSpacing(8)

        self.add_target_button = QToolButton()
        self.add_target_button.setStyleSheet(MODERN_TOOLBUTTON_STYLE)
        self.add_target_button.setProperty("orig_text", "添加新对象")
        self.add_target_button.setText(_TR("添加新对象"))
        self.add_target_button.setIcon(self._create_svg_icon("person-add.svg", color=QColor("#E0E0E0")))
        self.add_target_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.add_target_button.setIconSize(QSize(24, 24))
        self.add_target_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.add_target_button.clicked.connect(self.add_new_video_target)
        object_management_layout.addWidget(self.add_target_button)

        self.video_objects_list = QListWidget()
        self.video_objects_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.video_objects_list.setMinimumHeight(150)
        self.video_objects_list.setStyleSheet("""
            QListWidget { background-color: #333333; border: 1px solid #404040; border-radius: 6px; outline: none; }
            QListWidget::item { border-bottom: 1px solid #404040; padding: 2px; color: #E0E0E0; }
            QListWidget::item:selected { background-color: #1A73E8; border-left: 3px solid #60A5FA; color: #FFFFFF; }
        """)
        original_mouse_press = self.video_objects_list.mousePressEvent

        def custom_mouse_press(event):
            item = self.video_objects_list.itemAt(event.pos())
            if not item: self.video_objects_list.clearSelection()
            original_mouse_press(event)

        self.video_objects_list.mousePressEvent = custom_mouse_press
        self.video_objects_list.itemSelectionChanged.connect(self._on_video_object_selection_changed)
        object_management_layout.addWidget(self.video_objects_list)

        history_actions_layout = QHBoxLayout()
        history_actions_layout.setSpacing(4)

        self.vid_undo_button = QToolButton()
        self.vid_undo_button.setStyleSheet(MODERN_TOOLBUTTON_STYLE)
        self.vid_undo_button.setProperty("orig_text", "撤回")
        self.vid_undo_button.setText(_TR("撤回"))
        self.vid_undo_button.setIcon(self._create_svg_icon("arrow-90deg-left.svg", color=QColor("#E0E0E0")))
        self.vid_undo_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.vid_undo_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.vid_undo_button.clicked.connect(self.undo_video_action)
        history_actions_layout.addWidget(self.vid_undo_button)

        self.vid_redo_button = QToolButton()
        self.vid_redo_button.setStyleSheet(MODERN_TOOLBUTTON_STYLE)
        self.vid_redo_button.setProperty("orig_text", "重做")
        self.vid_redo_button.setText(_TR("重做"))
        self.vid_redo_button.setIcon(self._create_svg_icon("arrow-90deg-right.svg", color=QColor("#E0E0E0")))
        self.vid_redo_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.vid_redo_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.vid_redo_button.clicked.connect(self.redo_video_action)
        history_actions_layout.addWidget(self.vid_redo_button)

        self.delete_current_target_button = QToolButton()
        self.delete_current_target_button.setStyleSheet(MODERN_TOOLBUTTON_STYLE)
        self.delete_current_target_button.setProperty("orig_text", "删除选中")
        self.delete_current_target_button.setText(_TR("删除选中"))
        self.delete_current_target_button.setIcon(self._create_svg_icon("trash.svg", color=QColor("#EF4444")))
        self.delete_current_target_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.delete_current_target_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.delete_current_target_button.clicked.connect(self.clear_current_video_target)
        history_actions_layout.addWidget(self.delete_current_target_button)

        object_management_layout.addLayout(history_actions_layout)

        parent_layout.addWidget(object_management_card)

    def _build_video_process_panel(self, parent_layout: QVBoxLayout):
        process_actions_card, process_actions_layout = self.create_card_widget("处理与操作")
        process_actions_layout.setSpacing(8)

        manual_edit_group = QGroupBox()
        manual_edit_group.setProperty("orig_title", "当前帧蒙版手工修复")
        manual_edit_group.setTitle(_TR("当前帧蒙版手工修复"))
        manual_edit_group.setStyleSheet(
            "QGroupBox { font-weight: bold; margin-top: 10px; padding-top: 15px; color: #E0E0E0; }")
        manual_edit_layout = QVBoxLayout(manual_edit_group)
        manual_edit_layout.setSpacing(10)

        mode_layout = QHBoxLayout()
        self.vid_mode_point_btn = QRadioButton()
        self.vid_mode_point_btn.setProperty("orig_text", "打点/画框")
        self.vid_mode_point_btn.setText(_TR("打点/画框"))
        self.vid_mode_point_btn.setChecked(True)

        self.vid_mode_brush_btn = QRadioButton()
        self.vid_mode_brush_btn.setProperty("orig_text", "画笔修补")
        self.vid_mode_brush_btn.setText(_TR("画笔修补"))

        mode_group = QButtonGroup(self)
        mode_group.addButton(self.vid_mode_point_btn)
        mode_group.addButton(self.vid_mode_brush_btn)

        mode_layout.addWidget(self.vid_mode_point_btn)
        mode_layout.addWidget(self.vid_mode_brush_btn)
        manual_edit_layout.addLayout(mode_layout)

        brush_size_layout = QHBoxLayout()
        brush_lbl = QLabel()
        brush_lbl.setProperty("orig_text", "画笔粗细:")
        brush_lbl.setText(_TR("画笔粗细:"))
        brush_size_layout.addWidget(brush_lbl)

        self.vid_brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.vid_brush_slider.setRange(1, 100)
        self.vid_brush_slider.setValue(15)
        brush_size_layout.addWidget(self.vid_brush_slider)
        self.vid_brush_label = QLabel("15 px")
        brush_size_layout.addWidget(self.vid_brush_label)
        manual_edit_layout.addLayout(brush_size_layout)

        morph_layout = QHBoxLayout()
        morph_lbl = QLabel()
        morph_lbl.setProperty("orig_text", "蒙版收缩/扩张:")
        morph_lbl.setText(_TR("蒙版收缩/扩张:"))
        morph_layout.addWidget(morph_lbl)

        self.vid_morph_slider = QSlider(Qt.Orientation.Horizontal)
        self.vid_morph_slider.setRange(-20, 20)
        self.vid_morph_slider.setValue(0)
        morph_layout.addWidget(self.vid_morph_slider)
        self.vid_morph_label = QLabel("0 px")
        morph_layout.addWidget(self.vid_morph_label)
        manual_edit_layout.addLayout(morph_layout)

        process_actions_layout.addWidget(manual_edit_group)

        refine_config_group = QGroupBox()
        refine_config_group.setProperty("orig_title", "全视频精细抠图配置")
        refine_config_group.setTitle(_TR("全视频精细抠图配置"))
        refine_config_group.setStyleSheet(
            "QGroupBox { font-weight: bold; margin-top: 10px; padding-top: 15px; color: #E0E0E0; }")
        refine_config_layout = QVBoxLayout(refine_config_group)
        refine_config_layout.setSpacing(6)

        self.vid_matteformer_checkbox = QCheckBox()
        self.vid_matteformer_checkbox.setProperty("orig_text", "启用精细视频抠图")
        self.vid_matteformer_checkbox.setText(_TR("启用精细视频抠图"))
        self.vid_matteformer_checkbox.setProperty("orig_tooltip", "动态 ROI 发丝级边缘提取，效果极佳")
        self.vid_matteformer_checkbox.setToolTip(_TR("动态 ROI 发丝级边缘提取，效果极佳"))
        self.vid_matteformer_checkbox.setEnabled(getattr(self, 'matteformer_loaded', False))
        refine_config_layout.addWidget(self.vid_matteformer_checkbox)

        erode_layout = QHBoxLayout()
        erode_lbl = QLabel()
        erode_lbl.setProperty("orig_text", "腐蚀 (Erode):")
        erode_lbl.setText(_TR("腐蚀 (Erode):"))
        erode_layout.addWidget(erode_lbl)
        self.vid_erode_slider = QSlider(Qt.Orientation.Horizontal)
        self.vid_erode_slider.setRange(0, 30)
        self.vid_erode_slider.setValue(10)
        erode_layout.addWidget(self.vid_erode_slider)
        self.vid_erode_label = QLabel("10")
        self.vid_erode_slider.valueChanged.connect(lambda v: self.vid_erode_label.setText(str(v)))
        refine_config_layout.addLayout(erode_layout)

        dilate_layout = QHBoxLayout()
        dilate_lbl = QLabel()
        dilate_lbl.setProperty("orig_text", "膨胀 (Dilate):")
        dilate_lbl.setText(_TR("膨胀 (Dilate):"))
        dilate_layout.addWidget(dilate_lbl)
        self.vid_dilate_slider = QSlider(Qt.Orientation.Horizontal)
        self.vid_dilate_slider.setRange(0, 30)
        self.vid_dilate_slider.setValue(10)
        dilate_layout.addWidget(self.vid_dilate_slider)
        self.vid_dilate_label = QLabel("10")
        self.vid_dilate_slider.valueChanged.connect(lambda v: self.vid_dilate_label.setText(str(v)))
        refine_config_layout.addLayout(dilate_layout)

        process_actions_layout.addWidget(refine_config_group)
        process_actions_layout.addWidget(self._create_separator())

        video_processing_buttons_layout = QHBoxLayout()
        video_processing_buttons_layout.setSpacing(6)

        self.start_video_seg_button = QToolButton()
        self.start_video_seg_button.setStyleSheet(MODERN_TOOLBUTTON_STYLE)
        self.start_video_seg_button.setProperty("orig_text", "开始处理")
        self.start_video_seg_button.setText(_TR("开始处理"))
        self.start_video_seg_button.setIcon(self._create_svg_icon("play.svg", color=QColor("#E0E0E0")))
        self.start_video_seg_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.start_video_seg_button.setIconSize(QSize(24, 24))
        self.start_video_seg_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.start_video_seg_button.clicked.connect(self.start_video_segmentation_propagation)
        video_processing_buttons_layout.addWidget(self.start_video_seg_button)

        self.cancel_video_seg_button = QToolButton()
        self.cancel_video_seg_button.setStyleSheet(MODERN_TOOLBUTTON_STYLE)
        self.cancel_video_seg_button.setProperty("orig_text", "取消")
        self.cancel_video_seg_button.setText(_TR("取消"))
        self.cancel_video_seg_button.setIcon(self._create_svg_icon("stop.svg", color=QColor("#E0E0E0")))
        self.cancel_video_seg_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.cancel_video_seg_button.setIconSize(QSize(24, 24))
        self.cancel_video_seg_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cancel_video_seg_button.clicked.connect(self.cancel_current_video_operation)
        video_processing_buttons_layout.addWidget(self.cancel_video_seg_button)

        process_actions_layout.addLayout(video_processing_buttons_layout)
        parent_layout.addWidget(process_actions_card)

        def update_vid_interaction_mode():
            if self.vid_mode_brush_btn.isChecked():
                mode = 'brush'
            else:
                mode = 'point'
            if hasattr(self, 'video_display_label'):
                self.video_display_label.set_interaction_mode(mode)

        self.vid_mode_point_btn.toggled.connect(update_vid_interaction_mode)
        self.vid_mode_brush_btn.toggled.connect(update_vid_interaction_mode)

        self.vid_brush_slider.valueChanged.connect(lambda v: self.vid_brush_label.setText(f"{v} px"))
        self.vid_brush_slider.valueChanged.connect(
            lambda v: self.video_display_label.set_brush_size(v) if hasattr(self, 'video_display_label') else None)

        def _on_morph_released():
            if hasattr(self, 'video_display_label'): self.video_display_label.end_mask_shift()
            self.vid_morph_slider.blockSignals(True)
            self.vid_morph_slider.setValue(0)
            self.vid_morph_label.setText("0 px")
            self.vid_morph_slider.blockSignals(False)

        self.vid_morph_slider.sliderPressed.connect(
            lambda: self.video_display_label.begin_mask_shift() if hasattr(self, 'video_display_label') else None)
        self.vid_morph_slider.valueChanged.connect(lambda v: self.vid_morph_label.setText(f"{v} px"))
        self.vid_morph_slider.valueChanged.connect(
            lambda v: self.video_display_label.apply_mask_shift(v) if hasattr(self, 'video_display_label') else None)
        self.vid_morph_slider.sliderReleased.connect(_on_morph_released)

    def _build_video_output_panel(self, parent_layout: QVBoxLayout):
        preview_card, preview_layout = self.create_card_widget("当前帧效果预览")
        self.video_result_preview_label = QLabel()
        self.video_result_preview_label.setProperty("orig_text", "处理后预览将在此显示")
        self.video_result_preview_label.setText(_TR("处理后预览将在此显示"))
        self.video_result_preview_label.setObjectName("VideoPreviewDisplayLabel")
        self.video_result_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_result_preview_label.setMinimumHeight(150)
        self.video_result_preview_label.setStyleSheet("color: #777777;")
        preview_layout.addWidget(self.video_result_preview_label, 1)
        parent_layout.addWidget(preview_card, 0)

        output_settings_card, output_settings_layout = self.create_card_widget("输出设置")
        output_settings_layout.setSpacing(8)

        output_actions_layout = QHBoxLayout()
        output_actions_layout.setSpacing(6)

        bg_color_lbl = QLabel()
        bg_color_lbl.setProperty("orig_text", "背景色:")
        bg_color_lbl.setText(_TR("背景色:"))
        output_actions_layout.addWidget(bg_color_lbl)

        self.bg_color_button = QPushButton()
        self._update_bg_color_button_style()
        self.bg_color_button.setMinimumHeight(28)
        self.bg_color_button.clicked.connect(self._select_video_bg_color)
        output_actions_layout.addWidget(self.bg_color_button)
        output_actions_layout.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum))

        self.save_video_seg_button = QToolButton()
        self.save_video_seg_button.setStyleSheet(MODERN_TOOLBUTTON_STYLE)
        self.save_video_seg_button.setProperty("orig_text", "保存视频")
        self.save_video_seg_button.setText(_TR("保存视频"))
        self.save_video_seg_button.setIcon(self._create_svg_icon("download.svg", color=QColor("#E0E0E0")))
        self.save_video_seg_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.save_video_seg_button.setIconSize(QSize(24, 24))
        self.save_video_seg_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.save_video_seg_button.clicked.connect(self.save_video_segmentation_result)
        output_actions_layout.addWidget(self.save_video_seg_button)

        output_settings_layout.addLayout(output_actions_layout)
        parent_layout.addWidget(output_settings_card)

    def _build_workshop_item_tools_panel(self, parent_layout: QVBoxLayout):
        props_card, props_layout = self.create_card_widget("选中素材属性")

        self.stitch_props_widget = QWidget()
        self.stitch_props_widget.setEnabled(False)
        props_grid = QGridLayout(self.stitch_props_widget)
        props_grid.setContentsMargins(0, 0, 0, 0)
        props_grid.setSpacing(8)

        # X 坐标
        x_lbl = QLabel()
        x_lbl.setProperty("orig_text", "X:")
        x_lbl.setText(_TR("X:"))
        props_grid.addWidget(x_lbl, 0, 0)

        self.stitch_pos_x_spin = QDoubleSpinBox()
        self.stitch_pos_x_spin.setStyleSheet(MODERN_SPINBOX_STYLE)
        self.stitch_pos_x_spin.setRange(-10000, 10000)
        self.stitch_pos_x_spin.setDecimals(1)
        self.stitch_pos_x_spin.valueChanged.connect(
            lambda v: self.update_selected_item_property('x', v))
        props_grid.addWidget(self.stitch_pos_x_spin, 0, 1)

        # Y 坐标
        y_lbl = QLabel()
        y_lbl.setProperty("orig_text", "Y:")
        y_lbl.setText(_TR("Y:"))
        props_grid.addWidget(y_lbl, 0, 2)

        self.stitch_pos_y_spin = QDoubleSpinBox()
        self.stitch_pos_y_spin.setStyleSheet(MODERN_SPINBOX_STYLE)
        self.stitch_pos_y_spin.setRange(-10000, 10000)
        self.stitch_pos_y_spin.setDecimals(1)
        self.stitch_pos_y_spin.valueChanged.connect(
            lambda v: self.update_selected_item_property('y', v))
        props_grid.addWidget(self.stitch_pos_y_spin, 0, 3)

        # 宽度
        w_lbl = QLabel()
        w_lbl.setProperty("orig_text", "宽:")
        w_lbl.setText(_TR("宽:"))
        props_grid.addWidget(w_lbl, 1, 0)

        self.stitch_size_w_spin = QDoubleSpinBox()
        self.stitch_size_w_spin.setStyleSheet(MODERN_SPINBOX_STYLE)
        self.stitch_size_w_spin.setRange(1, 16000)
        self.stitch_size_w_spin.setDecimals(1)
        self.stitch_size_w_spin.valueChanged.connect(
            lambda v: self.update_selected_item_property('width', v))
        props_grid.addWidget(self.stitch_size_w_spin, 1, 1)

        # 高度
        h_lbl = QLabel()
        h_lbl.setProperty("orig_text", "高:")
        h_lbl.setText(_TR("高:"))
        props_grid.addWidget(h_lbl, 1, 2)

        self.stitch_size_h_spin = QDoubleSpinBox()
        self.stitch_size_h_spin.setStyleSheet(MODERN_SPINBOX_STYLE)
        self.stitch_size_h_spin.setRange(1, 16000)
        self.stitch_size_h_spin.setDecimals(1)
        self.stitch_size_h_spin.valueChanged.connect(
            lambda v: self.update_selected_item_property('height', v))
        props_grid.addWidget(self.stitch_size_h_spin, 1, 3)

        # 旋转
        rot_lbl = QLabel()
        rot_lbl.setProperty("orig_text", "旋转:")
        rot_lbl.setText(_TR("旋转:"))
        props_grid.addWidget(rot_lbl, 2, 0)

        self.stitch_rotation_spin = QDoubleSpinBox()
        self.stitch_rotation_spin.setStyleSheet(MODERN_SPINBOX_STYLE)
        self.stitch_rotation_spin.setRange(-360, 360)
        self.stitch_rotation_spin.setDecimals(1)
        self.stitch_rotation_spin.setSingleStep(1.0)
        self.stitch_rotation_spin.valueChanged.connect(
            lambda v: self.update_selected_item_property('rotation', v))
        props_grid.addWidget(self.stitch_rotation_spin, 2, 1, 1, 3)

        props_layout.addWidget(self.stitch_props_widget)

        combine_btn_layout = QHBoxLayout()
        self.combine_selected_button = QPushButton()
        self.combine_selected_button.setProperty("orig_text", "合并选中项")
        self.combine_selected_button.setText(_TR("合并选中项"))
        self.combine_selected_button.setStyleSheet(MODERN_PUSHBUTTON_STYLE)
        self.combine_selected_button.clicked.connect(self.combine_selected_items_action)

        self.uncombine_selected_button = QPushButton()
        self.uncombine_selected_button.setProperty("orig_text", "取消合并")
        self.uncombine_selected_button.setText(_TR("取消合并"))
        self.uncombine_selected_button.setStyleSheet(MODERN_PUSHBUTTON_STYLE)
        self.uncombine_selected_button.clicked.connect(self.uncombine_selected_item_action)

        combine_btn_layout.addWidget(self.combine_selected_button)
        combine_btn_layout.addWidget(self.uncombine_selected_button)
        props_layout.addLayout(combine_btn_layout)

        parent_layout.addWidget(props_card)

    def _build_stitch_enhance_panel(self, parent_layout: QVBoxLayout):
        enhance_card, enhance_layout = self.create_card_widget("放大参数设置")

        enhance_grid = QGridLayout()
        enhance_grid.setContentsMargins(0, 0, 0, 0)
        enhance_grid.setSpacing(8)

        enhance_ratio_lbl = QLabel()
        enhance_ratio_lbl.setProperty("orig_text", "放大倍率:")
        enhance_ratio_lbl.setText(_TR("放大倍率:"))
        enhance_grid.addWidget(enhance_ratio_lbl, 0, 0)

        self.enhance_scale_combo = QComboBox()
        self.enhance_scale_combo.setStyleSheet(MODERN_COMBOBOX_STYLE)
        self.enhance_scale_combo.addItems(["2x", "3x", "4x"])
        self.enhance_scale_combo.setCurrentText("4x")
        enhance_grid.addWidget(self.enhance_scale_combo, 0, 1)

        tile_mode_lbl = QLabel()
        tile_mode_lbl.setProperty("orig_text", "分块模式:")
        tile_mode_lbl.setText(_TR("分块模式:"))
        enhance_grid.addWidget(tile_mode_lbl, 1, 0)

        self.tile_mode_combo = QComboBox()
        self.tile_mode_combo.setStyleSheet(MODERN_COMBOBOX_STYLE)

        # Establish English mapping structure, relying on QComboBox's I18N traversal engine to complete subsequent hot reloading.
        self.tile_mode_options = {
            "自动 (推荐)": {"value": 0}, "小图模式": {"value": 512},
            "中图模式": {"value": 256}, "大图模式": {"value": 128}, "自定义": {"value": -1}
        }
        for opt_text in self.tile_mode_options.keys():
            self.tile_mode_combo.addItem(_TR(opt_text), opt_text)

        self.tile_mode_combo.currentIndexChanged.connect(self._on_tile_mode_changed)
        enhance_grid.addWidget(self.tile_mode_combo, 1, 1)

        self.custom_tile_size_spinbox = QSpinBox()
        self.custom_tile_size_spinbox.setStyleSheet(MODERN_SPINBOX_STYLE)
        self.custom_tile_size_spinbox.setRange(32, 1024)
        self.custom_tile_size_spinbox.setValue(256)
        self.custom_tile_size_spinbox.setVisible(False)
        enhance_grid.addWidget(self.custom_tile_size_spinbox, 2, 1)

        select_model_lbl = QLabel()
        select_model_lbl.setProperty("orig_text", "选择模型:")
        select_model_lbl.setText(_TR("选择模型:"))
        enhance_grid.addWidget(select_model_lbl, 3, 0)

        self.enhance_model_combo = QComboBox()
        self.enhance_model_combo.setStyleSheet(MODERN_COMBOBOX_STYLE)
        for model_name in ENHANCE_MODELS.keys():
            self.enhance_model_combo.addItem(_TR(model_name), model_name)
        self.enhance_model_combo.setCurrentText(_TR(DEFAULT_ENHANCE_MODEL_NAME))

        enhance_grid.addWidget(self.enhance_model_combo, 3, 1)
        enhance_layout.addLayout(enhance_grid)

        btn_layout = QHBoxLayout()
        self.enhance_selected_button = QPushButton()
        self.enhance_selected_button.setProperty("orig_text", "开始增强")
        self.enhance_selected_button.setText(_TR("开始增强"))
        self.enhance_selected_button.setStyleSheet(
            MODERN_PUSHBUTTON_STYLE + "QPushButton{background:#F59E0B; color:white; font-weight:bold; border:none;}")
        self.enhance_selected_button.clicked.connect(self._enhance_selected_item_action)

        self.save_enhanced_item_button = QPushButton()
        self.save_enhanced_item_button.setProperty("orig_text", "保存结果")
        self.save_enhanced_item_button.setText(_TR("保存结果"))
        self.save_enhanced_item_button.setStyleSheet(MODERN_PUSHBUTTON_STYLE)
        self.save_enhanced_item_button.clicked.connect(self.save_enhanced_item_result)

        btn_layout.addWidget(self.enhance_selected_button)
        btn_layout.addWidget(self.save_enhanced_item_button)
        enhance_layout.addLayout(btn_layout)

        parent_layout.addWidget(enhance_card, 0)

        preview_card, preview_layout = self.create_card_widget("增强效果预览")
        # --- 修复点：通过模块直接调用，绕过初始化期的互相依赖 ---
        self.image_compare_widget_enhance = ui.views.image_view.ImageCompareWidget(self)
        self.image_compare_widget_enhance.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_compare_widget_enhance.setStyleSheet("background-color: #121212; border-radius: 8px;")
        preview_layout.addWidget(self.image_compare_widget_enhance, 1)
        parent_layout.addWidget(preview_card, 1)

    def _build_stitch_canvas_settings_panel(self, parent_layout: QVBoxLayout):
        canvas_card, canvas_layout = self.create_card_widget("画布设置")

        canvas_size_layout = QGridLayout()
        canvas_size_layout.setContentsMargins(0, 0, 0, 0)

        canvas_w_lbl = QLabel()
        canvas_w_lbl.setProperty("orig_text", "宽度:")
        canvas_w_lbl.setText(_TR("宽度:"))
        canvas_size_layout.addWidget(canvas_w_lbl, 0, 0)

        self.stitch_canvas_width_spin = QSpinBox()
        self.stitch_canvas_width_spin.setStyleSheet(MODERN_SPINBOX_STYLE)
        self.stitch_canvas_width_spin.setRange(100, 16000)
        self.stitch_canvas_width_spin.setValue(STITCHING_DEFAULT_CANVAS_WIDTH)
        self.stitch_canvas_width_spin.valueChanged.connect(self.stitching_canvas.set_canvas_width)
        canvas_size_layout.addWidget(self.stitch_canvas_width_spin, 0, 1)

        canvas_h_lbl = QLabel()
        canvas_h_lbl.setProperty("orig_text", "高度:")
        canvas_h_lbl.setText(_TR("高度:"))
        canvas_size_layout.addWidget(canvas_h_lbl, 0, 2)

        self.stitch_canvas_height_spin = QSpinBox()
        self.stitch_canvas_height_spin.setStyleSheet(MODERN_SPINBOX_STYLE)
        self.stitch_canvas_height_spin.setRange(100, 16000)
        self.stitch_canvas_height_spin.setValue(STITCHING_DEFAULT_CANVAS_HEIGHT)
        self.stitch_canvas_height_spin.valueChanged.connect(self.stitching_canvas.set_canvas_height)
        canvas_size_layout.addWidget(self.stitch_canvas_height_spin, 0, 3)
        canvas_layout.addLayout(canvas_size_layout)

        bg_layout = QHBoxLayout()
        self.stitch_bg_transparent_radio = QRadioButton()
        self.stitch_bg_transparent_radio.setProperty("orig_text", "透明背景")
        self.stitch_bg_transparent_radio.setText(_TR("透明背景"))
        self.stitch_bg_transparent_radio.setChecked(True)
        self.stitch_bg_transparent_radio.toggled.connect(self._on_stitch_bg_mode_changed)

        self.stitch_bg_color_radio = QRadioButton()
        self.stitch_bg_color_radio.setProperty("orig_text", "纯色")
        self.stitch_bg_color_radio.setText(_TR("纯色"))
        self.stitch_bg_color_radio.toggled.connect(self._on_stitch_bg_mode_changed)

        self.stitch_bg_color_button = QPushButton()
        self.stitch_bg_color_button.setFixedSize(20, 20)
        self.stitch_bg_color_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stitch_bg_color_button.clicked.connect(self.select_stitch_bg_color)
        if not hasattr(self, 'stitch_solid_bg_color'):
            self.stitch_solid_bg_color = QColor(Qt.GlobalColor.white)
        self._update_stitch_bg_color_button_style(self.stitch_solid_bg_color)

        bg_layout.addWidget(self.stitch_bg_transparent_radio)
        bg_layout.addWidget(self.stitch_bg_color_radio)
        bg_layout.addWidget(self.stitch_bg_color_button)
        bg_layout.addStretch()

        canvas_layout.addLayout(bg_layout)
        parent_layout.addWidget(canvas_card)

    def _update_stitch_bg_color_button_style(self, color):
        """Style for the small color block of the canvas background."""
        if hasattr(self, 'stitch_bg_color_button'):
            self.stitch_bg_color_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color.name()};
                    border: 2px solid #555555;
                    border-radius: 6px;
                }}
                QPushButton:hover {{
                    border: 2px solid #1A73E8;
                }}
            """)

    def _update_bg_color_button_style(self):
        """Style for the long button of the video output background color."""
        if hasattr(self, 'bg_color_button') and hasattr(self, 'video_save_bg_color'):
            color = self.video_save_bg_color
            text_color = 'white' if color.lightnessF() < 0.5 else 'black'
            self.bg_color_button.setText(color.name().upper())
            self.bg_color_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color.name()}; 
                    color: {text_color};
                    border: 2px solid #555555;
                    border-radius: 6px;
                    font-weight: bold;
                    padding: 4px 12px;
                }}
                QPushButton:hover {{
                    border: 2px solid #1A73E8;
                }}
            """)
            self.bg_color_button.setProperty("orig_tooltip", "点击选择新的视频背景色")
            self.bg_color_button.setToolTip(_TR("点击选择新的视频背景色"))

    def _build_stitch_layers_panel(self, parent_layout: QVBoxLayout):
        layers_card, layers_layout = self.create_card_widget("图层与顺序")

        layer_buttons_layout = QHBoxLayout()
        layer_buttons_layout.setSpacing(6)

        self.stitch_layer_up_button = QToolButton()
        self.stitch_layer_up_button.setProperty("orig_tooltip", "向上移动一层")
        self.stitch_layer_up_button.setToolTip(_TR("向上移动一层"))
        self.stitch_layer_up_button.setIcon(self._create_svg_icon("chevron-up.svg", color=QColor("#E0E0E0")))

        self.stitch_layer_down_button = QToolButton()
        self.stitch_layer_down_button.setProperty("orig_tooltip", "向下移动一层")
        self.stitch_layer_down_button.setToolTip(_TR("向下移动一层"))
        self.stitch_layer_down_button.setIcon(self._create_svg_icon("chevron-down.svg", color=QColor("#E0E0E0")))

        self.stitch_layer_top_button = QToolButton()
        self.stitch_layer_top_button.setProperty("orig_tooltip", "移至最顶层")
        self.stitch_layer_top_button.setToolTip(_TR("移至最顶层"))
        self.stitch_layer_top_button.setIcon(self._create_svg_icon("chevron-double-up.svg", color=QColor("#E0E0E0")))

        self.stitch_layer_bottom_button = QToolButton()
        self.stitch_layer_bottom_button.setProperty("orig_tooltip", "移至最底层")
        self.stitch_layer_bottom_button.setToolTip(_TR("移至最底层"))
        self.stitch_layer_bottom_button.setIcon(
            self._create_svg_icon("chevron-double-down.svg", color=QColor("#E0E0E0")))

        for btn in [self.stitch_layer_up_button, self.stitch_layer_down_button, self.stitch_layer_top_button,
                    self.stitch_layer_bottom_button]:
            btn.setStyleSheet(MODERN_TOOLBUTTON_STYLE)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layer_buttons_layout.addWidget(btn)

        self.stitch_layer_up_button.clicked.connect(
            lambda: getattr(self, 'stitching_canvas', None).move_selected_layer(1) if hasattr(self,
                                                                                              'stitching_canvas') else None)
        self.stitch_layer_down_button.clicked.connect(
            lambda: getattr(self, 'stitching_canvas', None).move_selected_layer(-1) if hasattr(self,
                                                                                               'stitching_canvas') else None)
        self.stitch_layer_top_button.clicked.connect(
            lambda: getattr(self, 'stitching_canvas', None).move_selected_to_top() if hasattr(self,
                                                                                              'stitching_canvas') else None)
        self.stitch_layer_bottom_button.clicked.connect(
            lambda: getattr(self, 'stitching_canvas', None).move_selected_to_bottom() if hasattr(self,
                                                                                                 'stitching_canvas') else None)

        layers_layout.addLayout(layer_buttons_layout)

        self.stitch_layers_list = QListWidget()
        self.stitch_layers_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.stitch_layers_list.setStyleSheet("""
            QListWidget { background-color: transparent; border: none; outline: none; }
            QListWidget::item { padding: 8px 12px; border-radius: 6px; margin-bottom: 4px; color: #D0D0D0; background-color: rgba(255, 255, 255, 0.03); }
            QListWidget::item:hover { background-color: rgba(255, 255, 255, 0.08); color: #FFFFFF; }
            QListWidget::item:selected { background-color: #1A73E8; color: #FFFFFF; font-weight: bold; }
        """)
        self.stitch_layers_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.stitch_layers_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.stitch_layers_list.itemSelectionChanged.connect(self.on_stitch_list_selection_changed)

        layers_layout.addWidget(self.stitch_layers_list, 1)

        parent_layout.addWidget(layers_card, 1)

    def _build_segment_overlay_settings_panel(self, parent_layout: QVBoxLayout):
        """
        Build the smart working resolution and interactive function selection panel, forcing orig_text property injection.
        """
        self.seg_auto_tool = QPushButton()
        self.seg_auto_tool.setProperty("orig_text", "✨ 一键智能主体抠图")
        self.seg_auto_tool.setText(_TR("✨ 一键智能主体抠图"))
        self.seg_auto_tool.setStyleSheet(MODERN_PUSHBUTTON_STYLE + """
            QPushButton { background-color: #1A73E8; color: white; font-weight: bold; font-size: 13px; padding: 8px; }
            QPushButton:hover { background-color: #3B82F6; }
        """)
        self.seg_auto_tool.clicked.connect(self.auto_segment_action)
        parent_layout.addWidget(self.seg_auto_tool)

        # Build working resolution card
        res_card, res_layout = self.create_card_widget("工作分辨率")

        grid = QGridLayout()
        grid.setSpacing(6)

        self.res_original_radio = QRadioButton()
        self.res_original_radio.setProperty("orig_text", "原图尺寸")
        self.res_original_radio.setText(_TR("原图尺寸"))

        self.res_1920_radio = QRadioButton("1920px")
        self.res_1920_radio.setProperty("orig_text", "1920px")

        self.res_1280_radio = QRadioButton("1280px")
        self.res_1280_radio.setProperty("orig_text", "1280px")

        self.res_768_radio = QRadioButton("768px")
        self.res_768_radio.setProperty("orig_text", "768px")

        self.res_512_radio = QRadioButton()
        self.res_512_radio.setProperty("orig_text", "512像素 (推荐CPU)")
        self.res_512_radio.setText(_TR("512像素 (推荐CPU)"))

        self.res_custom_radio = QRadioButton()
        self.res_custom_radio.setProperty("orig_text", "自定义尺寸")
        self.res_custom_radio.setText(_TR("自定义尺寸"))

        radio_style = """
            QRadioButton {
                color: #E0E0E0;
                font-size: 12px;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
        """
        for r in [self.res_original_radio, self.res_1920_radio, self.res_1280_radio, self.res_768_radio,
                  self.res_512_radio, self.res_custom_radio]:
            r.setStyleSheet(radio_style)
            r.toggled.connect(self._on_work_resolution_mode_changed)

        grid.addWidget(self.res_original_radio, 0, 0)
        grid.addWidget(self.res_1920_radio, 0, 1)
        grid.addWidget(self.res_1280_radio, 1, 0)
        grid.addWidget(self.res_768_radio, 1, 1)
        grid.addWidget(self.res_512_radio, 2, 0)
        grid.addWidget(self.res_custom_radio, 2, 1)
        res_layout.addLayout(grid)

        custom_layout = QHBoxLayout()
        custom_layout.setSpacing(8)
        lbl = QLabel()
        lbl.setProperty("orig_text", "最大尺寸:")
        lbl.setText(_TR("最大尺寸:"))

        self.custom_max_dim_spinbox = QSpinBox()
        self.custom_max_dim_spinbox.setRange(128, 4096)
        self.custom_max_dim_spinbox.setValue(1280)
        self.custom_max_dim_spinbox.setStyleSheet(MODERN_SPINBOX_STYLE)
        custom_layout.addWidget(lbl)
        custom_layout.addWidget(self.custom_max_dim_spinbox)
        res_layout.addLayout(custom_layout)

        self.apply_resolution_button = QPushButton()
        self.apply_resolution_button.setProperty("orig_text", "应用并重载图层")
        self.apply_resolution_button.setText(_TR("应用并重载图层"))
        self.apply_resolution_button.setStyleSheet(MODERN_PUSHBUTTON_STYLE + """
            QPushButton { background-color: #333333; font-weight: bold; font-size: 12px; padding: 6px; }
            QPushButton:hover { background-color: #444444; }
        """)
        self.apply_resolution_button.clicked.connect(self._apply_and_reload_segmentation_image_with_new_resolution)
        res_layout.addWidget(self.apply_resolution_button)
        parent_layout.addWidget(res_card)

        # SAM2 interactive mode options
        sam_options_card, self.sam_options_layout = self.create_card_widget("智能框选模式 (SAM2)")
        self.cumulative_sam_checkbox = QCheckBox()
        self.cumulative_sam_checkbox.setProperty("orig_text", "在当前蒙版上累加 (Shift)")
        self.cumulative_sam_checkbox.setText(_TR("在当前蒙版上累加 (Shift)"))
        self.cumulative_sam_checkbox.setChecked(True)
        self.sam_options_layout.addWidget(self.cumulative_sam_checkbox)
        parent_layout.addWidget(sam_options_card)
        self.sam_options_card = sam_options_card

        # Brush manual repair mode options
        paint_options_card, self.paint_options_layout = self.create_card_widget("手工画笔模式 (Brush)")
        self.brush_controls_widget = QWidget()
        brush_controls_layout_inner = QVBoxLayout(self.brush_controls_widget)
        brush_controls_layout_inner.setContentsMargins(0, 0, 0, 0)

        brush_slider_layout = QHBoxLayout()
        thick_lbl = QLabel()
        thick_lbl.setProperty("orig_text", "粗细:")
        thick_lbl.setText(_TR("粗细:"))
        brush_slider_layout.addWidget(thick_lbl)

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 300)
        self.brush_slider.setValue(15)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        brush_slider_layout.addWidget(self.brush_slider)
        self.brush_size_label = QLabel("15px")
        brush_slider_layout.addWidget(self.brush_size_label)
        brush_controls_layout_inner.addLayout(brush_slider_layout)

        self.paint_render_mode_checkbox = QCheckBox()
        self.paint_render_mode_checkbox.setProperty("orig_text", "流畅笔迹实时预览")
        self.paint_render_mode_checkbox.setText(_TR("流畅笔迹实时预览"))
        self.paint_render_mode_checkbox.setChecked(True)
        self.paint_render_mode_checkbox.toggled.connect(self.on_paint_render_mode_changed)
        brush_controls_layout_inner.addWidget(self.paint_render_mode_checkbox)

        self.paint_options_layout.addWidget(self.brush_controls_widget)
        parent_layout.addWidget(paint_options_card)
        self.paint_options_card = paint_options_card

    def _build_segment_overlay_refine_panel(self, parent_layout: QVBoxLayout):
        refine_card, refine_layout = self.create_card_widget("蒙版边缘优化")

        self.matteformer_checkbox = QCheckBox()
        self.matteformer_checkbox.setProperty("orig_text", "启用极致发丝抠图 (耗时较长)")
        self.matteformer_checkbox.setText(_TR("启用极致发丝抠图 (耗时较长)"))
        self.matteformer_checkbox.setProperty("orig_tooltip", "开启后将强制进行一次全图边缘解算")
        self.matteformer_checkbox.setToolTip(_TR("开启后将强制进行一次全图边缘解算"))
        self.matteformer_checkbox.setEnabled(getattr(self, 'matteformer_loaded', False))

        def force_trigger_refine(checked):
            if hasattr(self, '_on_refinement_changed'):
                self._on_refinement_changed()

        self.matteformer_checkbox.toggled.connect(force_trigger_refine)
        refine_layout.addWidget(self.matteformer_checkbox)

        # Refinement slider row
        def create_slider_row(label_text, slider_obj, label_obj):
            row = QHBoxLayout()
            lbl = QLabel()
            lbl.setProperty("orig_text", label_text)
            lbl.setText(_TR(label_text))
            row.addWidget(lbl)
            slider_obj.setOrientation(Qt.Orientation.Horizontal)
            slider_obj.valueChanged.connect(self._on_refinement_changed)
            row.addWidget(slider_obj)
            row.addWidget(label_obj)
            return row

        self.smooth_slider = QSlider()
        self.smooth_slider.setRange(0, MAX_REFINE_SMOOTH)
        self.smooth_slider.setValue(DEFAULT_REFINE_SMOOTH)
        self.smooth_value_label = QLabel(str(DEFAULT_REFINE_SMOOTH))
        refine_layout.addLayout(create_slider_row("边缘平滑:", self.smooth_slider, self.smooth_value_label))

        self.feather_slider = QSlider()
        self.feather_slider.setRange(0, MAX_REFINE_FEATHER)
        self.feather_slider.setValue(DEFAULT_REFINE_FEATHER)
        self.feather_value_label = QLabel(str(DEFAULT_REFINE_FEATHER))
        refine_layout.addLayout(create_slider_row("边缘羽化:", self.feather_slider, self.feather_value_label))

        self.shift_slider = QSlider()
        self.shift_slider.setRange(-MAX_REFINE_SHIFT_SLIDER, MAX_REFINE_SHIFT_SLIDER)
        self.shift_slider.setValue(int(DEFAULT_REFINE_SHIFT_SLIDER))
        self.shift_value_label = QLabel(f"{DEFAULT_REFINE_SHIFT:.1f} px")
        refine_layout.addLayout(create_slider_row("收缩/扩张:", self.shift_slider, self.shift_value_label))

        parent_layout.addWidget(refine_card)

    def _build_segment_overlay_color_panel(self, parent_layout: QVBoxLayout):
        color_card, color_layout = self.create_card_widget("蒙版预览颜色")
        self.select_mask_color_button = QPushButton()
        self.select_mask_color_button.setProperty("orig_text", "选择预览颜色...")
        self.select_mask_color_button.setText(_TR("选择预览颜色..."))
        self.select_mask_color_button.setStyleSheet(MODERN_PUSHBUTTON_STYLE)
        self.select_mask_color_button.setMinimumHeight(40)
        self.select_mask_color_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.select_mask_color_button.clicked.connect(self._select_mask_preview_color)
        color_layout.addWidget(self.select_mask_color_button)
        QTimer.singleShot(0, self._update_color_preview_button_style)
        parent_layout.addWidget(color_card)

    def _populate_asset_library_panel(self, panel_frame: AssetPanelFrame):
        """Synchronously rearrange the left asset library to ensure perfect symmetry and multi-language responsiveness."""
        layout = panel_frame.content_layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        while layout.count() > 0:
            item = layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
            elif item:
                del item

        asset_title_label = QLabel()
        asset_title_label.setProperty("orig_text", "素材库")
        asset_title_label.setText(_TR("素材库"))
        asset_title_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
                padding-left: 4px;
                padding-top: 4px;
                padding-bottom: 2px;
            }
        """)
        layout.addWidget(asset_title_label)

        asset_scroll_area = QScrollArea()
        asset_scroll_area.setWidgetResizable(True)
        asset_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        asset_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        asset_scroll_area.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")

        self.stitch_asset_grid_widget = QWidget()
        self.stitch_asset_grid_widget.setStyleSheet("QWidget { background-color: transparent; }")
        self.stitch_asset_grid_layout = QGridLayout(self.stitch_asset_grid_widget)
        self.stitch_asset_grid_layout.setSpacing(14)
        self.stitch_asset_grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        asset_scroll_area.setWidget(self.stitch_asset_grid_widget)
        layout.addWidget(asset_scroll_area, 1)

        self.stitch_add_asset_button = QToolButton()
        self.stitch_add_asset_button.setStyleSheet(
            MODERN_TOOLBUTTON_STYLE + "QToolButton{ background: rgba(255,255,255,0.05); border-radius: 6px; }")
        self.stitch_add_asset_button.setProperty("orig_text", "添加新素材")
        self.stitch_add_asset_button.setText(_TR("添加新素材"))
        self.stitch_add_asset_button.setIcon(self._create_svg_icon("person-add.svg", color=QColor("#E0E0E0")))
        self.stitch_add_asset_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.stitch_add_asset_button.setIconSize(QSize(18, 18))
        self.stitch_add_asset_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.stitch_add_asset_button.clicked.connect(self.add_user_assets)

        layout.addWidget(self.stitch_add_asset_button)

        QTimer.singleShot(100, self.load_all_assets)