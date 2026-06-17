import os
import uuid
import math
import numpy as np

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QToolButton, QButtonGroup,
                               QMessageBox, QFileDialog, QGraphicsDropShadowEffect)
from PySide6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QCursor, QBrush,
                           QPaintEvent, QPainterPath, QTransform, QMouseEvent, QDragEnterEvent,
                           QDropEvent, QResizeEvent, QWheelEvent, QShowEvent, QKeyEvent, QImageReader)
from PySide6.QtCore import (Qt, QPoint, QPointF, QRect, QRectF, QSize, QSizeF, QTimer, Signal, Slot)

from config.settings import (STITCHING_DEFAULT_CANVAS_WIDTH, STITCHING_DEFAULT_CANVAS_HEIGHT,
                             MIN_ZOOM, MAX_ZOOM, STITCHING_HANDLE_SIZE, C_WINDOW_BG,
                             PREVIEW_BG_COLOR1, PREVIEW_BG_COLOR2)

# ==========================================
# Modern borderless top bar button style (Dark Refactored)
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


class FloatingPillToolbar(QWidget):
    """Modern floating capsule toolbar at the bottom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # Layout
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(20, 8, 20, 8)
        self.main_layout.setSpacing(12)

        # Drop shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 6)
        self.setGraphicsEffect(shadow)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Render capsule background (translucent dark gray)
        rect = self.rect().adjusted(2, 2, -2, -2)
        path = QPainterPath()
        path.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)

        painter.fillPath(path, QColor(30, 30, 30, 230))
        painter.setPen(QColor(60, 60, 60, 180))  # Subtle highlighted border
        painter.drawPath(path)

    def add_widget(self, widget):
        self.main_layout.addWidget(widget)

    def add_separator(self):
        sep = QWidget()
        sep.setFixedSize(1, 20)
        sep.setStyleSheet("background-color: #404040; border-radius: 0px;")
        self.main_layout.addWidget(sep)


class CanvasViewMixin:
    """Canvas page UI construction and event handling."""

    def setup_stitching_page(self):
        self.stitching_page = QWidget()
        self.stitching_page.setObjectName("StitchingPageContainer")
        # Apply global dark background style to the page
        self.stitching_page.setStyleSheet("QWidget#StitchingPageContainer { background-color: #181818; }")

        page_main_v_layout = QVBoxLayout(self.stitching_page)
        page_main_v_layout.setContentsMargins(10, 10, 10, 10)
        page_main_v_layout.setSpacing(10)

        self.top_button_bar_stitch = QWidget()
        top_button_bar_layout_stitch = QHBoxLayout(self.top_button_bar_stitch)
        top_button_bar_layout_stitch.setContentsMargins(0, 0, 0, 0)
        top_button_bar_layout_stitch.setSpacing(16)

        left_actions_stitch_widget = QWidget()
        left_actions_stitch_layout = QHBoxLayout(left_actions_stitch_widget)
        left_actions_stitch_layout.setContentsMargins(0, 0, 0, 0)
        left_actions_stitch_layout.setSpacing(6)
        left_actions_stitch_layout.setAlignment(Qt.AlignLeft)

        self.stitch_asset_library_toggle_button = QToolButton()
        self.stitch_asset_library_toggle_button.setText("素材库")
        self.stitch_asset_library_toggle_button.setIcon(
            self._create_svg_icon("palette.svg", size=32, color=QColor("#E0E0E0")))
        self.stitch_asset_library_toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.stitch_asset_library_toggle_button.setIconSize(QSize(32, 32))
        self.stitch_asset_library_toggle_button.setStyleSheet(MODERN_TOP_BAR_BUTTON_STYLE)
        self.stitch_asset_library_toggle_button.setCheckable(True)
        left_actions_stitch_layout.addWidget(self.stitch_asset_library_toggle_button)

        top_button_bar_layout_stitch.addWidget(left_actions_stitch_widget, 0)

        home_button_stitch_page = QToolButton(self.top_button_bar_stitch)
        home_button_stitch_page.setObjectName("FixedHomePageButton")
        home_button_stitch_page.setIcon(
            self._create_svg_icon("house.svg", size=24, color=QColor("#E0E0E0"), for_home_button=True))
        home_button_stitch_page.setToolTip("返回欢迎页面")
        home_button_stitch_page.clicked.connect(lambda: self.switch_page_with_slide(self.WELCOME_PAGE_INDEX))
        top_button_bar_layout_stitch.addWidget(home_button_stitch_page, 0, Qt.AlignCenter)

        right_panel_buttons_stitch_widget = QWidget()
        right_panel_buttons_stitch_layout = QHBoxLayout(right_panel_buttons_stitch_widget)
        right_panel_buttons_stitch_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_buttons_stitch_layout.setSpacing(6)
        right_panel_buttons_stitch_layout.setAlignment(Qt.AlignRight)

        stitch_actions_button = QToolButton()
        stitch_actions_button.setText("操作")
        stitch_actions_button.setIcon(self._create_svg_icon("play.svg", size=32, color=QColor("#E0E0E0")))
        stitch_actions_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        stitch_actions_button.setIconSize(QSize(32, 32))
        stitch_actions_button.setStyleSheet(MODERN_TOP_BAR_BUTTON_STYLE)
        stitch_actions_button.setCheckable(True)

        stitch_canvas_button = QToolButton()
        stitch_canvas_button.setText("画布")
        stitch_canvas_button.setIcon(self._create_svg_icon("image.svg", size=32, color=QColor("#E0E0E0")))
        stitch_canvas_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        stitch_canvas_button.setIconSize(QSize(32, 32))
        stitch_canvas_button.setStyleSheet(MODERN_TOP_BAR_BUTTON_STYLE)
        stitch_canvas_button.setCheckable(True)

        stitch_layers_button = QToolButton()
        stitch_layers_button.setText("图层")
        stitch_layers_button.setIcon(self._create_svg_icon("layers.svg", size=32, color=QColor("#E0E0E0")))
        stitch_layers_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        stitch_layers_button.setIconSize(QSize(32, 32))
        stitch_layers_button.setStyleSheet(MODERN_TOP_BAR_BUTTON_STYLE)
        stitch_layers_button.setCheckable(True)

        stitch_top_panel_button_group = QButtonGroup(self)
        stitch_top_panel_button_group.setExclusive(True)
        stitch_top_panel_button_group.addButton(stitch_actions_button)
        stitch_top_panel_button_group.addButton(stitch_canvas_button)
        stitch_top_panel_button_group.addButton(stitch_layers_button)
        self.page_top_button_groups['stitching'] = stitch_top_panel_button_group

        right_panel_buttons_stitch_layout.addWidget(stitch_actions_button)
        right_panel_buttons_stitch_layout.addWidget(stitch_canvas_button)
        right_panel_buttons_stitch_layout.addWidget(stitch_layers_button)
        top_button_bar_layout_stitch.addWidget(right_panel_buttons_stitch_widget, 0)

        page_main_v_layout.addWidget(self.top_button_bar_stitch)

        self.stitching_canvas = StitchingCanvas(self)
        page_main_v_layout.addWidget(self.stitching_canvas, 1)

        self.stacked_widget.addWidget(self.stitching_page)

        self._populate_asset_library_panel(self.asset_library_panel_floating)

        self.stitch_asset_library_toggle_button.clicked.connect(self.toggle_asset_library)
        stitch_actions_button.clicked.connect(
            lambda: self._on_top_panel_button_clicked(stitch_actions_button, "stitch_actions"))
        stitch_canvas_button.clicked.connect(
            lambda: self._on_top_panel_button_clicked(stitch_canvas_button, "stitch_canvas_settings"))
        stitch_layers_button.clicked.connect(
            lambda: self._on_top_panel_button_clicked(stitch_layers_button, "stitch_layers"))

        self.stitching_canvas.selection_changed.connect(self.on_stitch_selection_changed)
        self.stitching_canvas.layers_changed.connect(self.update_stitch_layers_list)
        self.stitching_canvas.canvas_resized.connect(self.on_stitch_canvas_resized)

        self.stitching_canvas.set_canvas_size(STITCHING_DEFAULT_CANVAS_WIDTH, STITCHING_DEFAULT_CANVAS_HEIGHT)
        self.stitching_canvas.set_background_color(QColor(Qt.GlobalColor.transparent))


class StitchedImageItem:
    """Extended canvas item data model supporting enhancement, matting, and grouping directly on the canvas."""

    def __init__(self, pixmap: QPixmap, name: str, original_path: str):
        self.id = str(uuid.uuid4())
        self.pixmap = pixmap
        self.name = name
        self.pos = QPointF(0, 0)
        self.size = QSizeF(pixmap.size())
        self.rotation = 0.0
        self.is_selected = False

        self.original_image_path = original_path
        self.segmentation_mask_np: np.ndarray | None = None
        self.is_enhanced = False
        self.has_alpha_channel = pixmap.hasAlphaChannel()

        self.source_items: list['StitchedImageItem'] | None = None

    def get_bounding_rect(self) -> QRectF:
        return QRectF(self.pos, self.size)

    def get_transform(self) -> QTransform:
        transform = QTransform()
        transform.translate(self.pos.x(), self.pos.y())
        transform.translate(self.size.width() / 2, self.size.height() / 2)
        transform.rotate(self.rotation)
        transform.translate(-self.size.width() / 2, -self.size.height() / 2)
        return transform

    def get_transformed_bounding_rect(self) -> QRectF:
        return self.get_transform().mapRect(QRectF(QPointF(0, 0), self.size))

    def get_handles_local(self) -> dict[str, QPointF]:
        handles = {}
        w, h = self.size.width(), self.size.height()
        handles['scale_tl'] = QPointF(0, 0)
        handles['scale_tr'] = QPointF(w, 0)
        handles['scale_bl'] = QPointF(0, h)
        handles['scale_br'] = QPointF(w, h)
        handles['rotate'] = QPointF(w / 2, -STITCHING_HANDLE_SIZE * 2.5)
        return handles


class StitchingCanvas(QWidget):
    """Stitching canvas control managing all rendering, dragging, scaling, and rotation logic."""

    selection_changed = Signal()
    layers_changed = Signal()
    canvas_resized = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_app_window = None

        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.items: list[StitchedImageItem] = []
        self.selected_items: list[StitchedImageItem] = []
        self.background_color = QColor(Qt.GlobalColor.transparent)
        self.canvas_pos = QPointF(200, 150)
        self.canvas_size = QSize(STITCHING_DEFAULT_CANVAS_WIDTH, STITCHING_DEFAULT_CANVAS_HEIGHT)

        self.zoom_scale = 1.0
        self.view_offset = QPointF(0, 0)
        self._is_user_zoomed = False
        self._initial_fit_done = False

        self._is_panning = False
        self._is_dragging_item = False
        self._is_scaling = False
        self._is_rotating = False
        self._is_resizing_canvas = False

        self._last_pan_pos = QPoint()
        self._active_handle = None
        self._drag_start_pos = QPointF()
        self._drag_items_start_states: dict[str, tuple[QPointF, QSizeF, float]] = {}

        self._canvas_resize_handle = None
        self._canvas_drag_start_rect = QRectF()

        self._last_mouse_pos = QPoint()
        self._cursor_cache = {}
        self._checkerboard_pixmap = None

    def showEvent(self, event: QShowEvent):
        if not self._initial_fit_done:
            self._is_user_zoomed = False
            QTimer.singleShot(0, self.fit_canvas_to_view)
            self._initial_fit_done = True
        super().showEvent(event)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)

        if not self._is_user_zoomed:
            self.fit_canvas_to_view()
            return

        old_size = event.oldSize()
        new_size = self.size()

        if old_size.width() <= 0 or old_size.height() <= 0 or old_size == new_size:
            return

        old_center_widget = QPoint(old_size.width() // 2, old_size.height() // 2)
        center_canvas_point = self.map_widget_to_canvas(old_center_widget)

        new_center_widget = QPoint(new_size.width() // 2, new_size.height() // 2)
        new_offset_x = -center_canvas_point.x() * self.zoom_scale + new_center_widget.x()
        new_offset_y = -center_canvas_point.y() * self.zoom_scale + new_center_widget.y()
        self.view_offset = QPointF(new_offset_x, new_offset_y)

        self.update()

    def fit_canvas_to_view(self, zoom_to_fit=True):
        if self.canvas_size.width() <= 0 or self.width() <= 0 or self.height() <= 0: return

        if zoom_to_fit:
            padding_factor = 0.95
            scale_x = (self.width() * padding_factor) / self.canvas_size.width()
            scale_y = (self.height() * padding_factor) / self.canvas_size.height()

            self.zoom_scale = min(scale_x, scale_y)
            self.zoom_scale = max(MIN_ZOOM, min(self.zoom_scale, MAX_ZOOM))
            self._is_user_zoomed = False

        self.view_offset = QPointF(
            (self.width() - self.canvas_size.width() * self.zoom_scale) / 2 - self.canvas_pos.x() * self.zoom_scale,
            (self.height() - self.canvas_size.height() * self.zoom_scale) / 2 - self.canvas_pos.y() * self.zoom_scale
        )
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1 / 1.15
        old_scale = self.zoom_scale
        new_scale = old_scale * zoom_factor
        self.zoom_scale = max(MIN_ZOOM, min(new_scale, MAX_ZOOM))
        if abs(self.zoom_scale - old_scale) < 1e-6: return

        self._is_user_zoomed = True

        mouse_pos = event.position()
        self.view_offset = mouse_pos - (mouse_pos - self.view_offset) * (self.zoom_scale / old_scale)

        self.update()
        event.accept()

    def set_canvas_size(self, width: int, height: int):
        new_size = QSize(max(100, width), max(100, height))
        if self.canvas_size != new_size:
            self.canvas_size = new_size
            self.canvas_resized.emit(new_size.width(), new_size.height())
            self.update()

    def set_canvas_width(self, width: int):
        self.set_canvas_size(width, self.canvas_size.height())

    def set_canvas_height(self, height: int):
        self.set_canvas_size(self.canvas_size.width(), height)

    def set_background_color(self, color: QColor):
        self.background_color = color
        self.update()

    def update_background(self, is_transparent: bool, color: QColor = QColor(Qt.GlobalColor.white)):
        if is_transparent:
            self.set_background_color(QColor(Qt.GlobalColor.transparent))
        else:
            self.set_background_color(color)

    def load_images(self):
        supported_formats = "图像文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;PNG 图像 (*.png);;所有文件 (*)"
        files, _ = QFileDialog.getOpenFileNames(self, "选择要拼接的图像 (可多选)", "", supported_formats)
        if files: self._process_dropped_files(files)

    def clear_all(self, confirm: bool = True):
        if not self.items:
            if confirm:
                QMessageBox.information(self, "提示", "画布上已经没有内容了。")
            return

        do_clear = False
        if confirm:
            reply = QMessageBox.question(self, "清空画布", "您确定要移除画布上的所有图像吗？\n此操作无法撤销。",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                do_clear = True
        else:
            do_clear = True

        if do_clear:
            self.items.clear()
            self.selected_items.clear()
            self.layers_changed.emit()
            self.selection_changed.emit()
            self.update()

    def _create_rotated_arrow_cursor(self, angle: float) -> QCursor:
        angle_key = int(round(angle)) % 360
        cache_key = f"bidirectional_arrow_{angle_key}"
        if cache_key in self._cursor_cache:
            return self._cursor_cache[cache_key]

        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(16, 16)
        painter.rotate(angle)

        outline_pen = QPen(QColor(0, 0, 0), 1.2)
        outline_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        fill_brush = QBrush(QColor(255, 255, 255))

        arrow_path = QPainterPath()
        head_len, head_half_width, shaft_end = 5.0, 4.0, 8.0
        arrow_path.moveTo(shaft_end, 0)
        arrow_path.lineTo(shaft_end - head_len, -head_half_width)
        arrow_path.lineTo(shaft_end - head_len, head_half_width)
        arrow_path.closeSubpath()

        painter.setPen(outline_pen)
        painter.setBrush(fill_brush)
        painter.drawRect(QRectF(-(shaft_end - head_len), -1, (shaft_end - head_len) * 2, 2))

        painter.drawPath(arrow_path)
        painter.save()
        painter.rotate(180)
        painter.drawPath(arrow_path)
        painter.restore()
        painter.end()

        new_cursor = QCursor(pixmap, 16, 16)
        self._cursor_cache[cache_key] = new_cursor
        return new_cursor

    def get_primary_selected_item(self) -> StitchedImageItem | None:
        return self.selected_items[-1] if self.selected_items else None

    def select_items_by_ids(self, ids: list[str]):
        new_selection = [item for item in self.items if item.id in ids]
        if set(item.id for item in self.selected_items) != set(ids):
            self.selected_items = new_selection
            self.selection_changed.emit()
            self.update()

    def delete_selected_item(self):
        if self.selected_items:
            self.items = [item for item in self.items if item not in self.selected_items]
            self.selected_items.clear()
            self.layers_changed.emit()
            self.selection_changed.emit()
            self.update()

    def move_selected_layer(self, delta: int):
        primary_item = self.get_primary_selected_item()
        if not primary_item: return
        try:
            idx = self.items.index(primary_item)
            new_idx = idx + delta
            if 0 <= new_idx < len(self.items):
                self.items.pop(idx)
                self.items.insert(new_idx, primary_item)
                self.layers_changed.emit()
                self.update()
        except ValueError:
            pass

    def move_selected_to_top(self):
        if not self.selected_items: return
        for item in self.selected_items:
            if item in self.items:
                self.items.remove(item)
                self.items.append(item)
        self.layers_changed.emit()
        self.update()

    def move_selected_to_bottom(self):
        if not self.selected_items: return
        for item in reversed(self.selected_items):
            if item in self.items:
                self.items.remove(item)
                self.items.insert(0, item)
        self.layers_changed.emit()
        self.update()

    def reorder_from_list(self, source_parent, source_start, source_end, dest_parent, dest_row):
        new_item_order = []
        for i in range(self.parent_window.stitch_layers_list.count()):
            list_item = self.parent_window.stitch_layers_list.item(i)
            item_id = list_item.data(Qt.ItemDataRole.UserRole)
            found_item = next((item for item in self.items if item.id == item_id), None)
            if found_item: new_item_order.append(found_item)
        self.items = list(reversed(new_item_order))
        self.layers_changed.emit()
        self.update()

    def update_selected_item_property(self, prop, value):
        primary_item = self.get_primary_selected_item()
        if primary_item and not primary_item.pixmap.isNull():
            parent_window = self.main_app_window

            spinbox_names_on_parent = [
                'stitch_pos_x_spin', 'stitch_pos_y_spin',
                'stitch_size_w_spin', 'stitch_size_h_spin', 'stitch_rotation_spin'
            ]
            parent_has_spinboxes = parent_window and all(
                hasattr(parent_window, name) for name in spinbox_names_on_parent)

            spinboxes_to_block = []
            if parent_has_spinboxes:
                spinboxes_to_block = [getattr(parent_window, name) for name in spinbox_names_on_parent]
                for w in spinboxes_to_block: w.blockSignals(True)

            try:
                if prop == 'x':
                    primary_item.pos.setX(value)
                elif prop == 'y':
                    primary_item.pos.setY(value)
                elif prop == 'width':
                    aspect_ratio = primary_item.pixmap.height() / primary_item.pixmap.width() if primary_item.pixmap.width() > 0 else 1
                    primary_item.size.setWidth(value)
                    new_height = value * aspect_ratio
                    primary_item.size.setHeight(new_height)
                    if parent_has_spinboxes:
                        parent_window.stitch_size_h_spin.setValue(new_height)
                elif prop == 'height':
                    aspect_ratio = primary_item.pixmap.width() / primary_item.pixmap.height() if primary_item.pixmap.height() > 0 else 1
                    primary_item.size.setHeight(value)
                    new_width = value * aspect_ratio
                    primary_item.size.setWidth(new_width)
                    if parent_has_spinboxes:
                        parent_window.stitch_size_w_spin.setValue(new_width)
                elif prop == 'rotation':
                    primary_item.rotation = value
            finally:
                if parent_has_spinboxes:
                    for w in spinboxes_to_block: w.blockSignals(False)

            self.update()

    def map_widget_to_canvas(self, pos: QPoint) -> QPointF:
        return (QPointF(pos) - self.view_offset) / self.zoom_scale

    def map_canvas_to_widget(self, pos: QPointF) -> QPoint:
        return (pos * self.zoom_scale + self.view_offset).toPoint()

    def _get_item_at(self, pos: QPointF) -> StitchedImageItem | None:
        for item in reversed(self.items):
            if item.get_transformed_bounding_rect().contains(pos): return item
        return None

    def _get_handle_at(self, item: StitchedImageItem, pos: QPointF) -> str | None:
        inv_transform, _ = item.get_transform().inverted()
        local_pos = inv_transform.map(pos)
        logical_handle_size = (STITCHING_HANDLE_SIZE + 6) / self.zoom_scale
        for name, handle_pos_local in item.get_handles_local().items():
            handle_hitbox = QRectF(handle_pos_local - QPointF(logical_handle_size / 2, logical_handle_size / 2),
                                   QSizeF(logical_handle_size, logical_handle_size))
            if handle_hitbox.contains(local_pos): return name
        return None

    def _get_canvas_resize_handle_at(self, widget_pos: QPoint) -> str | None:
        margin = 10
        canvas_rect_widget = QRect(
            self.map_canvas_to_widget(self.canvas_pos),
            QSize(int(self.canvas_size.width() * self.zoom_scale),
                  int(self.canvas_size.height() * self.zoom_scale))
        )
        on_left = abs(widget_pos.x() - canvas_rect_widget.left()) < margin
        on_right = abs(widget_pos.x() - canvas_rect_widget.right()) < margin
        on_top = abs(widget_pos.y() - canvas_rect_widget.top()) < margin
        on_bottom = abs(widget_pos.y() - canvas_rect_widget.bottom()) < margin

        if on_top and on_left: return 'top-left'
        if on_top and on_right: return 'top-right'
        if on_bottom and on_left: return 'bottom-left'
        if on_bottom and on_right: return 'bottom-right'
        if on_top: return 'top'
        if on_bottom: return 'bottom'
        if on_left: return 'left'
        if on_right: return 'right'
        return None

    def _process_dropped_files(self, file_paths: list[str], drop_pos_widget: QPoint = None, preview_image=None):
        if not self.main_app_window:
            return

        items_added, newly_added_items = False, []

        for i, file_path in enumerate(file_paths):
            # Enqueue recent projects task to avoid blocking layout interactions
            QTimer.singleShot(0, lambda p=file_path: self.main_app_window.add_to_recent_projects(p))

            name = os.path.splitext(os.path.basename(file_path))[0]

            # 1. [Instant metadata check]: Read only image header info to retrieve physical resolution under 1ms
            real_w, real_h = 800, 800
            try:
                reader = QImageReader(file_path)
                size = reader.size()
                if size.isValid():
                    real_w = size.width()
                    real_h = size.height()
            except Exception:
                pass

            # 2. Calculate optimal canvas rendering size maintaining aspect ratio (fit within 70% of canvas)
            canvas_w, canvas_h = self.canvas_size.width(), self.canvas_size.height()
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

            # 3. [Zero-latency stretch]: Force-stretch the drag-preview thumbnail to computed physical dimensions
            if preview_image is not None and not preview_image.isNull():
                fast_preview = QPixmap.fromImage(preview_image).scaled(
                    final_w, final_h,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            else:
                fast_preview = QPixmap(final_w, final_h)
                fast_preview.fill(QColor(40, 40, 40, 180))

            item = StitchedImageItem(fast_preview, name, original_path=file_path)
            item.size = QSizeF(final_w, final_h)

            # 4. Position target relative to drop point
            if drop_pos_widget:
                target_center_canvas = self.map_widget_to_canvas(drop_pos_widget)
                target_center_canvas += QPointF(i * 20, i * 20)
            else:
                target_center_canvas = self.canvas_pos + QPointF(canvas_w / 2, canvas_h / 2)

            item.pos = target_center_canvas - QPointF(final_w / 2, final_h / 2)

            self.items.append(item)
            newly_added_items.append(item)
            items_added = True

            # 5. [Async high-res swapping]: Immediate visual feedback with proxy; full-resolution data loads in the background
            self.main_app_window.start_async_image_load(file_path, item.id)

        if items_added:
            self.layers_changed.emit()
            self.selected_items = newly_added_items
            self.selection_changed.emit()
            self.update()

    @Slot(str, QPixmap)
    def update_item_pixmap(self, item_id: str, new_pixmap: QPixmap):
        for item in self.items:
            if item.id == item_id:
                item.pixmap = new_pixmap
                old_size = item.size
                new_size_raw = new_pixmap.size()

                if new_size_raw.width() > 0 and new_size_raw.height() > 0:
                    scale_factor = old_size.width() / new_size_raw.width()
                    new_size = QSizeF(new_size_raw.width() * scale_factor, new_size_raw.height() * scale_factor)

                    # [Shift Prevention Fix]: Compute center point coordinates of original boundary
                    old_center = item.pos + QPointF(old_size.width() / 2, old_size.height() / 2)

                    item.size = new_size
                    # Recenter the new high-resolution bounding box around original pivot to prevent shifting jumps
                    item.pos = old_center - QPointF(new_size.width() / 2, new_size.height() / 2)

                item.has_alpha_channel = new_pixmap.hasAlphaChannel()
                self.update()
                break

    @Slot(str)
    def handle_item_load_error(self, item_id: str):
        for item in self.items:
            if item.id == item_id:
                self.items.remove(item)
                if item in self.selected_items:
                    self.selected_items.remove(item)

                self.layers_changed.emit()
                self.selection_changed.emit()
                self.update()

                if self.main_app_window:
                    QMessageBox.warning(self.main_app_window, "加载失败",
                                        f"无法加载素材：\n{os.path.basename(item.original_image_path)}")
                break

    def map_canvas_to_widget_rect(self, canvas_rect: QRectF) -> QRect:
        tl_widget = self.map_canvas_to_widget(canvas_rect.topLeft())
        br_widget = self.map_canvas_to_widget(canvas_rect.bottomRight())
        return QRect(tl_widget, br_widget).normalized()

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse_pos = event.position().toPoint()
        canvas_pos_coords = self.map_widget_to_canvas(self._last_mouse_pos)
        modifiers = event.modifiers()

        if event.button() == Qt.MouseButton.LeftButton:
            canvas_handle = self._get_canvas_resize_handle_at(self._last_mouse_pos)
            if canvas_handle:
                self._is_resizing_canvas = True
                self._canvas_resize_handle = canvas_handle
                self._canvas_drag_start_rect = QRectF(self.canvas_pos, self.canvas_size)
                self.update_cursor(self._last_mouse_pos)
                event.accept()
                return

            primary_selected = self.get_primary_selected_item()
            if primary_selected:
                handle = self._get_handle_at(primary_selected, canvas_pos_coords)
                if handle:
                    if handle.startswith('scale'):
                        self._is_scaling, self._active_handle = True, handle
                    elif handle == 'rotate':
                        self._is_rotating = True
                    self._drag_start_pos = canvas_pos_coords
                    self._drag_items_start_states = {item.id: (item.pos, item.size, item.rotation) for item in
                                                     self.selected_items}
                    self.update_cursor(self._last_mouse_pos)
                    event.accept()
                    return

            item_under_mouse = self._get_item_at(canvas_pos_coords)
            if item_under_mouse:
                if modifiers & Qt.KeyboardModifier.ControlModifier:
                    if item_under_mouse in self.selected_items:
                        self.selected_items.remove(item_under_mouse)
                    else:
                        self.selected_items.append(item_under_mouse)
                elif item_under_mouse not in self.selected_items:
                    self.selected_items = [item_under_mouse]

                self._is_dragging_item = True
                self._drag_start_pos = canvas_pos_coords
                self._drag_items_start_pos = {item.id: item.pos for item in self.selected_items}

                # [Interaction Feedback Fix]: Instantly switch cursor shape to ClosedHand on item grab
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

                self.selection_changed.emit()
                self.update()
                event.accept()
                return

            if not (modifiers & Qt.KeyboardModifier.ControlModifier) and self.selected_items:
                self.selected_items.clear()
                self.selection_changed.emit()
                self.update()

        elif event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._last_pan_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        current_pos = event.position().toPoint()

        if self._is_panning:
            delta = current_pos - self._last_pan_pos
            self.view_offset += QPointF(delta)
            self._last_pan_pos = current_pos
            self.update()
            event.accept()
            return

        canvas_pos_coords = self.map_widget_to_canvas(current_pos)

        # Resize canvas boundary
        if self._is_resizing_canvas:
            start_rect = self._canvas_drag_start_rect
            start_left, start_top, start_right, start_bottom = start_rect.left(), start_rect.top(), start_rect.right(), start_rect.bottom()
            new_left, new_top, new_right, new_bottom = start_left, start_top, start_right, start_bottom

            handle = self._canvas_resize_handle
            if 'left' in handle: new_left = canvas_pos_coords.x()
            if 'right' in handle: new_right = canvas_pos_coords.x()
            if 'top' in handle: new_top = canvas_pos_coords.y()
            if 'bottom' in handle: new_bottom = canvas_pos_coords.y()

            min_size = 50.0
            if new_right - new_left < min_size:
                if 'right' in handle:
                    new_right = new_left + min_size
                else:
                    new_left = new_right - min_size
            if new_bottom - new_top < min_size:
                if 'bottom' in handle:
                    new_bottom = new_top + min_size
                else:
                    new_top = new_bottom - min_size

            self.canvas_pos = QPointF(new_left, new_top)
            self.set_canvas_size(int(new_right - new_left), int(new_bottom - new_top))
            event.accept()
            return

        # [Core Fix]: Handle drag movement.
        # Omit inner selection_changed.emit() to eliminate lag caused by frequent inspector updates.
        if self._is_dragging_item and self.selected_items:
            delta = canvas_pos_coords - self._drag_start_pos
            for item in self.selected_items:
                if item.id in self._drag_items_start_pos:
                    item.pos = self._drag_items_start_pos[item.id] + delta

            # Render using paintEvent dynamic rendering degradation for smooth tracking
            self.update()
            event.accept()
            return

        # Rotate
        if self._is_rotating and self.selected_items:
            start_group_bounds = QRectF()
            for item in self.selected_items:
                start_pos, start_size, start_rotation = self._drag_items_start_states[item.id]
                t = QTransform().translate(start_pos.x() + start_size.width() / 2,
                                           start_pos.y() + start_size.height() / 2).rotate(start_rotation).translate(
                    -start_size.width() / 2, -start_size.height() / 2)
                item_bounds = t.mapRect(QRectF(QPointF(0, 0), start_size))
                start_group_bounds = start_group_bounds.united(item_bounds)

            group_pivot = start_group_bounds.center() if not start_group_bounds.isNull() else QPointF()
            start_vec = self._drag_start_pos - group_pivot
            current_vec = canvas_pos_coords - group_pivot
            delta_angle = math.degrees(
                math.atan2(current_vec.y(), current_vec.x()) - math.atan2(start_vec.y(), start_vec.x()))

            for item in self.selected_items:
                start_pos, start_size, start_rotation = self._drag_items_start_states[item.id]
                item.rotation = (start_rotation + delta_angle + 360) % 360
                vec_to_item_center = (start_pos + QPointF(start_size.width() / 2,
                                                          start_size.height() / 2)) - group_pivot
                rotated_vec = QTransform().rotate(delta_angle).map(vec_to_item_center)
                item.pos = (group_pivot + rotated_vec) - QPointF(start_size.width() / 2, start_size.height() / 2)

            self.update()
            event.accept()
            return

        # Scale
        if self._is_scaling and self.selected_items:
            primary_item = self.get_primary_selected_item()
            if not primary_item: return
            start_primary_pos, start_primary_size, start_primary_rotation = self._drag_items_start_states[
                primary_item.id]
            start_transform = QTransform().translate(start_primary_pos.x() + start_primary_size.width() / 2,
                                                     start_primary_pos.y() + start_primary_size.height() / 2).rotate(
                start_primary_rotation).translate(-start_primary_size.width() / 2, -start_primary_size.height() / 2)

            w, h = start_primary_size.width(), start_primary_size.height()
            handle_map_local = {'scale_br': (QPointF(0, 0), QPointF(w, h)), 'scale_tl': (QPointF(w, h), QPointF(0, 0)),
                                'scale_tr': (QPointF(0, h), QPointF(w, 0)), 'scale_bl': (QPointF(w, 0), QPointF(0, h))}
            anchor_local, moving_local = handle_map_local.get(self._active_handle, (None, None))
            if anchor_local is None: return

            anchor_world_start, moving_world_start = start_transform.map(anchor_local), start_transform.map(
                moving_local)
            vec_start = moving_world_start - anchor_world_start
            vec_current = canvas_pos_coords - anchor_world_start

            dot_start = QPointF.dotProduct(vec_start, vec_start)
            scale_factor = QPointF.dotProduct(vec_current, vec_start) / dot_start if dot_start > 1e-6 else 1.0

            min_size = 10.0
            if start_primary_size.width() * scale_factor < min_size or start_primary_size.height() * scale_factor < min_size:
                scale_factor = max(min_size / start_primary_size.width() if start_primary_size.width() > 0 else 1,
                                   min_size / start_primary_size.height() if start_primary_size.height() > 0 else 1)

            for item in self.selected_items:
                start_pos, start_size, start_rotation = self._drag_items_start_states[item.id]
                item.size = start_size * scale_factor
                item_start_center = start_pos + QPointF(start_size.width() / 2, start_size.height() / 2)
                item.pos = (anchor_world_start + (item_start_center - anchor_world_start) * scale_factor) - QPointF(
                    item.size.width() / 2, item.size.height() / 2)
                item.rotation = start_rotation

            self.update()
            event.accept()
            return

        self.update_cursor(current_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            was_interacting = self._is_dragging_item or self._is_scaling or self._is_rotating or self._is_resizing_canvas

            self._is_dragging_item = False
            self._is_scaling = False
            self._is_rotating = False
            self._active_handle = None
            self._is_resizing_canvas = False
            self._canvas_resize_handle = None

            self.update_cursor(event.position().toPoint())

            if was_interacting:
                self.selection_changed.emit()
                self.update()

            event.accept()

        elif event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self.update_cursor(event.position().toPoint())
            self.update()
            event.accept()

        else:
            super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
            if any(u.toLocalFile().lower().endswith(supported_formats) for u in urls):
                event.setDropAction(Qt.DropAction.MoveAction)
                event.accept()
                return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            file_paths = [url.toLocalFile() for url in event.mimeData().urls()]
            drop_position_widget = event.position().toPoint()

            preview_image = None
            if event.mimeData().hasImage():
                preview_image = event.mimeData().imageData()

            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()

            QTimer.singleShot(10, lambda: self._process_dropped_files(file_paths, drop_position_widget, preview_image))
        else:
            event.ignore()

    def update_cursor(self, widget_pos: QPoint):
        if not self.main_app_window:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if any([self._is_panning, self._is_dragging_item, self._is_scaling, self._is_rotating,
                self._is_resizing_canvas]):
            return

        canvas_pos = self.map_widget_to_canvas(widget_pos)

        canvas_handle = self._get_canvas_resize_handle_at(widget_pos)
        if canvas_handle:
            if ('top' in canvas_handle and 'left' in canvas_handle) or \
                    ('bottom' in canvas_handle and 'right' in canvas_handle):
                cursor_shape = Qt.CursorShape.SizeFDiagCursor
            elif ('top' in canvas_handle and 'right' in canvas_handle) or \
                    ('bottom' in canvas_handle and 'left' in canvas_handle):
                cursor_shape = Qt.CursorShape.SizeBDiagCursor
            elif 'top' in canvas_handle or 'bottom' in canvas_handle:
                cursor_shape = Qt.CursorShape.SizeVerCursor
            else:
                cursor_shape = Qt.CursorShape.SizeHorCursor
            self.setCursor(cursor_shape)
            return

        cursor_to_set = None
        primary_selected = self.get_primary_selected_item()
        if primary_selected:
            handle = self._get_handle_at(primary_selected, canvas_pos)
            if handle:
                if handle == 'rotate':
                    cursor_to_set = self.main_app_window.rotate_cursor
                elif handle.startswith('scale'):
                    transform = primary_selected.get_transform()
                    handle_map_local = primary_selected.get_handles_local()
                    opposite_map = {'scale_br': 'scale_tl', 'scale_tl': 'scale_br', 'scale_tr': 'scale_bl',
                                    'scale_bl': 'scale_tr'}
                    opposite_handle_name = opposite_map.get(handle)
                    if opposite_handle_name:
                        p1_local, p2_local = handle_map_local[handle], handle_map_local[opposite_handle_name]
                        p1_widget, p2_widget = self.map_canvas_to_widget(
                            transform.map(p1_local)), self.map_canvas_to_widget(transform.map(p2_local))
                        screen_vector = QPointF(p2_widget - p1_widget)
                        final_cursor_angle = math.degrees(math.atan2(screen_vector.y(), screen_vector.x()))
                        cursor_to_set = self._create_rotated_arrow_cursor(final_cursor_angle)

        if cursor_to_set is None:
            if self._get_item_at(canvas_pos):
                cursor_to_set = Qt.CursorShape.SizeAllCursor
            else:
                cursor_to_set = Qt.CursorShape.OpenHandCursor

        self.setCursor(cursor_to_set)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)

        is_interacting = (self._is_dragging_item or self._is_panning or
                          self._is_scaling or self._is_rotating or self._is_resizing_canvas)

        # Performance optimization: Disable antialiasing during heavy interactive adjustments
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, not is_interacting)

        # 1. Render bottom theme-consistent dark background
        painter.fillRect(self.rect(), QColor("#181818"))

        # 2. Render rounded work viewport background
        white_bg_rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1C1C1C"))
        painter.drawRoundedRect(white_bg_rect, 15, 15)

        # 3. Apply rounded viewport clipping path
        clip_path = QPainterPath()
        clip_path.addRoundedRect(white_bg_rect, 15, 15)
        painter.setClipPath(clip_path)

        # 4. Apply viewport translation and scale transformations
        painter.save()
        painter.translate(self.view_offset)
        painter.scale(self.zoom_scale, self.zoom_scale)

        # 5. Render canvas checkerboard/background structure
        canvas_rect_local = QRectF(self.canvas_pos, self.canvas_size)
        if self.background_color.alpha() == 0:
            view_rect_canvas = self.map_widget_to_canvas(self.rect().topLeft())
            visible_checker_area = canvas_rect_local.intersected(
                QRectF(view_rect_canvas, QSizeF(self.width() / self.zoom_scale, self.height() / self.zoom_scale))
            )

            if visible_checker_area.isValid() and not visible_checker_area.isEmpty():
                painter.save()
                painter.setClipRect(visible_checker_area)
                checker_size = 32
                brush1, brush2 = QBrush(PREVIEW_BG_COLOR1), QBrush(PREVIEW_BG_COLOR2)
                painter.setPen(Qt.PenStyle.NoPen)

                start_x = int(visible_checker_area.left() / checker_size) * checker_size
                start_y = int(visible_checker_area.top() / checker_size) * checker_size

                for y in range(start_y, int(visible_checker_area.bottom()), checker_size):
                    for x in range(start_x, int(visible_checker_area.right()), checker_size):
                        painter.setBrush(brush1 if ((x // checker_size) % 2 == (y // checker_size) % 2) else brush2)
                        painter.drawRect(x, y, checker_size, checker_size)
                painter.restore()
        else:
            painter.fillRect(canvas_rect_local, self.background_color)

        # 6. Render canvas asset elements
        # Performance optimization: Disable bilinear filtering during dragging states
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, not is_interacting)

        for item in self.items:
            painter.save()
            painter.setTransform(item.get_transform(), True)
            painter.drawPixmap(QRectF(0, 0, item.size.width(), item.size.height()), item.pixmap, item.pixmap.rect())
            painter.restore()

        # 7. Render selection borders and primary transform handle gizmos
        if self.selected_items:
            secondary_pen = QPen(QColor(0, 150, 255, 150), 1.0 / self.zoom_scale, Qt.PenStyle.DotLine)
            primary_pen = QPen(QColor(0, 150, 255), 1.5 / self.zoom_scale, Qt.PenStyle.SolidLine)
            handle_pen = QPen(QColor(0, 150, 255), 2.0 / self.zoom_scale)
            handle_brush = QBrush(QColor(255, 255, 255))

            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            primary_selected = self.get_primary_selected_item()

            corner_radius = 8.0 / self.zoom_scale
            handle_radius = max(4.0, 4.0 / self.zoom_scale)
            rotate_handle_radius = max(5.0, 5.0 / self.zoom_scale)
            rotate_line_length = max(15.0, 15.0 / self.zoom_scale)
            icon_radius = rotate_handle_radius * 0.6
            icon_pen = QPen(QColor(0, 150, 255), 1.5 / self.zoom_scale)
            icon_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            arrow_size = icon_radius * 0.7

            for item in self.selected_items:
                is_primary = (item == primary_selected)

                painter.save()
                painter.setTransform(item.get_transform(), True)

                painter.setPen(primary_pen if is_primary else secondary_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                item_rect = QRectF(0, 0, item.size.width(), item.size.height())
                painter.drawRoundedRect(item_rect, corner_radius, corner_radius)

                if is_primary:
                    painter.setPen(handle_pen)
                    painter.setBrush(handle_brush)
                    w, h = item.size.width(), item.size.height()

                    # 4 corner scale handles
                    painter.drawEllipse(QPointF(0, 0), handle_radius, handle_radius)
                    painter.drawEllipse(QPointF(w, 0), handle_radius, handle_radius)
                    painter.drawEllipse(QPointF(0, h), handle_radius, handle_radius)
                    painter.drawEllipse(QPointF(w, h), handle_radius, handle_radius)

                    # Rotate handle connector line
                    rotate_handle_center = QPointF(w / 2, -rotate_line_length)
                    line_pen = QPen(QColor(0, 150, 255), 1.5 / self.zoom_scale)
                    painter.setPen(line_pen)
                    painter.drawLine(QPointF(w / 2, 0), QPointF(w / 2, -rotate_line_length + rotate_handle_radius))

                    # Rotate handle pivot dot
                    painter.setPen(handle_pen)
                    painter.setBrush(handle_brush)
                    painter.drawEllipse(rotate_handle_center, rotate_handle_radius, rotate_handle_radius)

                    # Rotation icon decoration
                    painter.save()
                    painter.translate(rotate_handle_center)
                    painter.setPen(icon_pen)
                    painter.drawArc(QRectF(-icon_radius, -icon_radius, icon_radius * 2, icon_radius * 2), 45 * 16,
                                    270 * 16)

                    arrow_angle_rad = math.radians(45 + 270)
                    painter.translate(
                        QPointF(icon_radius * math.cos(arrow_angle_rad), -icon_radius * math.sin(arrow_angle_rad)))
                    painter.rotate(-45)

                    arrow_path = QPainterPath()
                    arrow_path.moveTo(0, 0)
                    arrow_path.lineTo(-arrow_size, arrow_size / 2)
                    arrow_path.lineTo(-arrow_size, -arrow_size / 2)
                    arrow_path.closeSubpath()

                    painter.setBrush(QColor(0, 150, 255))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPath(arrow_path)
                    painter.restore()

                painter.restore()

        # 8. Restore zoom and translate transforms
        painter.restore()

        # 9. Render minimalist canvas border outline
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        canvas_rect_widget = QRect(
            int(self.canvas_pos.x() * self.zoom_scale + self.view_offset.x()),
            int(self.canvas_pos.y() * self.zoom_scale + self.view_offset.y()),
            int(self.canvas_size.width() * self.zoom_scale),
            int(self.canvas_size.height() * self.zoom_scale)
        )

        minimalist_pen = QPen(QColor(255, 255, 255, 40), 1, Qt.PenStyle.SolidLine)
        painter.setPen(minimalist_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(canvas_rect_widget)

        painter.end()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in [Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right]:
            delta = QPointF(0, 0)
            step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            if event.key() == Qt.Key.Key_Up:
                delta.setY(-step)
            elif event.key() == Qt.Key.Key_Down:
                delta.setY(step)
            elif event.key() == Qt.Key.Key_Left:
                delta.setX(-step)
            elif event.key() == Qt.Key.Key_Right:
                delta.setX(step)
            if self.selected_items:
                for item in self.selected_items: item.pos += delta
                self.selection_changed.emit()
                self.update()
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        super().keyReleaseEvent(event)

    def fit_selection_to_canvas(self):
        if not self.selected_items:
            QMessageBox.information(self, "提示", "请先选择一个或多个图层。")
            return
        group_rect = QRectF()
        for item in self.selected_items:
            group_rect = group_rect.united(item.get_bounding_rect())
        if group_rect.isEmpty(): return

        canvas_w, canvas_h = self.canvas_size.width(), self.canvas_size.height()
        target_w, target_h = canvas_w * 0.95, canvas_h * 0.95
        scale_x = target_w / group_rect.width() if group_rect.width() > 0 else 1
        scale_y = target_h / group_rect.height() if group_rect.height() > 0 else 1
        scale_factor = min(scale_x, scale_y)

        canvas_center, group_center_start = self.canvas_pos + QPointF(canvas_w / 2, canvas_h / 2), group_rect.center()
        for item in self.selected_items:
            item.size *= scale_factor
            vec_from_center = item.pos - group_center_start
            scaled_vec = vec_from_center * scale_factor
            item.pos = canvas_center + scaled_vec
            item.rotation = 0

        self.selection_changed.emit()
        self.update()

    def render_to_qimage(self) -> QImage:
        image = QImage(self.canvas_size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(self.background_color)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        for item in self.items:
            painter.save()
            relative_pos = item.pos - self.canvas_pos

            transform = QTransform()
            transform.translate(relative_pos.x() + item.size.width() / 2, relative_pos.y() + item.size.height() / 2)
            transform.rotate(item.rotation)
            transform.translate(-item.size.width() / 2, -item.size.height() / 2)

            painter.setTransform(transform)
            painter.drawPixmap(QRectF(QPointF(0, 0), item.size), item.pixmap, item.pixmap.rect())
            painter.restore()

        painter.end()
        return image