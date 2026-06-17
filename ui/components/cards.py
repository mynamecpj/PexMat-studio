import os
import math
import cv2

from PySide6.QtWidgets import QLabel, QWidget, QSizePolicy, QGraphicsBlurEffect, QMenu, QToolButton
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QMouseEvent, QMovie, QPaintEvent, QPainterPath, QDrag
from PySide6.QtCore import Qt, QSize, QRect, QRectF, Signal, Property, QTimer, Slot, QUrl, QMimeData, QPoint

from core.utils import convert_cv_to_pixmap

# ===============================================
# Customized: enforced dark mode styling using hardcoded variables
# ===============================================
C_WIDGET_BG = "#262626"
C_INPUT_BG = "#333333"
C_BORDER = "#333333"
C_TEXT_PRIMARY = "#E0E0E0"
C_TEXT_DISABLED = "#555555"
C_PRIMARY = "#1A73E8"
C_DIVIDER_COLOR = "#2A2A2A"
C_LIST_ITEM_SELECTED_BG = "#1F2937"
C_LIST_ITEM_SELECTED_TEXT = "#60A5FA"


class FrostedGlassCard(QWidget):
    """
    Frosted glass card with dark mode tint.
    """

    def __init__(self, tint_opacity=0.6, parent=None):
        super().__init__(parent)
        # Use dark tint for translucent backdrop
        self.tint_color = QColor(20, 20, 20, int(255 * tint_opacity))
        self.border_color = QColor(255, 255, 255, 20)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(self.rect(), 16, 16)
        painter.fillPath(path, self.tint_color)
        painter.setPen(QPen(self.border_color, 1))
        painter.drawPath(path)


class UnfoldingCard(QWidget):
    """
    Animation card transition widget.
    """

    def __init__(self, initial_pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.initial_pixmap = initial_pixmap
        self.target_pixmap = None
        self._radius = 20.0
        self._initial_opacity = 1.0
        self._target_opacity = 0.0
        self._unfold_factor = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    @Property(float)
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value; self.update()

    @Property(float)
    def initialOpacity(self):
        return self._initial_opacity

    @initialOpacity.setter
    def initialOpacity(self, value):
        self._initial_opacity = value; self.update()

    @Property(float)
    def targetOpacity(self):
        return self._target_opacity

    @targetOpacity.setter
    def targetOpacity(self, value):
        self._target_opacity = value; self.update()

    @Property(float)
    def unfoldFactor(self):
        return self._unfold_factor

    @unfoldFactor.setter
    def unfoldFactor(self, value):
        self._unfold_factor = value; self.update()

    def set_target_pixmap(self, pixmap: QPixmap):
        self.target_pixmap = pixmap

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        path = QPainterPath()
        path.addRoundedRect(self.rect(), self._radius, self._radius)
        painter.setClipPath(path)

        if self._initial_opacity > 0.01 and self.initial_pixmap and not self.initial_pixmap.isNull():
            painter.setOpacity(self._initial_opacity)
            painter.drawPixmap(self.rect(), self.initial_pixmap)

        if self.target_pixmap and self._target_opacity > 0.01 and self._unfold_factor > 0.01 and not self.target_pixmap.isNull():
            painter.setOpacity(self._target_opacity)
            source_w = self.target_pixmap.width() * self._unfold_factor
            source_h = self.target_pixmap.height() * self._unfold_factor
            source_x = (self.target_pixmap.width() - source_w) / 2
            source_y = (self.target_pixmap.height() - source_h) / 2
            source_rect = QRectF(source_x, source_y, source_w, source_h)
            destination_rect = QRectF(self.rect())
            painter.drawPixmap(destination_rect, self.target_pixmap, source_rect)


class TransitionCard(QWidget):
    """
    Transition animation card widget.
    """

    def __init__(self, initial_pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.initial_pixmap = initial_pixmap
        self.target_pixmap = None
        self._radius = 20.0
        self._initial_opacity = 1.0
        self._target_opacity = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    @Property(float)
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value; self.update()

    @Property(float)
    def initialOpacity(self):
        return self._initial_opacity

    @initialOpacity.setter
    def initialOpacity(self, value):
        self._initial_opacity = value; self.update()

    @Property(float)
    def targetOpacity(self):
        return self._target_opacity

    @targetOpacity.setter
    def targetOpacity(self, value):
        self._target_opacity = value; self.update()

    def set_target_pixmap(self, pixmap: QPixmap):
        self.target_pixmap = pixmap
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        path = QPainterPath()
        path.addRoundedRect(self.rect(), self._radius, self._radius)
        painter.setClipPath(path)

        if self._initial_opacity > 0.01 and self.initial_pixmap and not self.initial_pixmap.isNull():
            painter.setOpacity(self._initial_opacity)
            painter.drawPixmap(self.rect(), self.initial_pixmap)

        if self.target_pixmap and self._target_opacity > 0.01 and not self.target_pixmap.isNull():
            painter.setOpacity(self._target_opacity)
            painter.drawPixmap(self.rect(), self.target_pixmap)


class AnimatedPixmapCard(QWidget):
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self._radius = 0.0
        self._opacity = 1.0
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    @Property(float)
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value; self.update()

    @Property(float)
    def pixmapOpacity(self):
        return self._opacity

    @pixmapOpacity.setter
    def pixmapOpacity(self, value):
        self._opacity = value; self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        current_opacity = self.pixmapOpacity
        if current_opacity < 0.01:
            return

        painter.setOpacity(current_opacity)
        path = QPainterPath()
        path.addRoundedRect(self.rect(), self.radius, self.radius)
        painter.setClipPath(path)

        if self.pixmap and not self.pixmap.isNull():
            painter.drawPixmap(self.rect(), self.pixmap)


class FadingBlurredCard(QWidget):
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.blur_effect = QGraphicsBlurEffect(self)
        self.setGraphicsEffect(self.blur_effect)
        self.blur_effect.setBlurRadius(0)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    @Property(float)
    def blurRadius(self):
        return self.blur_effect.blurRadius()

    @blurRadius.setter
    def blurRadius(self, value):
        if not math.isclose(self.blur_effect.blurRadius(), value):
            self.blur_effect.setBlurRadius(value)
            self.update()

    @Slot(QPixmap)
    def setPixmap(self, pixmap: QPixmap):
        self.pixmap = pixmap
        if self.isVisible():
            self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self.pixmap and not self.pixmap.isNull():
            widget_rect = self.rect()
            scaled_pixmap = self.pixmap.scaled(
                widget_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (widget_rect.width() - scaled_pixmap.width()) / 2
            y = (widget_rect.height() - scaled_pixmap.height()) / 2
            target_rect = QRect(int(x), int(y), scaled_pixmap.width(), scaled_pixmap.height())
            painter.drawPixmap(target_rect, scaled_pixmap)


class WelcomeImageCard(QWidget):
    """
    Media card control for welcome page. Updated fallback background to dark mode style.
    """

    def __init__(self, asset_path: str, parent=None):
        super().__init__(parent)
        self.asset_path = asset_path
        self.media_type = "none"
        self.is_load_successful = False

        self.pixmap = None
        self.movie = None

        self.video_capture = None
        self.video_timer = QTimer(self)
        self.video_frame_count = 0
        self.current_video_frame_index = 0
        self.current_frame_pixmap = QPixmap()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.video_timer.timeout.connect(self._advance_video_frame)

        self._load_asset()

    def _load_asset(self):
        if not os.path.exists(self.asset_path):
            print(f"警告: WelcomeImageCard 资源文件不存在: {self.asset_path}")
            return

        ext = os.path.splitext(self.asset_path)[1].lower()
        if ext == '.gif':
            self.media_type = "gif"
            self.movie = QMovie(self.asset_path)
            if self.movie.isValid():
                self.movie.frameChanged.connect(self.update)
                self.movie.jumpToFrame(0)
                self.is_load_successful = True

        elif ext in ['.mp4', '.mov', '.avi', '.mkv']:
            self.media_type = "video"
            try:
                self.video_capture = cv2.VideoCapture(self.asset_path)
                if self.video_capture.isOpened():
                    self.video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                    self.video_timer.setInterval(int(1000 / fps) if fps > 0 else 40)
                    ret, frame = self.video_capture.read()
                    if ret:
                        self.current_frame_pixmap = convert_cv_to_pixmap(frame)
                        self.is_load_successful = True
                    else:
                        self.video_capture.release()
                else:
                    print(f"错误:无法使用OpenCV打开视频 {self.asset_path}")
            except Exception as e:
                print(f"处理视频时发生错误: {e}")
                if self.video_capture: self.video_capture.release()

        else:
            self.media_type = "image"
            self.pixmap = QPixmap(self.asset_path)
            self.is_load_successful = not self.pixmap.isNull()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)

        # [Key 1] Explicitly disable outline pen to avoid any system-default 1px borders
        painter.setPen(Qt.PenStyle.NoPen)

        if self.is_load_successful:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            path = QPainterPath()
            # [Key 2] Contract the rectangle by 0.5 pixels to resolve edge aliasing issues
            path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 20, 20)
            painter.setClipPath(path)

            pixmap_to_draw = None
            if self.media_type == "image":
                pixmap_to_draw = self.pixmap
            elif self.media_type == "gif" and self.movie:
                pixmap_to_draw = self.movie.currentPixmap()
            elif self.media_type == "video":
                pixmap_to_draw = self.current_frame_pixmap

            if pixmap_to_draw and not pixmap_to_draw.isNull():
                scaled_pixmap = pixmap_to_draw.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation
                )
                x = (self.width() - scaled_pixmap.width()) / 2
                y = (self.height() - scaled_pixmap.height()) / 2
                # Render final frame
                painter.drawPixmap(int(x), int(y), scaled_pixmap)
            else:
                # Fill with clean solid dark color when no media is loaded
                painter.fillPath(path, QColor(C_WIDGET_BG))
        else:
            path = QPainterPath()
            path.addRoundedRect(self.rect(), 20, 20)
            painter.fillPath(path, QColor(C_WIDGET_BG))
            painter.setPen(QColor(C_TEXT_DISABLED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "资源加载失败")

    @Slot()
    def _advance_video_frame(self):
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame_pixmap = convert_cv_to_pixmap(frame)
                self.current_video_frame_index += 1
                self.update()
            else:
                self.current_video_frame_index = 0
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_capture.read()
                if ret:
                    self.current_frame_pixmap = convert_cv_to_pixmap(frame)
                    self.update()
                else:
                    self.video_timer.stop()

    def start_media(self):
        if self.media_type == "gif" and self.movie:
            self.movie.start()
        elif self.media_type == "video" and self.video_capture:
            self.video_timer.start()

    def stop_media(self):
        if self.media_type == "gif" and self.movie:
            self.movie.stop()
            self.movie.jumpToFrame(0)
            self.update()
        elif self.media_type == "video" and self.video_capture:
            self.video_timer.stop()
            self.current_video_frame_index = 0
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame_pixmap = convert_cv_to_pixmap(frame)
                self.update()

    def __del__(self):
        if self.video_capture:
            self.video_capture.release()


class RoundedShadowCard(QWidget):
    """
    Rounded card with shadow effect styled for dark mode.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.background_color = QColor("#262626")  # Dark gray base
        self.radius = 20.0

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.background_color)
        path = QPainterPath()
        path.addRoundedRect(self.rect(), self.radius, self.radius)
        painter.drawPath(path)


# ==============================================================
# AssetThumbnail class customized to 108x108 with hover delete option
# ==============================================================
class AssetThumbnail(QWidget):
    """Modern large asset thumbnail component (inheriting from QWidget)."""
    delete_requested = Signal(str)

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path

        # Hardcode geometry size to exactly 108x108
        self.setFixedSize(108, 108)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # --- Load and parse image ---
        self.pixmap = QPixmap(image_path)
        if not self.pixmap.isNull():
            # Smooth scale to fit 100x100 inside layout margins
            self.pixmap = self.pixmap.scaled(
                100, 100,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        # --- Hover delete button configuration ---
        self.delete_btn = QToolButton(self)
        self.delete_btn.setText("×")
        self.delete_btn.setFixedSize(20, 20)
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setStyleSheet("""
            QToolButton {
                background-color: rgba(0, 0, 0, 0.6);
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }
            QToolButton:hover {
                background-color: #EF4444; /* Hover color */
            }
        """)
        # Anchor to the top-right corner
        self.delete_btn.move(84, 4)
        self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(self.image_path))
        self.delete_btn.hide()  # Hidden by default, displayed only on hover

        self.is_hovered = False
        self.drag_start_pos = None

    def enterEvent(self, event):
        """Mouse enter event: show delete button and trigger border highlight repaint."""
        self.is_hovered = True
        self.delete_btn.show()
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Mouse leave event: hide delete button and restore default appearance."""
        self.is_hovered = False
        self.delete_btn.hide()
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        """Custom painting routine to ensure clean rounded corners and high DPI rendering."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 1. Draw base background and rounded bounding path
        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, 8, 8)

        # Slightly brighten background color on hover to match design specs
        painter.fillPath(path, QColor(C_INPUT_BG) if self.is_hovered else QColor(C_WIDGET_BG))

        # 2. Centered pixmap drawing
        if hasattr(self, 'pixmap') and not self.pixmap.isNull():
            x = (self.width() - self.pixmap.width()) // 2
            y = (self.height() - self.pixmap.height()) // 2
            painter.setClipPath(path)  # Ensure drawing remains bound to rounded clipping path

            # Paint dark placeholder background if image has alpha transparency
            if self.pixmap.hasAlphaChannel():
                painter.fillRect(self.rect(), QColor(C_INPUT_BG))

            painter.drawPixmap(x, y, self.pixmap)
            painter.setClipping(False)
        else:
            painter.setPen(QColor(C_TEXT_DISABLED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "加载\n失败")

        # 3. Draw border styling
        if self.is_hovered:
            painter.setPen(QPen(QColor(C_PRIMARY), 1.5))  # Render hover state border highlights
            painter.drawPath(path)
        else:
            painter.setPen(QPen(QColor(C_BORDER), 1))  # Render normal state border
            painter.drawPath(path)

    # --- Drag-and-drop initialization ---
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = event.pos()
            # [Core fix] Accept event without calling super() or changing cursor immediately to prevent dragging lockouts.
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not (event.buttons() & Qt.MouseButton.LeftButton) or not self.drag_start_pos:
            return
        if (event.pos() - self.drag_start_pos).manhattanLength() < 5:
            return

        self.drag_start_pos = None

        drag = QDrag(self)
        mime_data = QMimeData()
        urls = [QUrl.fromLocalFile(self.image_path)]
        mime_data.setUrls(urls)

        # [Zero-latency fix] Include preloaded pixmap in the drag payload directly.
        if hasattr(self, 'pixmap') and not self.pixmap.isNull():
            mime_data.setImageData(self.pixmap.toImage())

        drag.setMimeData(mime_data)

        if hasattr(self, 'pixmap') and not self.pixmap.isNull():
            target_size = 64
            drag_pixmap_scaled = self.pixmap.scaled(
                target_size, target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            shadow_offset = 3
            drag_preview_pixmap = QPixmap(drag_pixmap_scaled.width() + shadow_offset,
                                          drag_pixmap_scaled.height() + shadow_offset)
            drag_preview_pixmap.fill(Qt.transparent)

            preview_painter = QPainter(drag_preview_pixmap)
            preview_painter.setOpacity(0.3)
            preview_painter.drawPixmap(shadow_offset, shadow_offset, drag_pixmap_scaled)
            preview_painter.setOpacity(1.0)
            preview_painter.drawPixmap(0, 0, drag_pixmap_scaled)
            preview_painter.end()

            drag.setPixmap(drag_preview_pixmap)

            scale_x = drag_pixmap_scaled.width() / self.pixmap.width() if self.pixmap.width() > 0 else 1
            scale_y = drag_pixmap_scaled.height() / self.pixmap.height() if self.pixmap.height() > 0 else 1

            img_x = (self.width() - self.pixmap.width()) // 2
            img_y = (self.height() - self.pixmap.height()) // 2

            mouse_in_img_x = event.pos().x() - img_x
            mouse_in_img_y = event.pos().y() - img_y

            hot_x = int(mouse_in_img_x * scale_x)
            hot_y = int(mouse_in_img_y * scale_y)

            hot_x = max(0, min(hot_x, drag_pixmap_scaled.width()))
            hot_y = max(0, min(hot_y, drag_pixmap_scaled.height()))

            drag.setHotSpot(QPoint(hot_x, hot_y))

        drag.exec(Qt.DropAction.MoveAction)


class AspectRatioPixmapWidget(QWidget):
    def __init__(self, pixmap: QPixmap = None, parent=None):
        super().__init__(parent)
        self._pixmap = pixmap
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.update()

    def pixmap(self) -> QPixmap | None:
        return self._pixmap

    def paintEvent(self, event: QPaintEvent):
        if not self._pixmap or self._pixmap.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        widget_rect = self.rect()
        pixmap_size = self._pixmap.size()

        if widget_rect.isEmpty() or pixmap_size.isEmpty():
            return

        scaled_pixmap = self._pixmap.scaled(widget_rect.size(),
                                            Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation)

        x = (widget_rect.width() - scaled_pixmap.width()) / 2
        y = (widget_rect.height() - scaled_pixmap.height()) / 2

        target_draw_rect = QRect(int(x), int(y), scaled_pixmap.width(), scaled_pixmap.height())
        painter.drawPixmap(target_draw_rect, scaled_pixmap)