from PySide6.QtWidgets import  QLabel,QHBoxLayout,QWidget, QSizePolicy
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QConicalGradient,QMouseEvent, QBrush,QPaintEvent,QFontMetrics,QPainterPath,QRadialGradient,QLinearGradient
from PySide6.QtCore import Qt, QPoint, QPointF, QRect, QRectF,QEvent,Signal, QPropertyAnimation, QEasingCurve, Property
import math
import os
import cv2
from core.utils import convert_cv_to_pixmap
from config.settings import C_PRIMARY,SUPPORTED_VIDEO_FORMATS

class HoverableLabel(QLabel):
    hover_enter = Signal(QRect)
    hover_leave = Signal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMouseTracking(True)
        self._is_hovering = False

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._is_hovering:
            self._is_hovering = True
            self.hover_enter.emit(self.rect())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent):
        if self._is_hovering:
            self._is_hovering = False
            self.hover_leave.emit()
        super().leaveEvent(event)

class GradientTitleLabel(QLabel):
    hover_enter_creative = Signal(QRect)
    hover_leave_creative = Signal()

    def __init__(self, text_parts: list[tuple[str, QColor | QLinearGradient]], parent=None):
        full_text = "".join([part[0] for part in text_parts])
        super().__init__(full_text, parent)
        self.text_parts = text_parts
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._animation_offset = 0.0
        self.gradient_animation = QPropertyAnimation(self, b"animationOffset")
        self.gradient_animation.setDuration(2500)
        self.gradient_animation.setStartValue(0.0)
        self.gradient_animation.setEndValue(2.0)
        self.gradient_animation.setLoopCount(-1)
        self.gradient_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.gradient_animation.start()

        self.setMouseTracking(True)
        self.creative_word_rect = QRect()
        self._is_hovering_creative = False

    @Property(float)
    def animationOffset(self):
        return self._animation_offset

    @animationOffset.setter
    def animationOffset(self, value):
        self._animation_offset = value
        self.update()

    def set_text_parts(self, new_text_parts):
        """支持动态更新带颜色的分段文字"""
        self.text_parts = new_text_parts
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        font = self.font()
        painter.setFont(font)
        fm = QFontMetrics(font)

        text_height = fm.boundingRect("".join([p[0] for p in self.text_parts])).height()
        y_pos = (self.height() - text_height) / 2 + fm.ascent()

        current_x = 0
        for text, style in self.text_parts:
            text_width = fm.horizontalAdvance(text)

            # 这里可以改用动态匹配：只要是在中文的"创意"或者英文的"Creativity"处悬停就触发
            if "创意" in text or "Creativity" in text:
                self.creative_word_rect = QRect(int(current_x), 0, int(text_width), self.height())

            if isinstance(style, QColor):
                if style.name() == "#1d1d1f" or style.name() == "#000000":
                    painter.setPen(QColor("#E0E0E0"))
                else:
                    painter.setPen(style)
            elif isinstance(style, QLinearGradient):
                offset_factor = math.sin(self._animation_offset * math.pi)
                color1 = QColor("#A052E8")
                color2 = QColor("#D88CFF")
                color3 = QColor("#6B25B3")
                dynamic_gradient = QLinearGradient(current_x, 0, current_x + text_width, 0)
                highlight_pos = (offset_factor + 1) / 2.0
                dynamic_gradient.setColorAt(0, color1)
                dynamic_gradient.setColorAt(max(0, highlight_pos - 0.2), color1)
                dynamic_gradient.setColorAt(highlight_pos, color2)
                dynamic_gradient.setColorAt(min(1, highlight_pos + 0.2), color3)
                dynamic_gradient.setColorAt(1, color3)
                pen = QPen()
                pen.setBrush(dynamic_gradient)
                painter.setPen(pen)

            painter.drawText(QPoint(int(current_x), int(y_pos)), text)
            current_x += text_width

    def mouseMoveEvent(self, event: QMouseEvent):
        is_inside = self.creative_word_rect.contains(event.position().toPoint())
        if is_inside and not self._is_hovering_creative:
            self._is_hovering_creative = True
            self.hover_enter_creative.emit(self.creative_word_rect)
        elif not is_inside and self._is_hovering_creative:
            self._is_hovering_creative = False
            self.hover_leave_creative.emit()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QEvent):
        if self._is_hovering_creative:
            self._is_hovering_creative = False
            self.hover_leave_creative.emit()
        super().leaveEvent(event)

class ElidedLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._original_text = text

    def setText(self, text):
        self._original_text = text
        self.update()

    def text(self):
        return self._original_text

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        fm = painter.fontMetrics()
        available_width = self.width()
        elided_text = fm.elidedText(self._original_text, Qt.TextElideMode.ElideMiddle, available_width)
        painter.drawText(self.rect(), self.alignment(), elided_text)

class ToggleSwitch(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False
        self.setFixedSize(48, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("开启或关闭该功能")

        self._circle_position = 3
        self.animation = QPropertyAnimation(self, b"circle_position_prop", self)
        self.animation.setDuration(150)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        if self._checked != checked:
            self._checked = checked
            self.animation.setStartValue(self._circle_position)
            self.animation.setEndValue(self.width() - self.height() + 3 if checked else 3)
            self.animation.start()
            self.toggled.emit(self._checked)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self.isChecked())
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 暗色主题开关底色
        track_color = QColor(C_PRIMARY) if self._checked else QColor("#333333")
        handle_color = QColor("#E0E0E0")

        track_rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track_rect, self.height() / 2, self.height() / 2)

        handle_radius = (self.height() / 2) - 2
        handle_y = self.height() / 2
        handle_x = self._circle_position + handle_radius

        painter.setBrush(handle_color)
        shadow_offset = 1
        painter.setPen(QColor(0,0,0,50)) # 轻微的阴影
        painter.drawEllipse(QPointF(handle_x + shadow_offset, handle_y + shadow_offset), handle_radius, handle_radius)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(handle_color)
        painter.drawEllipse(QPointF(handle_x, handle_y), handle_radius, handle_radius)

    @Property(int)
    def circle_position_prop(self):
        return self._circle_position

    @circle_position_prop.setter
    def circle_position_prop(self, position):
        self._circle_position = position
        self.update()

class DynamicGradientBackground(QWidget):
    """【黑化渐变背景】：提供非常深沉、低调的高级灰黑渐变"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.mouse_pos = QPointF(self.width() / 2, self.height() / 2)

        # 使用纯黑到极深灰的渐变
        self.color1 = QColor(18, 18, 18)
        self.color2 = QColor(24, 24, 24)
        self.color3 = QColor(20, 20, 22)
        self.color4 = QColor(15, 15, 15)

    def mouseMoveEvent(self, event: QMouseEvent):
        self.mouse_pos = event.position()
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        gradient = QConicalGradient(self.mouse_pos, 0)
        gradient.setColorAt(0.0, self.color1)
        gradient.setColorAt(0.25, self.color2)
        gradient.setColorAt(0.5, self.color3)
        gradient.setColorAt(0.75, self.color4)
        gradient.setColorAt(1.0, self.color1)

        painter.fillRect(self.rect(), QBrush(gradient))

class HighlightOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._gradient_center = QPointF(-1, -1)
        self._gradient_radius = 0
        self._opacity_val = 0.0

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    @Property(float)
    def highlightOpacity(self):
        return self._opacity_val

    @highlightOpacity.setter
    def highlightOpacity(self, value):
        self._opacity_val = value
        self.update()

    def set_highlight_geometry(self, center: QPointF, radius: float):
        self._gradient_center = center
        self._gradient_radius = radius
        if self.isVisible():
            self.update()

    def paintEvent(self, event: QPaintEvent):
        if self._opacity_val > 0.01 and self._gradient_radius > 1:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # 在暗黑主题下保留白色高光
            gradient = QRadialGradient(self._gradient_center, self._gradient_radius)
            gradient.setColorAt(0, QColor(255, 255, 255, int(150 * self._opacity_val)))
            gradient.setColorAt(0.4, QColor(255, 255, 255, int(80 * self._opacity_val)))
            gradient.setColorAt(1, QColor(255, 255, 255, 0))

            painter.fillRect(self.rect(), QBrush(gradient))

class RecentProjectItemWidget(QWidget):
    """用于“最近打开”列表的自定义项 (暗黑风格适配)"""

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setToolTip(file_path)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(64, 36)
        # 暗黑背景底色
        self.thumbnail_label.setStyleSheet("background-color: #2A2A2A; border-radius: 6px;")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ext = os.path.splitext(file_path)[1].lower()
        pixmap = None
        if ext in SUPPORTED_VIDEO_FORMATS:
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        pixmap = convert_cv_to_pixmap(frame)
                    cap.release()
            except Exception as e:
                print(f"为最近项目列表加载视频帧时出错: {e}")
        else:
            pixmap = QPixmap(file_path)

        if pixmap and not pixmap.isNull():
            final_pixmap = QPixmap(self.thumbnail_label.size())
            final_pixmap.fill(Qt.transparent)
            painter = QPainter(final_pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            path = QPainterPath()
            path.addRoundedRect(QRectF(final_pixmap.rect()), 6.0, 6.0)
            painter.setClipPath(path)
            source_scaled = pixmap.scaled(self.thumbnail_label.size(),
                                          Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                          Qt.TransformationMode.SmoothTransformation)
            target_rect = final_pixmap.rect()
            source_rect = source_scaled.rect()
            x = (target_rect.width() - source_rect.width()) / 2
            y = (target_rect.height() - source_rect.height()) / 2
            painter.drawPixmap(int(x), int(y), source_scaled)
            painter.end()
            self.thumbnail_label.setPixmap(final_pixmap)
        else:
            icon_char = "🎬" if ext in SUPPORTED_VIDEO_FORMATS else "🖼️"
            self.thumbnail_label.setText(icon_char)
            self.thumbnail_label.setStyleSheet("font-size: 16pt; color: #888; background-color: #2A2A2A; border-radius: 6px;")

        filename_label = QLabel(os.path.basename(file_path))
        filename_label.setStyleSheet("font-size: 10pt; font-weight: bold; color: #E0E0E0; background: transparent;")

        main_layout.addWidget(self.thumbnail_label)
        main_layout.addWidget(filename_label)
        main_layout.setAlignment(filename_label, Qt.AlignmentFlag.AlignVCenter)