# Welcome Page
import sys
import os
from typing import TYPE_CHECKING

import PySide6.QtWidgets as qw
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton,
                               QAbstractItemView, QListWidget, QGraphicsPixmapItem,
                               QGraphicsScene, QMenu, QGraphicsDropShadowEffect,
                               QGraphicsBlurEffect, QApplication, QToolButton)
from PySide6.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QConicalGradient,
                           QMouseEvent, QResizeEvent, QBrush, QPaintEvent, QPainterPath,
                           QShowEvent, QMovie, QLinearGradient)
from PySide6.QtCore import (Qt, QPoint, QRect, QSize, QEvent, Slot,
                                QPropertyAnimation, QEasingCurve, Property,
                                QParallelAnimationGroup, QAbstractAnimation, QTimer)

from ui.components.basic_widgets import GradientTitleLabel, HoverableLabel
from config.settings import (C_WIDGET_BG, C_LIST_ITEM_SELECTED_BG, C_LIST_ITEM_SELECTED_TEXT,
                                 C_DIVIDER_COLOR, _TR, get_app_lang, set_app_lang)
from core.utils import get_asset_path

if TYPE_CHECKING:
    from ui.main_window import ImageEnhancerApp

# ==========================================
# Modern borderless button style (dark mode adapted)
# ==========================================
MODERN_TOP_BAR_BUTTON_STYLE = """
    QToolButton {
        border: none;
        background-color: transparent;
        border-radius: 12px;
        padding: 8px 6px;
        min-width: 72px; 
        color: #E0E0E0; /* Text color under dark mode */
        font-weight: 500;
        font-size: 11px; 
    }
    QToolButton:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    QToolButton:pressed {
        background-color: rgba(255, 255, 255, 0.15);
    }
    QToolButton:checked {
        background-color: rgba(255, 255, 255, 0.12);
        font-weight: bold;
    }
"""


class WelcomePageWithCards(QWidget):
    def __init__(self, main_window: 'ImageEnhancerApp', parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

        self._gradient_angle = 0.0
        self.bg_animation = QPropertyAnimation(self, b"gradientAngle")
        self.bg_animation.setDuration(30000)  # Maintain gentle background flow speed
        self.bg_animation.setStartValue(0)
        self.bg_animation.setEndValue(360)
        self.bg_animation.setLoopCount(-1)
        self.bg_animation.start()

        # Dynamic gradient background palette for home page (dark gray/pure black)
        self.color1 = QColor("#181818")  # Base background color (matching the main window)
        self.color2 = QColor("#1D1D1D")  # Slight highlight
        self.color3 = QColor("#141414")  # Slight shadow
        self.color4 = QColor("#1A1A1A")  # Soft transitional shade

        # Translucent background tint for the right panel
        self.tint_color = QColor(38, 38, 38, int(255 * 0.45))
        self.border_color = QColor(255, 255, 255, 12)  # Fine edge highlight line

        self._blurred_bg_cache = QPixmap()
        self._cache_valid = False

        self._setup_foreground_widgets()

    def _setup_foreground_widgets(self):
        self.left_panel_widget = QWidget(self)
        left_panel_layout = QVBoxLayout(self.left_panel_widget)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(20)

        creative_gradient = QLinearGradient(0, 0, 250, 0)
        creative_gradient.setColorAt(0.0, QColor("#818CF8"))
        creative_gradient.setColorAt(0.5, QColor("#C084FC"))
        creative_gradient.setColorAt(1.0, QColor("#E879F9"))

        # Main title
        title_text_parts = [("创意", creative_gradient), (" 从这里开始", QColor("#E0E0E0"))]
        self.title_label = GradientTitleLabel(title_text_parts)
        self.title_label.setStyleSheet("background-color: transparent; font-size: 58pt; font-weight: 800;")
        self.title_label.setParent(self.left_panel_widget)

        # Subtitle with specific layout metadata
        subtitle_text = f"版本 {self.main_window.__version__} - 创意无限"
        self.subtitle_label = HoverableLabel(subtitle_text, self.left_panel_widget)
        self.subtitle_label.setStyleSheet(
            "font-size: 13pt; color: #888888; padding-left: 4px; background: transparent;")

        quick_actions_container = QWidget(self.left_panel_widget)
        self.quick_actions_layout = QHBoxLayout(quick_actions_container)
        self.quick_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.quick_actions_layout.setSpacing(15)

        # Quick access action buttons
        self.open_file_button = QToolButton()
        self.open_file_button.setText("打开文件")
        self.open_file_button.setIcon(
            self.main_window._create_svg_icon("folder2.svg", size=32, color=QColor("#E0E0E0")))
        self.open_file_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.open_file_button.setIconSize(QSize(32, 32))
        self.open_file_button.setStyleSheet(MODERN_TOP_BAR_BUTTON_STYLE)
        self.open_file_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_file_button.clicked.connect(lambda: self.main_window._handle_quick_action("open_file"))

        self.quick_enhance_button = QToolButton()
        self.quick_enhance_button.setText("快速高清")
        self.quick_enhance_button.setIcon(
            self.main_window._create_svg_icon("stars.svg", size=32, color=QColor("#E0E0E0")))
        self.quick_enhance_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.quick_enhance_button.setIconSize(QSize(32, 32))
        self.quick_enhance_button.setStyleSheet(MODERN_TOP_BAR_BUTTON_STYLE)
        self.quick_enhance_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.quick_enhance_button.clicked.connect(self.main_window.start_quick_enhance_flow)

        self.quick_seg_button = QToolButton()
        self.quick_seg_button.setText("快速抠图")
        self.quick_seg_button.setIcon(
            self.main_window._create_svg_icon("person-bounding-box.svg", size=32, color=QColor("#E0E0E0")))
        self.quick_seg_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.quick_seg_button.setIconSize(QSize(32, 32))
        self.quick_seg_button.setStyleSheet(MODERN_TOP_BAR_BUTTON_STYLE)
        self.quick_seg_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.quick_seg_button.clicked.connect(self.main_window.start_quick_segment_flow)

        self.from_clipboard_button = QToolButton()
        self.from_clipboard_button.setText("剪贴板创建")
        self.from_clipboard_button.setIcon(
            self.main_window._create_svg_icon("clipboard.svg", size=32, color=QColor("#E0E0E0")))
        self.from_clipboard_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.from_clipboard_button.setIconSize(QSize(32, 32))
        self.from_clipboard_button.setStyleSheet(MODERN_TOP_BAR_BUTTON_STYLE)
        self.from_clipboard_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.from_clipboard_button.clicked.connect(lambda: self.main_window._handle_quick_action("from_clipboard"))

        self.batch_seg_button = QToolButton()
        self.batch_seg_button.setText("批量抠图")
        self.batch_seg_button.setIcon(self.main_window._create_svg_icon("images.svg", size=32, color=QColor("#E0E0E0")))
        self.batch_seg_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.batch_seg_button.setIconSize(QSize(32, 32))
        self.batch_seg_button.setStyleSheet(MODERN_TOP_BAR_BUTTON_STYLE)
        self.batch_seg_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.batch_seg_button.clicked.connect(
            lambda: self.main_window.switch_page(getattr(self.main_window, 'BATCH_MATTING_INDEX', 3))
        )

        self.quick_actions_layout.addWidget(self.open_file_button)
        self.quick_actions_layout.addWidget(self.quick_enhance_button)
        self.quick_actions_layout.addWidget(self.quick_seg_button)
        self.quick_actions_layout.addWidget(self.batch_seg_button)
        self.quick_actions_layout.addWidget(self.from_clipboard_button)
        self.quick_actions_layout.addStretch(1)

        left_panel_layout.addWidget(self.title_label)
        left_panel_layout.addWidget(self.subtitle_label)
        left_panel_layout.addWidget(quick_actions_container)

        self.info_widget = QWidget(self)
        info_layout = QHBoxLayout(self.info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)

        info_icon = QLabel(self.info_widget)
        info_pixmap = self.main_window._create_svg_icon("lightbulb.svg", size=18, color=QColor("#888888")).pixmap(18,
                                                                                                                  18)
        info_icon.setPixmap(info_pixmap)
        info_icon.setStyleSheet("background: transparent;")

        # Shortcut tip message
        info_text = QLabel("提示: 按 H 键可从任何页面快速返回主页啦", self.info_widget)
        info_text.setStyleSheet("font-size: 9.5pt; color: #888888; background: transparent;")

        info_layout.addWidget(info_icon)
        info_layout.addWidget(info_text)
        info_layout.addStretch(1)

        self.right_panel_widget = QWidget(self)
        self.right_panel_widget.setObjectName("WelcomeRightPanelContainer")
        right_panel_layout = QVBoxLayout(self.right_panel_widget)
        right_panel_layout.setContentsMargins(25, 25, 25, 25)
        right_panel_layout.setSpacing(10)

        recent_label = QLabel("最近打开", self.right_panel_widget)
        recent_label.setObjectName("RecentLabel")
        recent_label.setStyleSheet(
            "font-size: 14pt; font-weight: bold; color: #E0E0E0; border: none; background: transparent;")

        self.recent_projects_list = QListWidget(self.right_panel_widget)
        self.recent_projects_list.setObjectName("RecentProjectsList")
        self.recent_projects_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; padding-right: 5px; color:#E0E0E0; }
            QListWidget::item { padding: 0px; margin: 2px 0; border: none; background-color: transparent; }
            QListWidget::item:hover {} 
            QListWidget::item:selected { background-color: rgba(255, 255, 255, 0.1); border-radius: 8px; }
            QListWidget::verticalScrollBar { width: 0px; }
        """)
        self.recent_projects_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.recent_projects_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent_projects_list.customContextMenuRequested.connect(self.show_recent_item_context_menu)
        self.recent_projects_list.itemDoubleClicked.connect(self.main_window._load_recent_project)
        right_panel_layout.addWidget(recent_label)
        right_panel_layout.addWidget(self.recent_projects_list, 1)

        cards_data = [("视频抠图", "video_montage.mp4", self.main_window.VIDEO_SEG_PAGE_INDEX),
                      ("创意工坊", "collage.MP4", self.main_window.CREATIVE_WORKSHOP_INDEX)]
        self.card1_container, card1_core = self.main_window._create_welcome_card_final(cards_data[0][0],
                                                                                       cards_data[0][1],
                                                                                       cards_data[0][2])
        self.card1_container.setParent(self)
        self.card1_container.setProperty("core_content_widget", card1_core)

        self.card2_container, card2_core = self.main_window._create_welcome_card_final(cards_data[1][0],
                                                                                       cards_data[1][1],
                                                                                       cards_data[1][2])
        self.card2_container.setParent(self)
        self.card2_container.setProperty("core_content_widget", card2_core)

    @Property(float)
    def gradientAngle(self):
        return self._gradient_angle

    @gradientAngle.setter
    def gradientAngle(self, value):
        self._gradient_angle = value
        self._cache_valid = False
        self.update()

    @Slot(QPoint)
    def show_recent_item_context_menu(self, pos: QPoint):
        item = self.recent_projects_list.itemAt(pos)
        if not item or not item.data(Qt.ItemDataRole.UserRole): return
        menu = QMenu(self)

        delete_action = menu.addAction("删除此记录")
        delete_action.setIcon(self.main_window._create_svg_icon("trash.svg", color=QColor("#E0E0E0")))

        menu.setStyleSheet(f"""
            QMenu {{ background-color: #262626; border: 1px solid #333333; border-radius: 8px; padding: 5px; }} 
            QMenu::item {{ padding: 8px 20px; border-radius: 6px; color: #E0E0E0; }} 
            QMenu::item:selected {{ background-color: #333333; color: #FFFFFF; }}
        """)
        action = menu.exec(self.recent_projects_list.mapToGlobal(pos))
        if action == delete_action: self.main_window.remove_from_recent_projects(item.data(Qt.ItemDataRole.UserRole))

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QConicalGradient(self.rect().center(), self._gradient_angle)
        gradient.setColorAt(0.0, self.color1)
        gradient.setColorAt(0.25, self.color2)
        gradient.setColorAt(0.5, self.color3)
        gradient.setColorAt(0.75, self.color4)
        gradient.setColorAt(1.0, self.color1)
        painter.fillRect(self.rect(), QBrush(gradient))

        if hasattr(self, 'right_panel_widget') and self.right_panel_widget.isVisible():
            panel_rect = self.right_panel_widget.geometry()
            if not self._cache_valid:
                full_bg_pixmap = QPixmap(self.size())
                full_bg_pixmap.fill(Qt.transparent)
                p = QPainter(full_bg_pixmap)
                p.fillRect(self.rect(), QBrush(gradient))
                p.end()
                sub_pixmap = full_bg_pixmap.copy(panel_rect)
                scene = QGraphicsScene()
                item = QGraphicsPixmapItem(sub_pixmap)
                blur_effect = QGraphicsBlurEffect()
                blur_effect.setBlurRadius(80)
                item.setGraphicsEffect(blur_effect)
                scene.addItem(item)
                image = QImage(sub_pixmap.size(), QImage.Format.Format_ARGB32_Premultiplied)
                image.fill(Qt.transparent)
                temp_painter = QPainter(image)
                scene.render(temp_painter)
                temp_painter.end()
                self._blurred_bg_cache = QPixmap.fromImage(image)
                self._cache_valid = True

            if not self._blurred_bg_cache.isNull():
                painter.drawPixmap(panel_rect.topLeft(), self._blurred_bg_cache)

            path = QPainterPath()
            path.addRoundedRect(panel_rect, 16, 16)
            painter.fillPath(path, self.tint_color)
            painter.setPen(QPen(self.border_color, 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self._cache_valid = False

        margin_h, margin_v = 40, 40
        spacing_h = 40
        right_panel_width = 320
        bottom_bar_height = 30
        w, h = self.width(), self.height()

        right_panel_x = w - margin_h - right_panel_width
        self.right_panel_widget.setGeometry(right_panel_x, margin_v, right_panel_width, h - 2 * margin_v)

        main_area_width = right_panel_x - margin_h - spacing_h

        info_y = h - margin_v - bottom_bar_height
        if hasattr(self, 'info_widget'):
            self.info_widget.setGeometry(margin_h, info_y, int(main_area_width), bottom_bar_height)

        top_area_height = self.left_panel_widget.sizeHint().height()
        self.left_panel_widget.setGeometry(margin_h, margin_v, int(main_area_width), top_area_height)

        if hasattr(self.main_window, 'settings_switches_container'):
            toggle_widget = self.main_window.settings_switches_container
            toggle_size = toggle_widget.sizeHint()
            right_panel_geom = self.right_panel_widget.geometry()
            toggle_x = right_panel_geom.left() - toggle_size.width() - 10
            panel_internal_top_margin = 25
            toggle_y = right_panel_geom.top() + panel_internal_top_margin
            toggle_widget.setGeometry(int(toggle_x), int(toggle_y), toggle_size.width(), toggle_size.height())

        cards_area_y_start = self.left_panel_widget.geometry().bottom() + 20
        cards_area_height = self.info_widget.geometry().top() - cards_area_y_start - 20

        if cards_area_height < 180 or main_area_width < 500:
            self.card1_container.hide()
            self.card2_container.hide()
        else:
            self.card1_container.show()
            self.card2_container.show()

            card_spacing = 30
            max_card_width = (main_area_width - card_spacing) / 2
            card_width = max_card_width
            card_height = card_width * 9 / 16

            if card_height > cards_area_height:
                card_height = cards_area_height
                card_width = card_height * 16 / 9

            final_card_width = int(card_width)
            final_card_height = int(card_height)
            cards_total_width = final_card_width * 2 + card_spacing

            cards_x_start = margin_h + (main_area_width - cards_total_width) / 2
            cards_y_start_centered = cards_area_y_start + (cards_area_height - final_card_height) / 2

            self.card1_container.setGeometry(int(cards_x_start), int(cards_y_start_centered), final_card_width,
                                             final_card_height)
            self.card2_container.setGeometry(int(cards_x_start + final_card_width + card_spacing),
                                             int(cards_y_start_centered), final_card_width, final_card_height)

        self.update()

    def showEvent(self, event: QShowEvent):
        self._cache_valid = False
        self.update()
        super().showEvent(event)


class WelcomeViewMixin:
    """Welcome page UI construction and logic"""

    def setup_welcome_page(self):
        self.welcome_page = WelcomePageWithCards(self)
        self.welcome_page.setObjectName("WelcomePageWithCardsContainer")
        self.stacked_widget.addWidget(self.welcome_page)

        self.settings_switches_container = QWidget(self.welcome_page)
        switches_layout = QHBoxLayout(self.settings_switches_container)
        switches_layout.setContentsMargins(0, 0, 0, 0)
        switches_layout.setSpacing(25)

        # Language switch button (ZH/EN)
        self.lang_toggle_btn = QPushButton("中 / EN")
        self.lang_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lang_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #2C2C2E;
                color: #A0A0A5;
                border-radius: 12px;
                padding: 4px 14px;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #3A3A3C;
            }
            QPushButton:hover { background-color: #3A3A3C; color: #FFFFFF; }
        """)
        self.lang_toggle_btn.clicked.connect(self._toggle_application_language)
        switches_layout.addWidget(self.lang_toggle_btn)

        self.hw_accel_toggle_container = self._create_hw_accel_toggle_switch()
        self.animation_toggle_container = self._create_animation_toggle_switch()

        switches_layout.addWidget(self.hw_accel_toggle_container)
        switches_layout.addWidget(self.animation_toggle_container)

        toggle_size = self.settings_switches_container.sizeHint()
        self.settings_switches_container.move(
            self.welcome_page.width() - toggle_size.width() - 20, 20
        )
        self.settings_switches_container.show()
        self.settings_switches_container.raise_()

    @Slot()
    def _toggle_application_language(self):
        """Action handler for Chinese/English language toggle"""
        current = get_app_lang()
        new_lang = "en" if current == "zh" else "zh"
        set_app_lang(new_lang)

        self.lang_toggle_btn.setText("EN / 中" if new_lang == "en" else "中 / EN")
        self.lang_toggle_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #1A73E8;
                        color: #FFFFFF;
                        border-radius: 12px;
                        padding: 4px 14px;
                        font-weight: bold;
                        font-size: 12px;
                        border: none;
                    }
                """ if new_lang == "en" else """
                    QPushButton {
                        background-color: #2C2C2E;
                        color: #A0A0A5;
                        border-radius: 12px;
                        padding: 4px 14px;
                        font-weight: bold;
                        font-size: 12px;
                        border: 1px solid #3A3A3C;
                    }
                """)

        self._translate_ui_node(self)
        self.show_status_message("Language switched to English" if new_lang == "en" else "已切换为中文", 2000)

    def _translate_ui_node(self, node):
        """Recursively scan and translate text for all child widgets"""
        if type(node).__name__ == "GradientTitleLabel":
            if hasattr(node, 'text_parts'):
                new_parts = []
                changed = False
                for i, (text, color_or_grad) in enumerate(node.text_parts):
                    orig_t = node.property(f"orig_text_{i}")
                    if orig_t is None:
                        orig_t = text
                        node.setProperty(f"orig_text_{i}", orig_t)

                    translated_t = _TR(orig_t)
                    if translated_t != text:
                        changed = True
                    new_parts.append((translated_t, color_or_grad))

                if changed:
                    node.set_text_parts(new_parts)

        elif isinstance(node, (qw.QLabel, qw.QPushButton, qw.QToolButton, qw.QCheckBox, qw.QRadioButton)):
            orig_text = node.property("orig_text")
            if orig_text is None:
                orig_text = node.text()
                if orig_text and not orig_text.isnumeric() and "0:00" not in orig_text and len(orig_text.strip()) > 0:
                    node.setProperty("orig_text", orig_text)

            if orig_text:
                translated = _TR(orig_text)
                if translated != node.text():
                    node.setText(translated)

            orig_tooltip = node.property("orig_tooltip")
            if orig_tooltip is None:
                current_tooltip = node.toolTip()
                if current_tooltip:
                    node.setProperty("orig_tooltip", current_tooltip)
                    orig_tooltip = current_tooltip

            if orig_tooltip:
                node.setToolTip(_TR(orig_tooltip))

        for child in node.findChildren(qw.QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            self._translate_ui_node(child)

    def _on_welcome_page_resize(self, event):
        if not hasattr(self, '_resize_blur_timer'):
            self._resize_blur_timer = QTimer(self)
            self._resize_blur_timer.setSingleShot(True)
            self._resize_blur_timer.setInterval(200)
            self._resize_blur_timer.timeout.connect(self._capture_and_blur_welcome_bg)
        self._resize_blur_timer.start()
        QWidget.resizeEvent(self.welcome_page, event)

    def _capture_and_blur_welcome_bg(self):
        if not self.welcome_dynamic_bg or not self.welcome_dynamic_bg.isVisible():
            return

        background_pixmap = self.welcome_dynamic_bg.grab()
        if background_pixmap.isNull():
            return

        scene = QGraphicsScene()
        item = QGraphicsPixmapItem(background_pixmap)
        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(80)
        item.setGraphicsEffect(blur_effect)
        scene.addItem(item)

        image = QImage(background_pixmap.size(), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)

        painter = QPainter(image)
        scene.render(painter)
        painter.end()

        blurred_pixmap = QPixmap.fromImage(image)
        self.welcome_blur_overlay.setPixmap(blurred_pixmap)

    def _create_welcome_entry_button(self, icon, title, subtitle, page_index):
        button = QPushButton()
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMinimumHeight(60)
        button.setObjectName("WelcomeEntryButton")
        button.setStyleSheet("""
            QPushButton#WelcomeEntryButton {
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                text-align: left;
                padding: 10px;
            }
            QPushButton#WelcomeEntryButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
                border-color: rgba(255, 255, 255, 0.15);
            }
        """)

        layout = QHBoxLayout(button)
        layout.setSpacing(12)

        icon_label = QLabel()
        icon_label.setPixmap(self._create_svg_icon(icon, size=24, color=QColor("#E0E0E0")).pixmap(24, 24))
        icon_label.setStyleSheet("background: transparent;")

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #E0E0E0; background: transparent;")

        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("font-size: 9pt; color: #888888; background: transparent;")

        text_layout.addWidget(title_label)
        text_layout.addWidget(subtitle_label)

        layout.addWidget(icon_label)
        layout.addLayout(text_layout)
        layout.addStretch(1)

        button.clicked.connect(lambda: self.switch_page_with_slide(page_index))
        return button

    def _pre_render_blur_background(self):
        if not self.welcome_page.size().isValid() or self.welcome_page.width() < 100:
            QTimer.singleShot(50, self._pre_render_blur_background)
            return

        QApplication.processEvents()

        pixmap = self.welcome_page.grab()
        if pixmap.isNull():
            return

        item = QGraphicsPixmapItem(pixmap)
        blur_effect = QGraphicsBlurEffect()
        blur_effect.setBlurRadius(60)
        item.setGraphicsEffect(blur_effect)

        scene = QGraphicsScene()
        scene.addItem(item)

        image = QImage(pixmap.size(), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)

        painter = QPainter(image)
        scene.render(painter, image.rect(), item.boundingRect())
        painter.end()

        self.welcome_page_blurred_pixmap = QPixmap.fromImage(image)